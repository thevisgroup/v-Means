# -*- coding: utf-8 -*-
"""
VLM-as-Participant runner.

One conversation per run:
  turn 1..4 : video_i (native video via video_url content part) + Q1-Q9
  turn 5    : Q17-Q23 (text only; all four videos remain in context)

Talks to any OpenAI-compatible endpoint (vLLM `serve`).  Start the matching
script in serve/ first, then e.g.:

  python runner.py --tag qwen3vl-8b \
      --base-url http://localhost:8000/v1 \
      --served-model Qwen/Qwen3-VL-8B-Instruct

Outputs:
  outputs/raw/<tag>_run<i>_<ts>.json   full transcript (audit trail)
  outputs/vlm_responses.csv            one flat row per session, form-shaped
"""

import argparse
import csv
import datetime as dt
import importlib.metadata
import json
import os
import subprocess
import sys
import time
from collections import Counter

from openai import OpenAI

import config as C
import questionnaire as Q

MAX_RETRIES = 3
BACKOFF_BASE = 30  # seconds between attempts: 30, 60


# ------------------------------ CSV layout ---------------------------------

METADATA_COLS = [
    "timestamp", "model_tag", "model_hf_id", "served_model",
    "model_revision", "params", "quant", "input_mode", "answer_constraint",
    "num_frames_per_video", "temperature", "run_id", "vllm_version",
    "gpu_name", "serve_script",
]


def build_header():
    cols = list(METADATA_COLS)
    for v in C.VIDEOS:
        vid = v["id"]
        for qk in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            cols.append(f"{vid}_{qk}")
        for step in Q.ALGORITHM_STEPS:
            cols.append(f"{vid}_Q6 [{step}]")
        for qk in ["Q7", "Q8", "Q9"]:
            cols.append(f"{vid}_{qk}")
    for qk in ["Q17", "Q18", "Q19", "Q20", "Q21", "Q22", "Q23"]:
        cols.append(qk)
    cols.append("validation_warnings")
    return cols


def flatten_row(meta, per_video_answers, overall_answers, warnings):
    row = dict(meta)
    for v in C.VIDEOS:
        vid = v["id"]
        a = per_video_answers.get(vid, {})
        for qk in ["Q1", "Q2", "Q3", "Q4", "Q5"]:
            row[f"{vid}_{qk}"] = a.get(qk, "")
        q6 = a.get("Q6", {})
        if not isinstance(q6, dict):
            q6 = {}
        for step in Q.ALGORITHM_STEPS:
            row[f"{vid}_Q6 [{step}]"] = q6.get(step, "")
        for qk in ["Q7", "Q8", "Q9"]:
            row[f"{vid}_{qk}"] = a.get(qk, "")
    for qk in ["Q17", "Q18", "Q19", "Q20", "Q21", "Q22", "Q23"]:
        row[qk] = overall_answers.get(qk, "")
    row["validation_warnings"] = " ; ".join(warnings)
    return row


def append_csv(row, path=None):
    path = path or C.CSV_PATH
    header = build_header()
    exists = os.path.exists(path)
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if exists:
        with open(path, newline="", encoding="utf-8") as f:
            existing_header = next(csv.reader(f), [])
        if existing_header != header:
            raise RuntimeError(
                f"existing CSV schema differs from the current questionnaire: {path}. "
                "Move or rename the old CSV before running this version.")
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=header)
        if not exists:
            w.writeheader()
        w.writerow(row)


# ------------------------------ session logic ------------------------------

def video_content_part(path):
    """Native video input. vLLM must be started with
    --allowed-local-media-path pointing at the videos directory."""
    return {"type": "video_url",
            "video_url": {"url": "file://" + os.path.abspath(path)}}


VIDEO_REMOVED_NOTE = ("[The video you watched in this turn is no longer "
                      "shown. Answer from memory and from your earlier "
                      "answers, like a participant who cannot rewatch it.]")


def strip_video_parts(messages):
    """Return a copy of `messages` with video content parts replaced by a
    text note. For models whose vLLM implementation allows at most one
    video per prompt (e.g. GLM-4V family), earlier videos must leave the
    context before the next one is sent; the human analogue is a
    participant who cannot rewatch previous videos."""
    stripped = []
    for msg in messages:
        content = msg.get("content")
        if isinstance(content, list):
            parts = [{"type": "text", "text": VIDEO_REMOVED_NOTE}
                     if part.get("type") == "video_url" else part
                     for part in content]
            stripped.append({**msg, "content": parts})
        else:
            stripped.append(msg)
    return stripped


def _is_retryable_error(error):
    """Retry connection/timeouts, HTTP 429, and HTTP 5xx only."""
    status = getattr(error, "status_code", None)
    if status is not None:
        return status == 429 or 500 <= status < 600
    if isinstance(error, (ConnectionError, TimeoutError)):
        return True
    return type(error).__name__ in {
        "APIConnectionError", "APITimeoutError", "RateLimitError",
    }


# Structured-output modes, tried in order until the server accepts one.
# "json_schema" = OpenAI response_format syntax, "structured_outputs" =
# current vLLM native extra_body syntax, "none" = unconstrained (validators
# still catch strays). vLLM removed the legacy guided_json field in v0.12.
STRUCTURED_MODES = ["json_schema", "structured_outputs", "none"]


def _request_kwargs(schema, mode):
    if schema is None or mode == "none":
        return {}
    if mode == "json_schema":
        return {"response_format": {
            "type": "json_schema",
            "json_schema": {"name": "questionnaire", "strict": True,
                            "schema": schema}}}
    if mode == "structured_outputs":
        return {"extra_body": {"structured_outputs": {"json": schema}}}
    raise ValueError(mode)


# Sampling fields the OpenAI SDK accepts natively; anything else
# (top_k, min_p, repetition_penalty, ...) rides in vLLM's extra_body.
NATIVE_SAMPLING_KEYS = {"temperature", "top_p", "presence_penalty",
                        "frequency_penalty"}


def _sampling_kwargs(sampling, seed):
    native, extra = {}, {}
    for key, value in (sampling or {}).items():
        if key in NATIVE_SAMPLING_KEYS:
            native[key] = value
        else:
            extra[key] = value
    if seed is not None:
        native["seed"] = seed
    return native, extra


def chat(client, served_model, messages, schema=None, constraint_state=None,
         sampling=None, seed=None):
    """One turn with retries; raises after MAX_RETRIES failures.

    When `schema` is given, answers are constrained by server-side guided
    decoding (the digital radio button).  The first syntax the server
    accepts is remembered in `constraint_state` for the whole session; a
    400/422 while a structured mode is active falls back to the next mode
    without consuming a transient-retry attempt.
    """
    state = constraint_state if constraint_state is not None else {}
    if not state.get("mode"):
        state["mode"] = STRUCTURED_MODES[0] if schema is not None else "none"
    last_err = None
    attempt = 0
    while attempt < MAX_RETRIES:
        attempt += 1
        try:
            native_sampling, extra_sampling = _sampling_kwargs(sampling, seed)
            request_kwargs = dict(
                model=served_model,
                messages=messages,
                timeout=C.REQUEST_TIMEOUT,
                **native_sampling,
                **_request_kwargs(schema, state["mode"]),
            )
            if C.MAX_TOKENS is not None:
                request_kwargs["max_tokens"] = C.MAX_TOKENS
            if extra_sampling:
                request_kwargs.setdefault("extra_body", {}).update(
                    extra_sampling)
            resp = client.chat.completions.create(**request_kwargs)
            content = resp.choices[0].message.content
            if content is None:
                # Reasoning models return content=None when generation
                # dies mid-thinking; deterministic at T=0, so fail the
                # turn instead of retrying or feeding None downstream.
                finish = getattr(resp.choices[0], "finish_reason", None)
                budget = (f"max_tokens={C.MAX_TOKENS}"
                          if C.MAX_TOKENS is not None
                          else "the serve context window (--max-model-len)")
                raise RuntimeError(
                    f"model returned empty content "
                    f"(finish_reason={finish}); thinking likely consumed "
                    f"{budget}")
            return content
        except Exception as e:
            last_err = e
            status = getattr(e, "status_code", None)
            if (status in {400, 422} and schema is not None
                    and state["mode"] != "none"):
                nxt = STRUCTURED_MODES[
                    STRUCTURED_MODES.index(state["mode"]) + 1]
                print(f"  structured-output mode '{state['mode']}' rejected "
                      f"by server ({e}) -> falling back to '{nxt}'",
                      flush=True)
                state["mode"] = nxt
                attempt -= 1
                continue
            if not _is_retryable_error(e) or attempt == MAX_RETRIES:
                break
            wait = BACKOFF_BASE * (2 ** (attempt - 1))
            print(f"  attempt {attempt}/{MAX_RETRIES} failed: {e} "
                  f"-> retrying in {wait}s", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"turn failed after {attempt} attempt(s): {last_err}")


def save_transcript(path, transcript):
    """Persist after EVERY turn so a crashed session keeps partial work."""
    with open(path, "w", encoding="utf-8") as f:
        json.dump(transcript, f, ensure_ascii=False, indent=2)


def collect_runtime_metadata():
    """Best-effort server reproducibility metadata; never blocks a run."""
    try:
        vllm_version = importlib.metadata.version("vllm")
    except importlib.metadata.PackageNotFoundError:
        vllm_version = ""
    try:
        proc = subprocess.run(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
            check=True, capture_output=True, text=True, timeout=10)
        gpu_names = [line.strip() for line in proc.stdout.splitlines()
                     if line.strip()]
        counts = Counter(gpu_names)
        gpu_name = " | ".join(f"{counts[name]}x {name}"
                              for name in dict.fromkeys(gpu_names))
    except (FileNotFoundError, subprocess.SubprocessError):
        gpu_name = ""
    return {"vllm_version": vllm_version, "gpu_name": gpu_name}


def run_session(client, served_model, tag, run_id, dry=False, pilot=False,
                model_revision="", runtime_meta=None):
    if dry and pilot:
        raise ValueError("dry-run and pilot are mutually exclusive")
    meta_model = C.MODELS.get(tag, {})
    sampling = meta_model.get("sampling", {})
    single_video = meta_model.get("max_videos_per_prompt") == 1
    runtime_meta = runtime_meta or {}
    ts = dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds")
    meta = {
        "timestamp": ts,
        "model_tag": tag,
        "model_hf_id": meta_model.get("hf_id", served_model),
        "served_model": served_model,
        "model_revision": model_revision,
        "params": meta_model.get("params", ""),
        "quant": meta_model.get("quant", ""),
        "input_mode": ("native_video_1perprompt" if single_video
                       else "native_video"),
        "answer_constraint": "",
        "num_frames_per_video": meta_model.get("num_frames",
                                               C.NUM_FRAMES_PER_VIDEO),
        "temperature": sampling.get("temperature", ""),
        "run_id": run_id,
        "vllm_version": runtime_meta.get("vllm_version", ""),
        "gpu_name": runtime_meta.get("gpu_name", ""),
        "serve_script": meta_model.get("serve_script", ""),
    }

    messages = [{"role": "system", "content": Q.SYSTEM_PROMPT}]
    constraint_state = {}  # structured-output mode, negotiated on first turn
    transcript = {"meta": meta, "status": "running", "turns": []}
    per_video_answers, warnings = {}, []

    raw_dir = C.PILOT_RAW_DIR if pilot else C.RAW_DIR
    csv_path = C.PILOT_CSV_PATH if pilot else C.CSV_PATH
    os.makedirs(raw_dir, exist_ok=True)
    suffix = "_dryrun" if dry else ""
    raw_path = os.path.join(
        raw_dir,
        f"{tag}_run{run_id}_{ts.replace(':', '-')}{suffix}.json")
    save_transcript(raw_path, transcript)

    for v in C.VIDEOS:
        if single_video:
            messages = strip_video_parts(messages)
        prompt = Q.per_video_prompt(v["name"])
        user_msg = {"role": "user",
                    "content": [video_content_part(v["file"]),
                                {"type": "text", "text": prompt}]}
        messages.append(user_msg)
        print(f"[{tag} run{run_id}] {v['id']} -> asking Q1-Q9 ...",
              flush=True)
        if dry:
            reply = "{}"
        else:
            try:
                reply = chat(client, served_model, messages,
                             schema=Q.per_video_schema(),
                             constraint_state=constraint_state,
                             sampling=sampling, seed=run_id)
            except RuntimeError as e:
                transcript["status"] = f"failed at {v['id']}: {e}"
                save_transcript(raw_path, transcript)
                print(f"[{tag} run{run_id}] SESSION FAILED at {v['id']} — "
                      f"partial transcript kept at {raw_path}; "
                      f"no CSV row written.", flush=True)
                return False
        messages.append({"role": "assistant", "content": reply})
        transcript["turns"].append({"video": v["id"], "prompt": prompt,
                                    "reply": reply})
        save_transcript(raw_path, transcript)
        try:
            ans = Q.parse_json_reply(reply)
        except Exception as e:  # keep going; raw reply is logged
            ans = {}
            warnings.append(f"{v['id']}: JSON parse failed ({e})")
        else:
            try:
                warnings += [f"{v['id']}: {w}"
                             for w in Q.validate_per_video(ans)]
            except Exception as e:
                warnings.append(f"{v['id']}: validation failed ({e})")
        per_video_answers[v["id"]] = ans

    # final overall turn — text only, memory of all four videos
    if single_video:
        messages = strip_video_parts(messages)
    prompt = Q.overall_prompt()
    messages.append({"role": "user", "content": prompt})
    print(f"[{tag} run{run_id}] overall -> asking Q17-Q23 ...", flush=True)
    if dry:
        reply = "{}"
    else:
        try:
            reply = chat(client, served_model, messages,
                         schema=Q.overall_schema(),
                         constraint_state=constraint_state,
                         sampling=sampling, seed=run_id)
        except RuntimeError as e:
            transcript["status"] = f"failed at overall turn: {e}"
            save_transcript(raw_path, transcript)
            print(f"[{tag} run{run_id}] SESSION FAILED at overall turn — "
                  f"partial transcript kept at {raw_path}; "
                  f"no CSV row written.", flush=True)
            return False
    transcript["turns"].append({"video": "overall", "prompt": prompt,
                                "reply": reply})
    try:
        overall = Q.parse_json_reply(reply)
    except Exception as e:
        overall = {}
        warnings.append(f"overall: JSON parse failed ({e})")
    else:
        try:
            warnings += [f"overall: {w}"
                         for w in Q.validate_overall(overall)]
        except Exception as e:
            warnings.append(f"overall: validation failed ({e})")

    meta["answer_constraint"] = (
        "dry_run" if dry else constraint_state.get("mode", "none"))
    transcript["status"] = "dry_run" if dry else ("pilot" if pilot else "complete")
    transcript["validation_warnings"] = warnings
    save_transcript(raw_path, transcript)

    if dry:
        print(f"[{tag} run{run_id}] dry-run done. Prompts logged to "
              f"{raw_path} — NOT written to the CSV.", flush=True)
        return True

    row = flatten_row(meta, per_video_answers, overall, warnings)
    append_csv(row, csv_path)
    print(f"[{tag} run{run_id}] done. raw -> {raw_path}", flush=True)
    if warnings:
        print(f"[{tag} run{run_id}] WARNINGS:\n  " + "\n  ".join(warnings),
              flush=True)
    return True


# ---------------------------------- main -----------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True,
                    help="short model tag, key in config.MODELS")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--served-model", default=None,
                    help="model name as served by vLLM "
                         "(default: hf_id from config.MODELS[tag])")
    ap.add_argument("--runs", type=int, default=C.N_RUNS)
    ap.add_argument("--start-run-id", type=int, default=1,
                    help="first run_id (useful when replacing a failed run)")
    ap.add_argument("--dry-run", action="store_true",
                    help="build prompts without calling the endpoint")
    ap.add_argument("--pilot", action="store_true",
                    help="run one real session into outputs/pilot only")
    ap.add_argument("--model-revision", default=os.environ.get(
                    "MODEL_REVISION", ""),
                    help="Hugging Face commit SHA/revision recorded in metadata")
    args = ap.parse_args()

    if args.dry_run and args.pilot:
        ap.error("--dry-run and --pilot cannot be used together")
    if args.runs < 1 or args.start_run_id < 1:
        ap.error("--runs and --start-run-id must be positive")

    served = args.served_model or C.MODELS.get(args.tag, {}).get("hf_id")
    if not served:
        sys.exit("unknown tag and no --served-model given")

    missing = [v["file"] for v in C.VIDEOS if not os.path.exists(v["file"])]
    if missing and not args.dry_run:
        sys.exit("missing video files (run download_videos.sh first):\n  "
                 + "\n  ".join(missing))

    # Disable the SDK's hidden retry layer; chat() owns the exact policy.
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", max_retries=0)
    n_runs = 1 if (args.dry_run or args.pilot) else args.runs
    runtime_meta = collect_runtime_metadata()
    ok = 0
    first_run_id = 1 if (args.dry_run or args.pilot) else args.start_run_id
    for run_id in range(first_run_id, first_run_id + n_runs):
        if run_session(client, served, args.tag, run_id,
                       dry=args.dry_run, pilot=args.pilot,
                       model_revision=args.model_revision,
                       runtime_meta=runtime_meta):
            ok += 1
    print(f"finished: {ok}/{n_runs} sessions completed", flush=True)


if __name__ == "__main__":
    main()
