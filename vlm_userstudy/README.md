# VLM-as-Participant: Reproducing the Clustering Visualization User Study with Open-Source VLMs

This directory is an **independent research module** in the `V-Means` repository. It
is decoupled from the Qt desktop interface and has its own dependencies, command-line
entry points, tests, model registry, and output directories. It can be copied to and
run on a GPU server on its own. The runner targets an OpenAI-compatible Chat
Completions endpoint that accepts native video content. Five formal model
configurations are included in the current experiment, and additional vision-language
models can be registered when they satisfy the same interface. Running this module does
not require starting the V-Means GUI.

This directory is an integrated snapshot of the independent `Slian22/vlm_userstudy`
repository at commit `6d62fc2fba430a02e0496fa08f4c2c4fc632bb29`. The two repositories do
not synchronize automatically. When importing a later snapshot, explicitly compare the
source code, tests, and serving scripts, then rerun the complete regression suite in
this directory.

The module treats state-of-the-art open-source vision-language models as
"participants" and reproduces the questionnaire workflow used with 212 human
participants. Results can be written to the `VLM_Responses` tab of the same Google
Spreadsheet for human-versus-model comparisons (an open-weight VLM baseline). Q6 is
identical to the human form and all 13 items are administered. The only VLM-specific
pre-registration adjustment is that open-ended questions must receive a non-empty
qualitative answer.

## Human-study-aligned protocol

Each model × run is **one conversation**. The runner presents four videos in sequence
as native video content (the processor performs frame sampling; frames are not extracted
manually), asks Q1–Q9 after each video, and then asks the text-only Q17–Q23 after all
four videos remain in the conversation. The human form's Overall page is also
video-free.

There is one model-specific exception: the GLM-4.6V vLLM implementation accepts at most
one video per prompt. Before each new video round, the runner replaces earlier video
parts with text placeholders (`max_videos_per_prompt: 1`). The model must rely on its
own previous answers, matching the human condition in which a participant cannot replay
the videos. The CSV `input_mode` column records this as
`native_video_1perprompt`.

Every model has three independent runs. Sampling uses the parameters recommended by
each model's model card (`sampling` in `config.py`), and the actual temperature is
recorded in the CSV metadata. Every request includes `seed=run_id` for reproducibility
and independence between runs. The default is 64 frames per video; InternVL3.5-38B
uses 24 frames because it has no video-token compression (about 260 tokens per frame),
so four videos × 64 frames would be roughly 66k tokens and exceed its 40960-token
context. With 24 frames, the heaviest round is about 36k tokens. The
`num_frames_per_video` CSV column records the value actually used.

Do not use `temperature=0`: greedy decoding can make thinking models repeat forever
(observed with Qwen3.5-9B).

Question wording and answer choices are copied **verbatim** from the original Google
Form (the response CSV header is checked by tests). All 13 Q6 items match the form in
both wording and order. Q17–Q19 use the anchors 1 = Strongly disagree and 5 = Strongly
agree; Q20 uses 1 = Not confident at all and 5 = Very confident. Guided decoding pins
each answer to the form's allowed option set—the numeric equivalent of a radio button,
without adding content hints. The constraint actually negotiated for each session is
recorded in the `answer_constraint` metadata column (`json_schema`,
`structured_outputs`, or `none`). Open-ended questions (Q8/Q9/Q21–Q23) require a
non-empty answer of one to three sentences. If the model sees nothing unclear or has no
improvement suggestion, it must say so explicitly and briefly explain why.

## Supported models and extension points (five-model study set)

| Tag | Weights | GPU allocation (node03) | Approx. disk | Role / rationale |
|---|---|---|---|---|
| `qwen3vl-8b` | `Qwen/Qwen3-VL-8B-Instruct` | GPU 3 | ~17G | Consumer-scale baseline |
| `qwen3.5-9b` | `Qwen/Qwen3.5-9B` | GPU 3 | ~20G | New early-fusion generation; architectural contrast with the 8B model |
| `minicpm-v-4.5` | `openbmb/MiniCPM-V-4_5` | GPU 3 | ~18G | Independent model family with video-token compression |
| `internvl3.5-38b` | `OpenGVLab/InternVL3_5-38B` | GPUs 4 and 5 | ~76G | Independent family at a mid-range scale |
| `glm-4.6v` | `zai-org/GLM-4.6V` | GPUs 4, 5, 6, and 7 | ~212G (BF16) | Flagship 106B-MoE thinking model; the A800 has no native FP8 support, and the Marlin fallback is dimension-incompatible with GLM, so the original BF16 weights are used |

The total disk requirement is approximately 260G. GLM-4.6V and Qwen3.5-9B are thinking
models; their serving scripts use `--reasoning-parser glm45` and
`--reasoning-parser qwen3`, respectively. `config.py` sets `MAX_TOKENS=None`, so the
request does not impose a token cap: reasoning plus the answer can use the full model
context window (`--max-model-len` in the serving script). Use each model's vendor
sampling recommendations from `config.py`. With `temperature=0`, a thinking model can
repeat until the context window is exhausted (`empty content (finish_reason=length)`).

`serve/glm45v.sh` and `serve/qwen3vl_235b_fp8.sh` are historical reference scripts. They
are not registered in `config.MODELS` and are not part of the five-model experiment;
do not run them as formal study conditions.

The five entries are the fixed comparison group for the current experiment, not a limit
on the runner's interface. To add a model, add its tag, model ID, sampling parameters,
and serving script to `config.py`'s `MODELS` registry. The corresponding service must
provide an OpenAI-compatible endpoint with native video support. Then run a `--dry-run`,
the regression suite, and an isolated pilot before collecting formal runs. Keep
model-specific frame limits and per-prompt video limits in the registry rather than
scattering them through `runner.py`. Server-side `vllm` is intentionally omitted from
this module's client `requirements.txt`; install it separately on the GPU environment
for the selected model and hardware.

## Security and operational boundary

The `serve/*.sh` files are research launchers for reproducible runs on a trusted GPU
node, not hardened public inference services. Do not expose vLLM port 8000 to the public
Internet or an untrusted shared network. The runner is intended to access the server on
the same node through `localhost`. For cross-machine runs, prefer an SSH tunnel and
configure firewall rules, access control, and transport encryption. When supported by
the installed vLLM version, explicitly bind the service to a loopback address.

The InternVL and MiniCPM-V scripts require `--trust-remote-code`. Before a formal run,
review and pin immutable Hugging Face model/code revisions, and run in a dedicated,
low-privilege environment with no extra credentials. Do not place a Google service-
account key, an HF token, or an SSH agent in the model-serving process. The serving
scripts retain parameters validated in the source research environment, so operators
must enforce isolation and revision controls at the server boundary.

Runner command-line arguments are trusted operator input. Do not pass web requests, job
names, or other external strings directly to the CLI. Automation must whitelist `--tag`
against the formal tags in `config.MODELS` and restrict `--base-url` to the local host or
a controlled tunnel. On a shared node, run `umask 077` first to limit default
transcript and CSV permissions. Model-generated open-ended text is untrusted data; do
not open the CSV directly in a spreadsheet application that evaluates formulas. Review
the data through the study's publication workflow before sharing it.

The Google service-account JSON key must be stored outside this repository and readable
only by the current user. Google Sheets Editor permission applies to the entire workbook,
not just the `VLM_Responses` tab. If the human-study data is sensitive, write to a
separate VLM-output workbook first and merge it under controlled access. `--replace`
clears and rewrites the target tab; manually verify the spreadsheet and tab before using
that option.

## Verified node03 environment

The existing `vlmstudy` environment on the server has vLLM 0.19.0 and has completed a
Qwen3-VL-8B pilot. Do not rebuild the environment preemptively for another model. Run a
pilot for each new model; upgrade or create a model-specific environment only when the
server reports that an architecture is unsupported.

```bash
cd /path/to/v-Means/vlm_userstudy
conda activate vlmstudy
python -m pip install -r requirements.txt
python -c "import vllm, openai; print('vLLM', vllm.__version__)"
python -m unittest discover -s tests -v
```

The four videos must be present under `videos/` with exactly these filenames:

```text
videos/v1_blobs.mp4
videos/v2_cross.mp4
videos/v3_aggregation.mp4
videos/v4_hospital.mp4
```

The integrated repository does not track these large MP4 files. Before the first run,
download them with `bash download_videos.sh`. Downloaded files and partial downloads
under `videos/` are ignored except for `.gitkeep`, so they cannot be committed by
mistake.

Formal results are written to `outputs/vlm_responses.csv`; per-round audit transcripts
are written to `outputs/raw/`; pilots write only to `outputs/pilot/`. The entire
`outputs/` directory is ignored by Git and is not pushed to GitHub.

The current schema contains all 13 Q6 items and has 107 CSV columns. A pilot produced
with the old 12-item schema has 103 columns. Rename the old `outputs/pilot/` directory
for archival before running a new pilot; the runner actively refuses to mix the two
schemas.

## Execution loop for each model

```bash
# Serving window (tmux):
bash serve/<model>.sh                    # The node03 GPU assignment is in each script.
# Runner window:
curl -f http://localhost:8000/v1/models  # Continue only after the server is ready.
python runner.py --tag <tag> --pilot     # Required for each new model; isolated in outputs/pilot/.
python score.py outputs/pilot/vlm_responses.csv
# Inspect raw transcripts, the CSV, format rates, and warnings. There is no preset
# pass/fail threshold; use the model's actual answers to decide whether to run formally:
REV=$(python3 -c "from huggingface_hub import HfApi; print(HfApi().model_info('<hf_id>').sha)")
python runner.py --tag <tag> --model-revision $REV      # Three formal runs.
# Resume after repairing a failed run: --start-run-id <N>
# Change models: stop vLLM, switch serving scripts, and repeat.
```

After all runs finish, use `python score.py` to view the summary table. Copy the results
back to a local machine and run `push_to_sheet.py` to write the Sheet; the service-
account key must not be placed on the GPU server.

## Scoring conventions (pre-registered)

- Q3 uses `q3_expected` from `config.py` (`3 / 8 / 6 / I couldn't tell`); the correct
  answer for Q4 is `False`.
- The primary Q6 result is **human12**: administer all 13 items, but exclude `Finding
  empty space` from the scored 12-item subset and mark `Early termination` as No. Use
  the same 12-item subset when comparing with human data. **design13** is a sensitivity
  analysis over all 13 administered items; only the four distractors are marked No,
  while both `Early termination` and `Finding empty space` are marked Yes.
- Missing or invalid answers use valid-only denominators. Report format adherence
  separately in the `videoFmt` and `overallFmt` columns.

## Reliability behavior

- Each request retries transient connection, timeout, 5xx, or 429 errors up to three
  times with 30/60-second waits. Deterministic 4xx errors fail immediately without an
  artificial wait. If the server rejects the guided-decoding grammar, the runner
  automatically falls back and records the downgrade.
- Each round is persisted to a transcript. A failed session produces no CSV row. Dry
  runs, pilots, and formal data use separate output locations.
- The CSV schema guard refuses to mix old and new headers. Sheet publishing appends by
  default with `(model_tag, run_id, timestamp)` de-duplication and permits manually
  added trailing columns; only `--replace` rewrites the entire tab.
- Reproducibility metadata includes `served_model`, `model_revision`, `vllm_version`,
  `gpu_name`, `serve_script`, and `answer_constraint`.

## Directory layout

```
README.md         Independent-module overview, study protocol, and runbook
requirements.txt  Client / Sheet / download-tool dependencies (no server-side vLLM)
config.py         Video paths, q3_expected, run parameters, and model registry
questionnaire.py  Verbatim questionnaire, choices, prompts, JSON schemas, and validators
runner.py         Multi-round runner (--pilot / --dry-run / --start-run-id / --model-revision)
score.py          Dual-convention scoring and format rates
push_to_sheet.py  Write the VLM_Responses tab (run locally)
serve/*.sh        Five formal-model launchers plus two explicitly marked legacy scripts
tests/            Regression tests (python -m unittest discover -s tests -v)
download_videos.sh  Download the four configured videos with yt-dlp
videos/.gitkeep   Keep the media directory; MP4s are acquired separately by download_videos.sh
```
