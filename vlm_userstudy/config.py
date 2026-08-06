# -*- coding: utf-8 -*-
"""
Central configuration for the VLM-as-Participant user study.

Protocol (mirrors the human study):
  One conversation per model per run:
    turn 1..4 : video_i (native video input) + per-video questions Q1-Q9
    turn 5    : overall questions Q17-Q23 (text only, all four videos in context)
"""

import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
VIDEO_DIR = os.path.join(BASE_DIR, "videos")
OUT_DIR = os.path.join(BASE_DIR, "outputs")
RAW_DIR = os.path.join(OUT_DIR, "raw")
CSV_PATH = os.path.join(OUT_DIR, "vlm_responses.csv")
PILOT_DIR = os.path.join(OUT_DIR, "pilot")
PILOT_RAW_DIR = os.path.join(PILOT_DIR, "raw")
PILOT_CSV_PATH = os.path.join(PILOT_DIR, "vlm_responses.csv")

# ---------------------------------------------------------------------------
# Stimuli — same four videos as the human study.
# q3_expected: the CORRECT Q3 answer as a string matching CLUSTER_COUNTS
#              (confirmed against the original response CSV / answer key).
# ---------------------------------------------------------------------------
VIDEOS = [
    {
        "id": "V1",
        "name": "Video 1: Blobs",
        "youtube": "https://youtu.be/HRO9I9SAnPE",
        "file": os.path.join(VIDEO_DIR, "v1_blobs.mp4"),
        "q3_expected": "3",
    },
    {
        "id": "V2",
        "name": "Video 2: Cross",
        "youtube": "https://youtu.be/XJE1sP6E7BE",
        "file": os.path.join(VIDEO_DIR, "v2_cross.mp4"),
        "q3_expected": "8",
    },
    {
        "id": "V3",
        "name": "Video 3: Aggregation",
        "youtube": "https://youtu.be/5w4qfmG87q8",
        "file": os.path.join(VIDEO_DIR, "v3_aggregation.mp4"),
        "q3_expected": "6",
    },
    {
        "id": "V4",
        "name": "Video 4: Hospital Admissions",
        "youtube": "https://youtu.be/joD1h7QhaNU",
        "file": os.path.join(VIDEO_DIR, "v4_hospital.mp4"),
        "q3_expected": "I couldn't tell",
    },
]

# Q6 scoring standards (score.py reports both):
#   "human12":  primary human-comparable result over 12 scored items;
#               "Finding empty space" is excluded, and the distractors plus
#               "Early termination" are scored No.
#   "design13": sensitivity analysis over all 13 presented items; only the
#               four distractors are scored No ("Early termination" and
#               "Finding empty space" are both scored Yes).

# ---------------------------------------------------------------------------
# Run settings. Sampling follows each vendor's recommended inference
# settings (per-model "sampling" in MODELS below) — greedy decoding
# (temperature=0) sends thinking models into endless repetition loops.
# Every request carries seed=run_id, so runs are reproducible and
# distinct; actual temperature is recorded in the CSV metadata.
# ---------------------------------------------------------------------------
N_RUNS = 3                 # independent sessions per model
MAX_TOKENS = None          # None = no request-side cap: vLLM lets the model
                           # generate until its context window (--max-model-len
                           # in serve/*.sh) is full. Thinking models need this;
                           # fixed budgets (3000/16384/32768) all starved
                           # Qwen3.5-9B's reasoning at some turn.
REQUEST_TIMEOUT = 7200     # seconds; uncapped thinking on the overall turn
                           # can run tens of minutes
NUM_FRAMES_PER_VIDEO = 64  # must match --media-io-kwargs in serve/*.sh

# ---------------------------------------------------------------------------
# Model registry. `tag` is the short name used in filenames / the Sheet.
# The runner talks to whatever OpenAI-compatible endpoint you point it at,
# so this table is metadata only — start the matching serve/*.sh first.
# ---------------------------------------------------------------------------
MODELS = {
    "qwen3vl-8b": {
        "hf_id": "Qwen/Qwen3-VL-8B-Instruct",
        "params": "8B",
        "quant": "bf16",
        "serve_script": "serve/qwen3vl_8b.sh",
        # Model card "Generation Hyperparameters", VL inference.
        "sampling": {"temperature": 0.7, "top_p": 0.8, "top_k": 20,
                     "presence_penalty": 1.5},
    },
    "glm-4.6v": {
        "hf_id": "zai-org/GLM-4.6V",
        "params": "106B-A12B",
        # BF16, not the vendor-recommended FP8: A800 (Ampere) has no native
        # FP8 and the Marlin fallback rejects GLM's MLP intermediate 10944
        # at any usable TP degree. See serve/glm46v.sh for details.
        "quant": "bf16",
        "serve_script": "serve/glm46v.sh",
        # Model card "Evaluation Settings" (matches shipped
        # generation_config.json; repetition_penalty from the card).
        "sampling": {"temperature": 0.8, "top_p": 0.6, "top_k": 2,
                     "repetition_penalty": 1.1},
        # vLLM's GLM-4V implementation hard-caps video at 1 per prompt
        # (glm4_1v.py get_supported_mm_limits). The runner therefore
        # replaces earlier videos with a text note before each new video
        # turn — the model answers overall questions from memory of its
        # own earlier answers, like a human who cannot rewatch.
        "max_videos_per_prompt": 1,
    },
    "qwen3.5-9b": {
        "hf_id": "Qwen/Qwen3.5-9B",
        "params": "9B",
        "quant": "bf16",
        "serve_script": "serve/qwen35_9b.sh",
        # Model card "Best Practices", thinking mode / general tasks
        # (repo ships no generation_config.json).
        "sampling": {"temperature": 1.0, "top_p": 0.95, "top_k": 20,
                     "min_p": 0.0, "presence_penalty": 1.5},
    },
    "internvl3.5-38b": {
        "hf_id": "OpenGVLab/InternVL3_5-38B",
        "params": "38B",
        "quant": "bf16",
        "serve_script": "serve/internvl35_38b.sh",
        # Model card OpenAI-compatible API example
        # (generation_config.json ships no sampling values).
        "sampling": {"temperature": 0.8, "top_p": 0.8},
        # InternVL has no video token compression (~260 tokens/frame), so
        # 4 videos at the default 64 frames overflow its 40960 context.
        # 24 frames keeps the worst turn at ~36k. Must match serve script.
        "num_frames": 24,
    },
    "minicpm-v-4.5": {
        "hf_id": "openbmb/MiniCPM-V-4_5",
        "params": "8B",
        "quant": "bf16",
        "serve_script": "serve/minicpmv45.sh",
        # Card gives no sampling advice; shipped generation_config.json.
        "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    },
}
