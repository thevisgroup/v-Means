#!/usr/bin/env bash
# MiniCPM-V 4.5 (8B). Pilot multi-video multi-turn first — if it degrades,
# note it in the paper or drop to the transformers adapter. node03 uses GPU 3.
VID_DIR="$(cd "$(dirname "$0")/../videos" && pwd)"
CUDA_VISIBLE_DEVICES=3 vllm serve openbmb/MiniCPM-V-4_5 \
  --trust-remote-code \
  --max-model-len 40960 \
  --limit-mm-per-prompt '{"video": 4}' \
  --media-io-kwargs '{"video": {"num_frames": 64}}' \
  --allowed-local-media-path "$VID_DIR" \
  --port 8000
