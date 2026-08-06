#!/usr/bin/env bash
# LEGACY/NOT IN THE STUDY: retained only for reference; not in config.MODELS.
# NOTE: on some vLLM versions --limit-mm-per-prompt takes 'video=4' instead of JSON.
VID_DIR="$(cd "$(dirname "$0")/../videos" && pwd)"
vllm serve Qwen/Qwen3-VL-235B-A22B-Instruct-FP8 \
  --tensor-parallel-size 8 \
  --mm-encoder-tp-mode data \
  --enable-expert-parallel \
  --max-model-len 131072 \
  --limit-mm-per-prompt '{"video": 4}' \
  --media-io-kwargs '{"video": {"num_frames": 64}}' \
  --allowed-local-media-path "$VID_DIR" \
  --port 8000
