#!/usr/bin/env bash
# Small-scale comparison point: single GPU (node03 GPU 3).
VID_DIR="$(cd "$(dirname "$0")/../videos" && pwd)"
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3-VL-8B-Instruct \
  --max-model-len 131072 \
  --limit-mm-per-prompt '{"video": 4}' \
  --media-io-kwargs '{"video": {"num_frames": 64}}' \
  --allowed-local-media-path "$VID_DIR" \
  --port 8000
