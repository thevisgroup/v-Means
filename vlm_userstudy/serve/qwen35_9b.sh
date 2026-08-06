#!/usr/bin/env bash
# Qwen3.5-9B — natively multimodal (early-fusion) dense model, 262K context.
# Architecture-generation contrast pair for Qwen3-VL-8B (same family/scale,
# adapter-based vs early-fusion). Pilot in the verified vLLM 0.19.0
# environment first; upgrade only if this model actually fails to load.
# Run with: bash serve/qwen35_9b.sh   (single GPU, node03 uses GPU 3)
VID_DIR="$(cd "$(dirname "$0")/../videos" && pwd)"
CUDA_VISIBLE_DEVICES=3 vllm serve Qwen/Qwen3.5-9B \
  --reasoning-parser qwen3 \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --max-model-len 131072 \
  --limit-mm-per-prompt '{"video": 4}' \
  --media-io-kwargs '{"video": {"num_frames": 64}}' \
  --allowed-local-media-path "$VID_DIR" \
  --port 8000
