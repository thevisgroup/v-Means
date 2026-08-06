#!/usr/bin/env bash
VID_DIR="$(cd "$(dirname "$0")/../videos" && pwd)"
# node03 uses physical GPUs 4 and 5; vLLM sees them as local devices 0 and 1.
# num_frames 24 (not the study default 64): InternVL3.5 spends ~260 tokens per
# 448px frame with no video token compression, so 4 videos x 64 frames (~66k
# tokens) cannot fit the Qwen3-32B backbone's 40960 context. 32 frames still
# overflows on the final overall turn (~44k); 24 frames peaks at ~36k.
CUDA_VISIBLE_DEVICES=4,5 vllm serve OpenGVLab/InternVL3_5-38B \
  --tensor-parallel-size 2 \
  --trust-remote-code \
  --max-model-len 40960 \
  --limit-mm-per-prompt '{"video": 4}' \
  --media-io-kwargs '{"video": {"num_frames": 24}}' \
  --allowed-local-media-path "$VID_DIR" \
  --port 8000
