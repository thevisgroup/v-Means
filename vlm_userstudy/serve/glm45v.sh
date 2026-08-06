#!/usr/bin/env bash
# LEGACY/NOT IN THE STUDY: retained only for reference. Use glm46v.sh.
VID_DIR="$(cd "$(dirname "$0")/../videos" && pwd)"
vllm serve zai-org/GLM-4.5V \
  --tensor-parallel-size 4 \
  --max-model-len 65536 \
  --limit-mm-per-prompt '{"video": 4}' \
  --media-io-kwargs '{"video": {"num_frames": 64}}' \
  --allowed-local-media-path "$VID_DIR" \
  --port 8000
