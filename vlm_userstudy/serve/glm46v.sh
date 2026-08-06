#!/usr/bin/env bash
# GLM-4.6V (106B-A12B MoE), BF16 original weights — NOT the FP8 repo.
# The official recipe (https://recipes.vllm.ai/zai-org/GLM-4.6V) recommends
# FP8, but that assumes Hopper. A800 is Ampere with no native FP8, so vLLM
# routes FP8 through the Marlin weight-only kernel, and GLM's MLP
# intermediate size 10944 violates Marlin's 64/128 tile alignment at every
# TP degree we tried: TP=4 dies repacking gate_up N=5472 (zai-org/GLM-V#178),
# TP=2 dies at runtime on down_proj K=5472 (same class as vllm#38022).
# BF16 uses standard GEMMs — no alignment constraint. ~212GB weights need
# all 4 GPUs (TP=4, ~53GB/GPU).
# THINKING model: --reasoning-parser glm45 strips reasoning from
# message.content so the runner receives only the final JSON answer.
# Thinking is ON by default; if the pilot shows truncated JSON, first raise
# MAX_TOKENS in config.py, then consider disabling thinking via
# extra_body={"chat_template_kwargs": {"enable_thinking": false}}.
# node03 uses physical GPUs 4,5,6,7; vLLM sees local devices 0..3.
VID_DIR="$(cd "$(dirname "$0")/../videos" && pwd)"
CUDA_VISIBLE_DEVICES=4,5,6,7 vllm serve zai-org/GLM-4.6V \
  --tensor-parallel-size 4 \
  --enable-expert-parallel \
  --reasoning-parser glm45 \
  --tool-call-parser glm45 \
  --enable-auto-tool-choice \
  --mm-encoder-tp-mode data \
  --mm-processor-cache-type shm \
  --max-model-len 65536 \
  --limit-mm-per-prompt '{"video": 1}' \
  --media-io-kwargs '{"video": {"num_frames": 64}}' \
  --allowed-local-media-path "$VID_DIR" \
  --port 8000
