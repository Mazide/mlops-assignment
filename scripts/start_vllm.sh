#!/usr/bin/env bash
#
# Start vLLM with your chosen configuration.
# Reference: https://docs.vllm.ai/en/latest/serving/openai_compatible_server.html

set -euo pipefail

# Load .env so HF_TOKEN (and anything else) is exported for vLLM. vLLM reads
# the script's environment directly - it does NOT load .env on its own.
if [[ -f .env ]]; then
    set -a
    # shellcheck disable=SC1091
    source .env
    set +a
fi

MODEL="Qwen/Qwen3-30B-A3B-Instruct-2507"

# Flags chosen for one H100 (80GB) serving Qwen3-30B-A3B (MoE, ~60GB weights):
#   --max-model-len 32768       default is 262144 (256K) -> KV cache OOMs on
#                               start. 32K covers BIRD schema + question with
#                               headroom. Trade-off: shorter context for KV room.
#   --gpu-memory-utilization 0.90  give vLLM 90% of the card for weights + KV.
exec uv run python -m vllm.entrypoints.openai.api_server \
    --model "$MODEL" \
    --host 0.0.0.0 \
    --port 8000 \
    --max-model-len 32768 \
    --gpu-memory-utilization 0.90
