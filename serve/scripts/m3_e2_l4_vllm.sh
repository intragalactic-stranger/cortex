#!/usr/bin/env bash
# M3 E2 on L4, vLLM: each quantized format in its own container, in parallel (bf16 baseline = E1 L4 vLLM).
set -u
cd "$(dirname "$0")/../.."
for t in fp8-dynamic fp8-official w4a16-gptq w4a16-awq w8a8-int8; do
  ( uv run --extra modal modal run serve/deploy/modal/bench_app.py --config "serve/configs/gpu/e2_l4_vllm_$t.yaml" --yes \
      > "serve/results/logs/m3-e2-l4-$t.log" 2>&1 || echo "!!! $t failed" ) &
  sleep 12
done
wait
echo "=== $(date -u +%FT%TZ) E2 L4 vLLM finished"
