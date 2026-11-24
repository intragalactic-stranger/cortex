#!/usr/bin/env bash
# M2 E1 on L4: run the three engines one after another (one GPU job at a time, D-030).
# Each launch re-checks Modal spend + worst case against the approved gate and refuses if anything is billed to the card.
set -u
cd "$(dirname "$0")/../.."
for engine in vllm sglang llamacpp; do
  echo "=== $(date -u +%FT%TZ) E1 L4 $engine"
  uv run --extra modal modal run serve/deploy/modal/bench_app.py --config "serve/configs/gpu/e1_l4_${engine}.yaml" --yes \
    || echo "!!! $engine failed (exit $?)"
done
echo "=== $(date -u +%FT%TZ) E1 L4 finished"
