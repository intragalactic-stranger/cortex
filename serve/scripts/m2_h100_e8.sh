#!/usr/bin/env bash
# M2 after E1 L4: H100 subset (vLLM, SGLang), then E8 memory knobs (8 vLLM configs in ONE L4 container).
# Waits for the E1 L4 script so only one GPU job runs at a time (D-030).
set -u
cd "$(dirname "$0")/../.."
while pgrep -f m2_e1_l4.sh >/dev/null; do sleep 30; done
run() { echo "=== $(date -u +%FT%TZ) $1"; uv run --extra modal modal run serve/deploy/modal/bench_app.py --config "$2" --yes || echo "!!! $1 failed (exit $?)"; }
run "E1 H100 vllm" serve/configs/gpu/e1_h100_vllm.yaml
run "E1 H100 sglang" serve/configs/gpu/e1_h100_sglang.yaml
run "E8 L4 vllm (8 configs)" "$(ls serve/configs/gpu/e8_l4_vllm_*.yaml | paste -sd, -)"
echo "=== $(date -u +%FT%TZ) M2 H100 + E8 finished"
