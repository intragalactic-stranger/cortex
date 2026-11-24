#!/usr/bin/env bash
# M4 feature experiments: one container per experiment batch, batches in parallel (D-033 reservations apply).
set -u
cd "$(dirname "$0")/../.."
C=serve/configs/gpu
run() { echo "=== $(date -u +%FT%TZ) $1"; uv run --extra modal modal run serve/deploy/modal/bench_app.py --config "$2" --yes > "serve/results/logs/m4-$1.log" 2>&1 || echo "!!! $1 failed (exit $?)"; echo "=== $(date -u +%FT%TZ) $1 done"; }
j() { ls $C/$1 | paste -sd, -; }
run e15-vllm "$(j 'e15_l4_vllm_*.yaml')" & sleep 15
run e15-sglang "$(j 'e15_l4_sglang_*.yaml')" & sleep 15
run e4-vllm "$(j 'e4_l4_vllm_*.yaml')" & sleep 15
run e4-sglang "$(j 'e4_l4_sglang_*.yaml')" & sleep 15
run e5 "$(j 'e5_*.yaml')" & sleep 15
run e6 "$(j 'e6_*.yaml')" & sleep 15
run e7 "$(j 'e7_*.yaml')" & sleep 15
run e10 "$(j 'e10_*.yaml')" & sleep 15
run e12 "$C/e12_l4x2_vllm_tp2.yaml" & sleep 15
run e9 "$C/e9_h100_vllm_qwen3_30b_a3b.yaml" &
wait
echo "=== $(date -u +%FT%TZ) M4 batches finished"
