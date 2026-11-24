#!/usr/bin/env bash
# Rerun E1 L4 llama.cpp after the path fix; waits for the H100 + E8 queue (one GPU job at a time, D-030).
set -u
cd "$(dirname "$0")/../.."
while pgrep -f m2_h100_e8.sh >/dev/null; do sleep 30; done
echo "=== $(date -u +%FT%TZ) E1 L4 llamacpp (rerun)"
uv run --extra modal modal run serve/deploy/modal/bench_app.py --config serve/configs/gpu/e1_l4_llamacpp.yaml --yes || echo "!!! llamacpp failed (exit $?)"
echo "=== $(date -u +%FT%TZ) llama.cpp rerun finished"
