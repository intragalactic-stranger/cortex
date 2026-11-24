#!/usr/bin/env bash
# M3: produce the calibrated quantized checkpoints one after another on an L4 (each hard-capped at 60 min / $0.80).
set -u
cd "$(dirname "$0")/../.."
q() { echo "=== $(date -u +%FT%TZ) quantize $*"; uv run --extra modal modal run serve/deploy/modal/quantize_app.py "$@" --yes || echo "!!! quantize $* failed (exit $?)"; }
q --recipe w4a16-gptq-g128 --calib generic
q --recipe w4a16-awq-asym --calib generic
q --recipe w8a8-int8 --calib generic
q --recipe w4a16-gptq-g128 --calib bfsi
q --recipe w4a16-gptq-g32-sym --model gemma4-e4b --calib generic
echo "=== $(date -u +%FT%TZ) M3 quantization finished"
