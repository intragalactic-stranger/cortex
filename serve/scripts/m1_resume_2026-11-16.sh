#!/usr/bin/env bash
# Resume of m1_pipeline.sh after the 2026-11-05 run died at quality artifact 4/8 (D-023).
# Phase 4 for the 5 missing artifacts, then phase 5 (E6). Ran at commit ac60f58.
# Usage: serve/scripts/m1_resume_2026-11-16.sh <models_dir>
set -uo pipefail
cd "$(dirname "$0")/../.."
MODELS="$(realpath "$1")"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
log "RESUME 2026-11-16: phase 4 interrupted on 2026-11-05 at qwen3-1.7b-q8_0 (session ended); rerunning remaining 5 artifacts"
uv run servebench quality serve/configs/cpu/e13_llamacpp.yaml --models-dir "$MODELS" --arc-n 100 --ppl-chunks 20 \
  -a qwen3-1.7b-q8_0 -a gemma4-e2b-qat-q4_0 -a gemma4-e2b-ptq-q4_0 -a gemma4-e2b-q4_k_m -a gemma4-e2b-q8_0 || log "quality FAILED"
log "5/5 E6 CPU preview: baseline vs 0.6B draft"
uv run servebench run serve/configs/cpu/e6_cpu_baseline.yaml --models-dir "$MODELS" || log "E6 baseline FAILED"
uv run servebench run serve/configs/cpu/e6_cpu_preview.yaml --models-dir "$MODELS" || log "E6 draft FAILED"
log "pipeline done"
