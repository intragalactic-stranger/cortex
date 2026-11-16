#!/usr/bin/env bash
# E6 rerun at commit 75d6f73, after the first preview (20261116T065339Z) turned out to have --spec-type none (D-024).
# Usage: serve/scripts/m1_e6_rerun_2026-11-16.sh <models_dir>
set -uo pipefail
cd "$(dirname "$0")/../.."
MODELS="$(realpath "$1")"
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
log "E6 RERUN at 75d6f73: first preview (20261116T065339Z) had --spec-type none (draft never ran)"
uv run servebench run serve/configs/cpu/e6_cpu_baseline.yaml --models-dir "$MODELS" || log "E6 baseline FAILED"
uv run servebench run serve/configs/cpu/e6_cpu_preview.yaml --models-dir "$MODELS" || log "E6 draft FAILED"
log "e6 rerun done"
