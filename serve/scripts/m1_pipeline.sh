#!/usr/bin/env bash
# M1 measurement pipeline: runs every CPU experiment sequentially (never concurrently: they share 4 threads).
# Usage: serve/scripts/m1_pipeline.sh <models_dir>
set -uo pipefail
cd "$(dirname "$0")/../.."
MODELS="$(realpath "$1")"
OUT=serve/results
STAMP=$(date -u +%Y%m%dT%H%M%SZ)
TOOLS=ghcr.io/ggml-org/llama.cpp:full@sha256:93573d882ea21d1b77712a33162934daaabcdf3a076dfff91a9ed3c2aed8cd41
log() { echo "[$(date -u +%H:%M:%S)] $*"; }

log "1/5 llama-bench (pp128, pp512, tg64; 3 reps; 4 threads)"
mkdir -p "$OUT/e13-llama-bench/$STAMP"
MODEL_ARGS=$(find "$MODELS" -name '*.gguf' | sort | sed "s#^$MODELS#/models#" | paste -sd, -)
docker run --rm -v "$MODELS:/models:ro" --entrypoint /app/llama-bench "$TOOLS" \
  -m "$MODEL_ARGS" -t 4 -p 128,512 -n 64 -r 3 -o json > "$OUT/e13-llama-bench/$STAMP/llama-bench.json" \
  2> "$OUT/e13-llama-bench/$STAMP/llama-bench.stderr" || log "llama-bench FAILED"
docker image inspect --format '{{index .RepoDigests 0}}' "$TOOLS" > "$OUT/e13-llama-bench/$STAMP/image.txt"

log "2/5 servebench E13 llama.cpp (8 artifacts)"
uv run servebench run serve/configs/cpu/e13_llamacpp.yaml --models-dir "$MODELS" || log "E13 llama.cpp FAILED"
log "3/5 servebench E13 Ollama (2 artifacts)"
uv run servebench run serve/configs/cpu/e13_ollama.yaml --models-dir "$MODELS" || log "E13 Ollama FAILED"
log "4/5 quality: WikiText-2 PPL (20x512) + ARC-Challenge (n=100), 8 artifacts"
uv run servebench quality serve/configs/cpu/e13_llamacpp.yaml --models-dir "$MODELS" --arc-n 100 --ppl-chunks 20 || log "quality FAILED"
log "5/5 E6 CPU preview: baseline vs 0.6B draft"
uv run servebench run serve/configs/cpu/e6_cpu_baseline.yaml --models-dir "$MODELS" || log "E6 baseline FAILED"
uv run servebench run serve/configs/cpu/e6_cpu_preview.yaml --models-dir "$MODELS" || log "E6 draft FAILED"
log "pipeline done"
