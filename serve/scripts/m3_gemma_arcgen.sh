#!/usr/bin/env bash
# Gemma 4 E4B: reasoning-first ARC (n=500, our scorer) for bf16, Google QAT, our GPTQ PTQ and RTN (L40S, parallel).
set -u
cd "$(dirname "$0")/../.."
TOK=google/gemma-4-E4B-it
EXTRA='--limit-mm-per-prompt {"image":0,"audio":0} --default-chat-template-kwargs {"enable_thinking":false}'
ev() { uv run --extra modal modal run serve/deploy/modal/eval_app.py --name "$1" --model "$2" ${3:+--revision $3} --parts arcgen --tokenizer "$TOK" --gpu L40S --extra "$EXTRA" --yes > "serve/results/logs/m3-arcgen-$1.log" 2>&1 || echo "!!! $1 failed"; echo "=== $1 done"; }
ev gemma4-e4b-bf16 google/gemma-4-E4B-it 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c & sleep 15
ev gemma4-e4b-qat-w4a16-g32 google/gemma-4-E4B-it-qat-w4a16-ct 6cd26aaa2357fb2bad8c51699a7558a4d1a965bb & sleep 15
ev gemma4-e4b-ptq-w4a16-g32 /hf/artifacts/gemma4-e4b-w4a16-gptq-g32-sym-generic & sleep 15
ev gemma4-e4b-rtn-w4a16-g32 /hf/artifacts/gemma4-e4b-w4a16-rtn-g32-sym &
wait
echo "=== $(date -u +%FT%TZ) Gemma arcgen finished"
