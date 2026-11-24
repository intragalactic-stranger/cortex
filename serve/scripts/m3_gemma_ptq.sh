#!/usr/bin/env bash
# Gemma 4 E4B PTQ (GPTQ W4A16 g32 sym, Google's QAT scheme) then its quality eval on L40S.
set -u
cd "$(dirname "$0")/../.."
echo "=== $(date -u +%FT%TZ) quantize gemma4-e4b w4a16-gptq-g32-sym"
uv run --extra modal modal run serve/deploy/modal/quantize_app.py --recipe w4a16-gptq-g32-sym --model gemma4-e4b --calib generic --yes || { echo "!!! gemma quantize failed"; exit 1; }
a=gemma4-e4b-w4a16-gptq-g32-sym-generic
EXTRA='--limit-mm-per-prompt {"image":0,"audio":0} --default-chat-template-kwargs {"enable_thinking":false}'
echo "=== $(date -u +%FT%TZ) eval gemma4-e4b-ptq-w4a16-g32"
uv run --extra modal modal run serve/deploy/modal/eval_app.py --name gemma4-e4b-ptq-w4a16-g32 --model "/hf/artifacts/$a" --tokenizer google/gemma-4-E4B-it --gpu L40S --extra "$EXTRA" --yes || echo "!!! gemma ptq eval failed"
echo "=== $(date -u +%FT%TZ) Gemma PTQ done"
