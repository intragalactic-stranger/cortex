#!/usr/bin/env bash
# M3 QAT vs PTQ (D-014): Gemma 4 E4B original bf16 vs Google's QAT W4A16 (g32 sym) vs our GPTQ PTQ at the same scheme.
# L40S (48 GB): E4B bf16 is ~16 GB incl. vision/audio towers. Multimodal inputs disabled; thinking off.
set -u
cd "$(dirname "$0")/../.."
TOK=google/gemma-4-E4B-it
EXTRA='--limit-mm-per-prompt {"image":0,"audio":0} --default-chat-template-kwargs {"enable_thinking":false}'
ev() { echo "=== $(date -u +%FT%TZ) eval $1"; uv run --extra modal modal run serve/deploy/modal/eval_app.py --name "$1" "${@:2}" --tokenizer "$TOK" --gpu L40S --extra "$EXTRA" --yes || echo "!!! eval $1 failed (exit $?)"; }
ev gemma4-e4b-bf16 --model google/gemma-4-E4B-it --revision 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c &
sleep 20
ev gemma4-e4b-qat-w4a16-g32 --model google/gemma-4-E4B-it-qat-w4a16-ct --revision 6cd26aaa2357fb2bad8c51699a7558a4d1a965bb &
a=gemma4-e4b-w4a16-gptq-g32-sym-generic
until [ -f "serve/results/quantize/$a/cortex_manifest.json" ] || ! pgrep -f m3_quantize_all.sh >/dev/null; do sleep 60; done
if [ -f "serve/results/quantize/$a/cortex_manifest.json" ]; then ev gemma4-e4b-ptq-w4a16-g32 --model "/hf/artifacts/$a" & else echo "!!! $a never produced"; fi
wait
echo "=== $(date -u +%FT%TZ) Gemma QAT vs PTQ evals finished"
