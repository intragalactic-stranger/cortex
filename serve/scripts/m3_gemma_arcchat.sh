#!/usr/bin/env bash
# Gemma 4 E4B: generative ARC for all variants (loglikelihood MC is at chance even for bf16), plus the RTN baseline's
# generative tasks + domain perplexity. L40S, in parallel.
set -u
cd "$(dirname "$0")/../.."
TOK=google/gemma-4-E4B-it
EXTRA='--limit-mm-per-prompt {"image":0,"audio":0} --default-chat-template-kwargs {"enable_thinking":false}'
ev() { uv run --extra modal modal run serve/deploy/modal/eval_app.py --name "$1" "${@:2}" --tokenizer "$TOK" --gpu L40S --extra "$EXTRA" --add-bos --yes > "serve/results/logs/m3-gemma-$1-$(echo "${@: -1}" | tr , +).log" 2>&1 || echo "!!! $1 failed"; }
ev gemma4-e4b-bf16 --model google/gemma-4-E4B-it --revision 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c --parts arcchat & sleep 15
ev gemma4-e4b-qat-w4a16-g32 --model google/gemma-4-E4B-it-qat-w4a16-ct --revision 6cd26aaa2357fb2bad8c51699a7558a4d1a965bb --parts arcchat & sleep 15
ev gemma4-e4b-ptq-w4a16-g32 --model /hf/artifacts/gemma4-e4b-w4a16-gptq-g32-sym-generic --parts arcchat & sleep 15
ev gemma4-e4b-rtn-w4a16-g32 --model /hf/artifacts/gemma4-e4b-w4a16-rtn-g32-sym --parts gen,arcchat,domain &
wait
echo "=== $(date -u +%FT%TZ) Gemma arcchat finished"
