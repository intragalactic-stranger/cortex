#!/usr/bin/env bash
# Re-run the Gemma PTQ (GPTQ g32) and RTN evals after processor_config.json was added to the artifacts.
set -u
cd "$(dirname "$0")/../.."
TOK=google/gemma-4-E4B-it
EXTRA='--limit-mm-per-prompt {"image":0,"audio":0} --default-chat-template-kwargs {"enable_thinking":false}'
ev() { uv run --extra modal modal run serve/deploy/modal/eval_app.py --name "$1" --model "$2" --parts "$3" --tokenizer "$TOK" --gpu L40S --extra "$EXTRA" --add-bos --yes > "serve/results/logs/m3-gemma-$1-rerun.log" 2>&1 || echo "!!! $1 failed"; }
ev gemma4-e4b-ptq-w4a16-g32 /hf/artifacts/gemma4-e4b-w4a16-gptq-g32-sym-generic loglik,gen,domain,arcchat & sleep 15
ev gemma4-e4b-rtn-w4a16-g32 /hf/artifacts/gemma4-e4b-w4a16-rtn-g32-sym gen,arcchat,domain &
wait
echo "=== $(date -u +%FT%TZ) Gemma PTQ/RTN evals finished"
