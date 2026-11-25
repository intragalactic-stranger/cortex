#!/usr/bin/env bash
# E15 quality: Qwen3.5-4B (hybrid) on the full suite + reasoning-first ARC; Qwen3-4B reasoning-first ARC for comparison.
set -u
cd "$(dirname "$0")/../.."
EXTRA='--limit-mm-per-prompt {"image":0,"video":0} --default-chat-template-kwargs {"enable_thinking":false}'
uv run --extra modal modal run serve/deploy/modal/eval_app.py --name qwen3.5-4b-bf16 --model Qwen/Qwen3.5-4B \
  --revision 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a --tokenizer Qwen/Qwen3.5-4B --parts loglik,gen,domain,arcgen \
  --extra "$EXTRA" --yes > serve/results/logs/m4-e15-q35-eval.log 2>&1 || echo "!!! qwen3.5 eval failed" &
sleep 15
uv run --extra modal modal run serve/deploy/modal/eval_app.py --name qwen3-4b-bf16 --model Qwen/Qwen3-4B-Instruct-2507 \
  --revision cdbee75f17c01a7cc42f958dc650907174af0554 --parts arcgen --max-minutes 30 --yes \
  > serve/results/logs/m4-q3-arcgen.log 2>&1 || echo "!!! qwen3 arcgen failed" &
wait
echo "=== $(date -u +%FT%TZ) E15 quality finished"
