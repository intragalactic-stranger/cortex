#!/usr/bin/env bash
# M3 E3: full quality evals. Reference models now (parallel L4s), then each quantized artifact as soon as the
# quantize queue has written its manifest. Every launch passes the shared spend guard (in-flight worst cases count).
set -u
cd "$(dirname "$0")/../.."
ev() { echo "=== $(date -u +%FT%TZ) eval $1"; uv run --extra modal modal run serve/deploy/modal/eval_app.py --name "$1" "${@:2}" --yes || echo "!!! eval $1 failed (exit $?)"; }
ev qwen3-4b-bf16 --model Qwen/Qwen3-4B-Instruct-2507 --revision cdbee75f17c01a7cc42f958dc650907174af0554 &
sleep 20
ev qwen3-4b-fp8-dynamic --model /hf/artifacts/qwen3-4b-fp8-dynamic &
sleep 20
ev qwen3-4b-fp8-official --model Qwen/Qwen3-4B-Instruct-2507-FP8 --revision 8591804019c8b22094c3b5b4454e0edc05dffc98 &
for a in qwen3-4b-w4a16-gptq-g128-generic qwen3-4b-w4a16-awq-asym-generic qwen3-4b-w8a8-int8-generic qwen3-4b-w4a16-gptq-g128-bfsi; do
  until [ -f "serve/results/quantize/$a/cortex_manifest.json" ] || ! pgrep -f m3_quantize_all.sh >/dev/null; do sleep 60; done
  if [ -f "serve/results/quantize/$a/cortex_manifest.json" ]; then ev "$a" --model "/hf/artifacts/$a" & sleep 20; else echo "!!! $a never produced"; fi
done
wait
echo "=== $(date -u +%FT%TZ) M3 evals finished"
