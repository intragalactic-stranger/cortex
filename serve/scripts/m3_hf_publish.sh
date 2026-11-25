#!/usr/bin/env bash
# Publish the owner-approved M3 artifacts as PRIVATE Hugging Face repos (2026-11-25; owner flips them public).
set -u
cd "$(dirname "$0")/../.."
pub() { echo "=== $(date -u +%FT%TZ) $2"; uv run --extra modal modal run serve/deploy/modal/hf_publish.py --artifact "$1" --repo "$2" --private --yes || echo "!!! $2 failed"; }
pub qwen3-4b-w4a16-gptq-g128-generic Qwen3-4B-Instruct-2507-W4A16-GPTQ
pub qwen3-4b-w4a16-awq-asym-generic Qwen3-4B-Instruct-2507-W4A16-AWQ
pub qwen3-4b-w8a8-int8-generic Qwen3-4B-Instruct-2507-W8A8-INT8
pub qwen3-4b-w4a16-gptq-g128-bfsi Qwen3-4B-Instruct-2507-W4A16-GPTQ-BFSI
echo "=== $(date -u +%FT%TZ) publishing finished"
