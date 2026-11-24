#!/usr/bin/env bash
# M5 evidence 3 (D-021): primary pool down -> the gateway serves cortex-router from the second
# self-hosted model (Qwen3-1.7B, cortex-fallback). The pool is taken to 0 replicas by pausing KEDA
# at 0 (autoscaling.keda.sh/paused-replicas), then restored.
# Usage: serve/scripts/m5_fallback.sh
set -euo pipefail
. "$(dirname "$0")/../deploy/k3d/env.sh"
cd "$REPO_DIR"
OUT="serve/results/m5-fallback/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT"
CORTEX_API_KEY=$(serve/scripts/m5_key.sh m5-fallback); export CORTEX_API_KEY
mark() { echo "$(date -u +%H:%M:%S.%3N) $(date +%s.%N) $*" | tee -a "$OUT/events.log"; }
log "load c=2 for 360 s -> $OUT"
uv run gatewaybench load --out "$OUT" --profile "360:2" --alias cortex-router \
  --input-tokens 64 --output-tokens 32 --note "fallback demo: primary pool scaled to 0 mid-run" &
LOAD=$!
sleep 60
mark "PRIMARY DOWN: pause KEDA at 0 replicas"
kubectl -n "$NS" annotate scaledobject cortex-model-primary autoscaling.keda.sh/paused-replicas=0 --overwrite
kubectl -n "$NS" wait --for=delete pod -l cortex.dev/model=primary --timeout=300s || true
mark "primary pods gone"
sleep 150
mark "PRIMARY RESTORE: unpause KEDA"
kubectl -n "$NS" annotate scaledobject cortex-model-primary autoscaling.keda.sh/paused-replicas-
kubectl -n "$NS" rollout status deployment/cortex-model-primary --timeout=300s
mark "primary ready"
wait $LOAD
log "done: $OUT"; python3 -c "import json;s=json.load(open('$OUT/summary.json'));print({k:s[k] for k in ('requests','ok','failed','by_deployment','with_fallbacks','errors')})"
