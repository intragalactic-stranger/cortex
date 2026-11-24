#!/usr/bin/env bash
# M5 evidence 2: rolling update of the primary pool under steady load; every request outcome logged.
# The rollout replaces every primary pod (kubectl rollout restart = new pod template, same path as an
# image/args change) with maxUnavailable 0 / maxSurge 1, readiness-gated, drain-aware preStop.
# Usage: serve/scripts/m5_rollout.sh [concurrency] [seconds]
set -euo pipefail
. "$(dirname "$0")/../deploy/k3d/env.sh"
cd "$REPO_DIR"
CONC="${1:-4}"; SECS="${2:-420}"
OUT="serve/results/m5-rollout/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT"
CORTEX_API_KEY=$(serve/scripts/m5_key.sh m5-rollout); export CORTEX_API_KEY
kubectl -n "$NS" get pods -l cortex.dev/model=primary -w \
  -o custom-columns=NAME:.metadata.name,READY:.status.containerStatuses[0].ready,PHASE:.status.phase,DEL:.metadata.deletionTimestamp \
  | while IFS= read -r l; do echo "$(date -u +%H:%M:%S.%3N) $l"; done > "$OUT/pods_watch.log" &
WATCH=$!
trap 'kill $WATCH 2>/dev/null || true' EXIT
log "steady load c=$CONC for ${SECS}s -> $OUT"
uv run gatewaybench load --out "$OUT" --profile "$SECS:$CONC" --alias cortex-router \
  --input-tokens 64 --output-tokens 32 --note "rolling update of primary pool under load" &
LOAD=$!
sleep 90
log "rollout restart (1/2)"; echo "$(date -u +%H:%M:%S.%3N) ROLLOUT 1 START" >> "$OUT/pods_watch.log"
kubectl -n "$NS" rollout restart deployment/cortex-model-primary
kubectl -n "$NS" rollout status deployment/cortex-model-primary --timeout=600s
echo "$(date -u +%H:%M:%S.%3N) ROLLOUT 1 DONE" >> "$OUT/pods_watch.log"
sleep 60
log "rollout restart (2/2)"; echo "$(date -u +%H:%M:%S.%3N) ROLLOUT 2 START" >> "$OUT/pods_watch.log"
kubectl -n "$NS" rollout restart deployment/cortex-model-primary
kubectl -n "$NS" rollout status deployment/cortex-model-primary --timeout=600s
echo "$(date -u +%H:%M:%S.%3N) ROLLOUT 2 DONE" >> "$OUT/pods_watch.log"
wait $LOAD
kubectl -n "$NS" rollout history deployment/cortex-model-primary > "$OUT/rollout_history.txt"
uv run gatewaybench timeline "$OUT" --model primary --bin 10 --title "Rolling update under load (2 rollouts), k3d CPU"
uv run gatewaybench pods "$OUT" --model primary
log "done: $OUT"; python3 -c "import json;s=json.load(open('$OUT/summary.json'));print({k:s[k] for k in ('requests','ok','failed','with_retries','errors')})"
