#!/usr/bin/env bash
# M5 evidence 1: load ramp through the gateway -> KEDA scales the primary pool on queue depth.
# Writes serve/results/m5-autoscale/<UTC>/{manifest.json,outcomes.jsonl,summary.json,series.json,timeline.png}
# Usage: serve/scripts/m5_autoscale.sh [profile]
set -euo pipefail
. "$(dirname "$0")/../deploy/k3d/env.sh"
cd "$REPO_DIR"
PROFILE="${1:-90:1,150:4,180:6,150:2,150:0}"
OUT="serve/results/m5-autoscale/$(date -u +%Y%m%dT%H%M%SZ)"
CORTEX_API_KEY=$(serve/scripts/m5_key.sh m5-autoscale)
export CORTEX_API_KEY
log "ramp $PROFILE -> $OUT"
uv run gatewaybench load --out "$OUT" --profile "$PROFILE" --alias cortex-router \
  --input-tokens 64 --output-tokens 32 --note "autoscaling ramp, primary pool only (cortex-router)"
log "waiting 90 s for scale-down to show in the series"; sleep 90
uv run gatewaybench timeline "$OUT" --model primary --bin 10 \
  --title "KEDA queue-depth autoscaling, k3d CPU (Qwen3-0.6B Q4_K_M, 1 thread x 2 slots per replica)"
kubectl -n "$NS" get events --sort-by=.lastTimestamp -o custom-columns=TIME:.lastTimestamp,OBJ:.involvedObject.name,REASON:.reason,MSG:.message \
  | grep -E 'cortex-model-primary|keda-hpa' > "$OUT/k8s_events.txt" || true
log "done: $OUT"
