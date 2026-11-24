#!/usr/bin/env bash
# M5 evidence 4: 90/10 canary on cortex-default, then the promotion gate (error rate + TTFT from the
# traffic, exact-answer eval on both versions). Applies the verdict to the gateway:
#   PROMOTE -> gateway.canary.weight=100, ROLLBACK -> 0, HOLD -> unchanged (10).
# Usage: serve/scripts/m5_canary.sh [seconds] [concurrency] [min_canary_requests] [--dry-run]
set -euo pipefail
. "$(dirname "$0")/../deploy/k3d/env.sh"
cd "$REPO_DIR"
SECS="${1:-600}"; CONC="${2:-4}"; MIN="${3:-20}"; DRY="${4:-}"
OUT="serve/results/m5-canary/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT"
CORTEX_API_KEY=$(serve/scripts/m5_key.sh m5-canary); export CORTEX_API_KEY
log "canary traffic: cortex-default c=$CONC for ${SECS}s -> $OUT/traffic"
uv run gatewaybench load --out "$OUT/traffic" --profile "$SECS:$CONC" --alias cortex-default \
  --input-tokens 64 --output-tokens 32 --note "90/10 canary window"
uv run gatewaybench canary --traffic "$OUT/traffic" --out "$OUT" --min-requests "$MIN"
VERDICT=$(python3 -c "import json;print(json.load(open('$OUT/decision.json'))['decision']['verdict'])")
case "$VERDICT" in PROMOTE) W=100 ;; ROLLBACK) W=0 ;; *) W="" ;; esac
if [[ -n "$W" && "$DRY" != "--dry-run" ]]; then
  log "verdict $VERDICT -> gateway.canary.weight=$W"
  helm upgrade cortex "$DEPLOY_DIR/helm/cortex-serve" -n "$NS" --reuse-values --set gateway.canary.weight="$W" --wait --timeout 10m
  helm -n "$NS" history cortex -o json > "$OUT/helm_history.json"
else
  log "verdict $VERDICT (no change applied)"
fi
echo "{\"verdict\": \"$VERDICT\", \"applied_weight\": \"${W:-unchanged}\", \"dry_run\": \"${DRY:-no}\"}" > "$OUT/applied.json"
