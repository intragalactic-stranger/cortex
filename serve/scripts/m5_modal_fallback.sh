#!/usr/bin/env bash
# M5 GPU fallback demo: cortex-default normally runs on the local CPU pool; with the pool at 0 replicas the
# gateway falls back to the Modal vLLM endpoint (our M3 FP8 Qwen3-4B on an L4), then to the second local model.
# Needs: a running k3d cluster (make k3d-up), the deployed endpoint (modal deploy serve/deploy/modal/vllm_endpoint.py),
#        CORTEX_MODAL_URL=https://<workspace>--cortex-gpu-endpoint-serve.<region>.modal.run (not committed),
#        CORTEX_MODAL_KEY_FILE=<0600 file with the endpoint key> (not committed).
# Usage: serve/scripts/m5_modal_fallback.sh
set -euo pipefail
. "$(dirname "$0")/../deploy/k3d/env.sh"
cd "$REPO_DIR"
: "${CORTEX_MODAL_URL:?set CORTEX_MODAL_URL}" "${CORTEX_MODAL_KEY_FILE:?set CORTEX_MODAL_KEY_FILE}"
kubectl -n "$NS" get secret modal-endpoint >/dev/null 2>&1 || \
  kubectl -n "$NS" create secret generic modal-endpoint --from-file=api-key="$CORTEX_MODAL_KEY_FILE" >/dev/null
log "enabling the Modal slot in the gateway"
helm upgrade cortex "$DEPLOY_DIR/helm/cortex-serve" -n "$NS" --reuse-values \
  --set gateway.modal.enabled=true --set gateway.modal.apiBase="$CORTEX_MODAL_URL/v1" \
  --set gateway.modal.model=cortex-gpu --wait --timeout 10m
OUT="serve/results/m5-modal-fallback/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT"
CORTEX_API_KEY=$(serve/scripts/m5_key.sh m5-modal-fallback); export CORTEX_API_KEY
mark() { echo "$(date +%s.%N) $*" | tee -a "$OUT/events.log"; }
log "warming the GPU endpoint (cold start ~2-3 min) and loading c=2 for 420 s -> $OUT"
curl -s -o /dev/null --max-time 900 -H "Authorization: Bearer $(cat "$CORTEX_MODAL_KEY_FILE")" "$CORTEX_MODAL_URL/v1/models" || true
uv run gatewaybench load --out "$OUT" --profile "420:2" --alias cortex-default \
  --input-tokens 64 --output-tokens 32 --note "GPU fallback demo: local pool to 0, Modal L4 vLLM serves" &
LOAD=$!
sleep 60
mark "PRIMARY DOWN: pause KEDA at 0 replicas"
kubectl -n "$NS" annotate scaledobject cortex-model-primary autoscaling.keda.sh/paused-replicas=0 --overwrite
kubectl -n "$NS" wait --for=delete pod -l cortex.dev/model=primary --timeout=300s || true
mark "primary pods gone (gateway should route to the Modal endpoint)"
sleep 200
mark "PRIMARY RESTORE: unpause KEDA"
kubectl -n "$NS" annotate scaledobject cortex-model-primary autoscaling.keda.sh/paused-replicas-
kubectl -n "$NS" rollout status deployment/cortex-model-primary --timeout=300s
mark "primary ready"
wait $LOAD
log "done: $OUT"
python3 -c "import json;s=json.load(open('$OUT/summary.json'));print({k:s.get(k) for k in ('requests','ok','failed','by_deployment','with_fallbacks','errors')})"
