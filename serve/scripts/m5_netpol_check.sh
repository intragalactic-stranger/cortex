#!/usr/bin/env bash
# M5: verify NetworkPolicies and backend auth from inside the cluster. Writes a PASS/FAIL table.
# Usage: serve/scripts/m5_netpol_check.sh
set -uo pipefail
. "$(dirname "$0")/../deploy/k3d/env.sh"
cd "$REPO_DIR"
OUT="serve/results/m5-netpol/$(date -u +%Y%m%dT%H%M%SZ)"; mkdir -p "$OUT"
IMG=$(kubectl -n "$NS" get deploy cortex-model-primary -o jsonpath='{.spec.template.spec.containers[0].image}')
SVC="http://cortex-model-primary.$NS.svc.cluster.local:8080"
fails=0
row() {  # row <expect: ok|blocked|401> <description> <actual http code or 000>
  local want=$1 desc=$2 got=$3 verdict=FAIL
  case "$want" in
    blocked) [[ "$got" == 000 ]] && verdict=PASS ;;
    *) [[ "$got" == "$want" ]] && verdict=PASS ;;
  esac
  [[ $verdict == FAIL ]] && fails=$((fails+1))
  printf '| %s | expect %s | got %s | %s |\n' "$desc" "$want" "$got" "$verdict" | tee -a "$OUT/result.md"
}
probe_pod() {  # probe_pod <ns> <url>: throwaway unlabelled pod, prints HTTP code (000 = no connection)
  kubectl -n "$1" run "np-probe-$RANDOM" --rm -i --restart=Never --quiet --image="$IMG" \
    --overrides='{"spec":{"securityContext":{"runAsNonRoot":true,"runAsUser":10001},"automountServiceAccountToken":false}}' \
    --command -- curl -s -o /dev/null -w '%{http_code}' -m 5 "$2" 2>/dev/null | tail -c 3
}
gw_py() {  # gw_py <url> [ENV]: HTTP code of a GET from inside the gateway pod (python stdlib); ENV = send the backend key
  kubectl -n "$NS" exec deploy/cortex-gateway -- python -c "
import os, sys, urllib.request, urllib.error
key = os.environ['BACKEND_API_KEY'] if sys.argv[2] == 'ENV' else ''
req = urllib.request.Request(sys.argv[1], headers={'Authorization': 'Bearer ' + key} if key else {})
try: print(urllib.request.urlopen(req, timeout=5).status)
except urllib.error.HTTPError as e: print(e.code)
except Exception: print('000')" "$1" "${2:-}" 2>/dev/null | tail -1
}
echo "| Path | Expected | Actual | Result |" > "$OUT/result.md"; echo "|---|---|---|---|" >> "$OUT/result.md"
row blocked "unlabelled pod in ns $NS -> primary :8080/health" "$(probe_pod "$NS" "$SVC/health")"
row blocked "pod in ns default -> primary :8080/health" "$(probe_pod default "$SVC/health")"
row blocked "unlabelled pod in ns $NS -> gateway DB :5432" "$(probe_pod "$NS" "http://cortex-gateway-db.$NS.svc.cluster.local:5432/")"
row 200 "gateway -> primary /health" "$(gw_py "$SVC/health")"
row 401 "gateway -> primary /v1/models without backend key" "$(gw_py "$SVC/v1/models")"
row 401 "gateway -> primary /metrics without backend key" "$(gw_py "$SVC/metrics")"
row 200 "gateway -> primary /v1/models with backend key" "$(gw_py "$SVC/v1/models" ENV)"
row blocked "gateway -> internet (https://huggingface.co)" "$(gw_py https://huggingface.co)"
row blocked "primary pod -> internet (egress denied)" "$(kubectl -n "$NS" exec deploy/cortex-model-primary -- curl -s -o /dev/null -w '%{http_code}' -m 5 https://huggingface.co 2>/dev/null | tail -c 3)"
row blocked "primary pod -> gateway :4000 (egress denied)" "$(kubectl -n "$NS" exec deploy/cortex-model-primary -- curl -s -o /dev/null -w '%{http_code}' -m 5 http://cortex-gateway.$NS.svc.cluster.local:4000/health/liveliness 2>/dev/null | tail -c 3)"
echo "failures: $fails" | tee -a "$OUT/result.md"
