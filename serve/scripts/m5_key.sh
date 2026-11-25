#!/usr/bin/env bash
# Create a LiteLLM virtual key for local load tests and print it (nothing is written to disk).
# Usage: export CORTEX_API_KEY=$(serve/scripts/m5_key.sh [alias] [rpm] [max_parallel])
set -euo pipefail
. "$(dirname "$0")/../deploy/k3d/env.sh"
ALIAS="${1:-m5-loadtest}"; RPM="${2:-1200}"; PAR="${3:-32}"
MASTER=$(kubectl -n "$NS" get secret litellm-secrets -o jsonpath='{.data.master-key}' | base64 -d)
curl -sf --max-time 30 --retry 3 http://127.0.0.1:4000/key/generate \
  -H "Authorization: Bearer $MASTER" -H 'Content-Type: application/json' \
  -d "{\"key_alias\": \"$ALIAS-$(date +%s)\", \"models\": [\"cortex-default\", \"cortex-router\", \"cortex-judge\", \"cortex-fallback\", \"cortex-stable\", \"cortex-candidate\", \"cortex-gpu\"],
       \"rpm_limit\": $RPM, \"max_parallel_requests\": $PAR, \"duration\": \"24h\", \"metadata\": {\"purpose\": \"m5 local load test\"}}" \
  | python3 -c 'import sys,json; print(json.load(sys.stdin)["key"])'
