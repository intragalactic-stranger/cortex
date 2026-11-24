#!/usr/bin/env bash
# Delete the k3d cluster (and with it every image pulled into the node). The GGUF cache is untouched.
set -euo pipefail
. "$(dirname "$0")/env.sh"
if k3d cluster list "$CLUSTER" >/dev/null 2>&1; then
  k3d cluster delete "$CLUSTER"
else
  log "no cluster named $CLUSTER"
fi
rm -f "$KUBECONFIG"
