#!/usr/bin/env bash
# Create the local k3d cluster and install KEDA, the trimmed Prometheus stack and the cortex-serve chart.
# Idempotent: re-running upgrades in place. Secrets are generated here, never committed.
# Usage: serve/deploy/k3d/up.sh [infra|app|all]   (default: all)
set -euo pipefail
. "$(dirname "$0")/env.sh"
PHASE="${1:-all}"

need() { command -v "$1" >/dev/null || { echo "missing $1 (install pinned binaries, see docs/VERSIONS.md)"; exit 1; }; }
need docker; need k3d; need kubectl; need helm; need openssl
k3d version | grep -q "k3d version $K3D_VERSION" || echo "WARN: k3d is not $K3D_VERSION"
helm version --short | grep -q "^$HELM_VERSION" || echo "WARN: helm is not $HELM_VERSION"
[[ -f "$CORTEX_MODELS_DIR/unsloth_Qwen3-0.6B-GGUF/Qwen3-0.6B-Q4_K_M.gguf" ]] || {
  echo "GGUF cache not found under CORTEX_MODELS_DIR=$CORTEX_MODELS_DIR"; exit 1; }

secret() {  # secret <ns> <name> <key>=<generator>... : create once, keep on re-runs
  local ns=$1 name=$2; shift 2
  kubectl -n "$ns" get secret "$name" >/dev/null 2>&1 && return 0
  local args=()
  for kv in "$@"; do args+=("--from-literal=${kv%%=*}=$(eval "${kv#*=}")"); done
  kubectl -n "$ns" create secret generic "$name" "${args[@]}" >/dev/null
  log "created secret $ns/$name"
}

infra() {
  if ! k3d cluster list "$CLUSTER" >/dev/null 2>&1; then
    log "creating k3d cluster $CLUSTER (models: $CORTEX_MODELS_DIR)"
    k3d cluster create --config "$K3D_DIR/cluster.yaml"
  fi
  mkdir -p "$(dirname "$KUBECONFIG")"
  k3d kubeconfig get "$CLUSTER" > "$KUBECONFIG"; chmod 600 "$KUBECONFIG"
  kubectl wait --for=condition=Ready node --all --timeout=180s

  helm repo add kedacore https://kedacore.github.io/charts --force-update >/dev/null
  helm repo add prometheus-community https://prometheus-community.github.io/helm-charts --force-update >/dev/null
  helm repo update kedacore prometheus-community >/dev/null

  kubectl create namespace monitoring --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  secret monitoring grafana-admin "admin-user=echo admin" "admin-password=openssl rand -hex 16"
  log "installing kube-prometheus-stack $KPS_CHART_VERSION"
  helm upgrade --install kps prometheus-community/kube-prometheus-stack --version "$KPS_CHART_VERSION" \
    -n monitoring -f "$DEPLOY_DIR/monitoring/kube-prometheus-stack.values.yaml" --wait --timeout 10m
  log "installing KEDA $KEDA_CHART_VERSION"
  helm upgrade --install keda kedacore/keda --version "$KEDA_CHART_VERSION" \
    -n keda --create-namespace -f "$DEPLOY_DIR/monitoring/keda.values.yaml" --wait --timeout 10m
}

app() {
  kubectl create namespace "$NS" --dry-run=client -o yaml | kubectl apply -f - >/dev/null
  # Gateway master key, key-hashing salt, the backend key that only the gateway (and Prometheus) hold,
  # and the gateway's Postgres password. All random, generated once per cluster.
  secret "$NS" litellm-secrets \
    "master-key=echo sk-cortex-\$(openssl rand -hex 24)" \
    "salt-key=echo sk-salt-\$(openssl rand -hex 24)"
  secret "$NS" backend-api-key "api-key=openssl rand -hex 24"
  secret "$NS" gateway-db "password=openssl rand -hex 24"
  log "installing cortex-serve chart"
  helm upgrade --install cortex "$DEPLOY_DIR/helm/cortex-serve" -n "$NS" \
    ${CORTEX_VALUES:+-f "$CORTEX_VALUES"} --wait --timeout 15m
  kubectl -n "$NS" get pods -o wide
  log "gateway: http://127.0.0.1:4000  prometheus: http://127.0.0.1:9090  grafana: http://127.0.0.1:3300"
  log "master key: kubectl -n $NS get secret litellm-secrets -o jsonpath='{.data.master-key}' | base64 -d"
}

case "$PHASE" in
  infra) infra ;;
  app) app ;;
  all) infra; app ;;
  *) echo "usage: $0 [infra|app|all]"; exit 2 ;;
esac
