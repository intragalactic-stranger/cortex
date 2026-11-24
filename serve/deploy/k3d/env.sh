# Shared environment for the k3d scripts. Source it: `. serve/deploy/k3d/env.sh`.
K3D_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DEPLOY_DIR="$(dirname "$K3D_DIR")"
REPO_DIR="$(cd "$DEPLOY_DIR/../.." && pwd)"
# shellcheck source=versions.env
. "$K3D_DIR/versions.env"
export CLUSTER=cortex
export KUBECONFIG="${CORTEX_KUBECONFIG:-$HOME/.config/k3d/kubeconfig-$CLUSTER.yaml}"
# GGUF cache from M1 (serve/scripts/m1_*.sh <models_dir>); default: sibling of the checkout.
export CORTEX_MODELS_DIR="${CORTEX_MODELS_DIR:-$(cd "$REPO_DIR/.." && pwd)/models}"
export NS=cortex
log() { echo "[$(date -u +%H:%M:%S)] $*"; }
