#!/usr/bin/env bash
# `make up`: start the compose stack. Generates serve/deploy/compose/.env (gitignored) once.
set -euo pipefail
cd "$(dirname "$0")"
if [[ ! -f .env ]]; then
  umask 077
  { echo "LITELLM_MASTER_KEY=sk-cortex-$(openssl rand -hex 24)"
    echo "BACKEND_API_KEY=$(openssl rand -hex 24)"
    echo "CORTEX_MODELS_DIR=${CORTEX_MODELS_DIR:-$(cd ../../../.. && pwd)/models}"; } > .env
  echo "wrote $(pwd)/.env (master key for http://127.0.0.1:4001)"
fi
docker compose up -d --wait
docker compose ps
