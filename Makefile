.PHONY: setup lint fmt typecheck test test-network check demo k3d-up k3d-down k3d-status helm-check up down \
	web-setup web-dev web-test web-build web-vectors

setup:            ## install all workspace packages + dev tools, enable git hooks
	uv sync --all-packages --locked
	uv run pre-commit install

fmt:              ## auto-format and auto-fix
	uv run ruff format serve
	uv run ruff check serve --fix

lint:             ## lint + format check + strict type check
	uv run ruff check serve
	uv run ruff format --check serve
	uv run mypy

test:             ## offline unit tests
	uv run pytest

test-network:     ## opt-in tests that hit the Hugging Face Hub
	uv run pytest -m network

check: lint test  ## everything CI runs

demo:             ## E15 architecture comparison from vendored fixtures
	uv run memcalc compare --ctx 8192 --ctx 32768 -d L4 \
		-m serve/tests/fixtures/models/Qwen__Qwen3-4B-Instruct-2507 \
		-m serve/tests/fixtures/models/Qwen__Qwen3.5-4B \
		-m serve/tests/fixtures/models/google__gemma-4-E4B-it

# --- M5: local Kubernetes (k3d) and compose stacks -------------------------------------------------
k3d-up:           ## create the k3d cluster + KEDA + Prometheus/Grafana + cortex-serve chart
	serve/deploy/k3d/up.sh all

k3d-down:         ## delete the k3d cluster (frees its images and volumes)
	serve/deploy/k3d/down.sh

k3d-status:       ## pods, autoscaler and endpoints
	. serve/deploy/k3d/env.sh && kubectl -n cortex get pods,hpa,scaledobject,pdb,networkpolicy

helm-check:       ## lint + render the chart (no cluster needed)
	helm lint serve/deploy/helm/cortex-serve
	helm template cortex serve/deploy/helm/cortex-serve -n cortex > /dev/null

up:               ## single-command local stack without Kubernetes (docker compose)
	serve/deploy/compose/up.sh

down:             ## stop the compose stack
	docker compose -f serve/deploy/compose/compose.yaml down

# --- M6: results site (web/, static Next.js export) ------------------------------------------------
web-setup:        ## install the site's locked npm dependencies
	cd web && npm ci

web-dev:          ## local dev server on http://localhost:3000
	cd web && npm run dev

web-test:         ## type check + unit tests (memcalc parity, markdown, flowchart)
	cd web && npm run typecheck && npm test

web-build:        ## static export into web/out
	cd web && npm run build

web-vectors:      ## regenerate the memcalc parity vectors from the Python tool
	uv run python serve/scripts/memcalc_web_vectors.py
