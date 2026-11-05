# CLAUDE.md: conventions for the Cortex monorepo

Cortex is a self-hosted "personal AI brain" built in four packages. `serve/` (Project 2, Cortex Serve)
is the first: LLM inference memory math, benchmarking, quantization and production serving.

## Non-negotiables
- **uv only.** `uv sync`, `uv run`, `uv add`. Never bare `pip`, `venv` or `python -m pip`.
- **Python 3.12** (`.python-version`). `ruff` (lint + format), `mypy --strict`, `pytest`.
- `make check` must pass before every commit (CI runs the same).
- Pure functions for anything numeric; every formula has a hand-calculated test (`serve/tests/fixtures/golden.json`).
- Every external call has a timeout and retries. No secrets in code or fixtures (gitleaks runs in CI).
- Versions are verified at build time, never assumed; record them in `serve/docs/VERSIONS.md`.
- Every benchmark result needs a manifest (engine version, image digest, model revision SHA, GPU, flags, git SHA).
- Only claims the data supports go into RESULTS.md, docs or the decision guide.
- Non-obvious decisions get an ADR in `serve/docs/adr/NNNN-title.md`.

## Money
- Never start a paid GPU run (Modal or any cloud) beyond the approved per-milestone cap.
  Estimate first; every remote job has a hard timeout and an idle kill.

## Layout
- `serve/src/cortex_serve/memcalc/`: memory calculator (config -> per-layer specs -> arithmetic)
- `serve/tests/fixtures/models/<org>__<name>/`: vendored config.json + safetensors header table + meta.json (pinned SHA)
- `serve/docs/`: SPEC, learn/, adr/, milestones/, VERSIONS.md

## Useful commands
`make setup` · `make check` · `make demo` · `uv run memcalc --help` · `uv run pytest -m network`
