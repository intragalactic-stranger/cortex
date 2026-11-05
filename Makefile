.PHONY: setup lint fmt typecheck test test-network check demo

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
