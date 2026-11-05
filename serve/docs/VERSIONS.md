# Versions

Resolved and verified at build time. Exact pins for Python packages live in `uv.lock`; engine image digests are added from M1/M2.

| Component | Version | Verified | Notes |
|---|---|---|---|
| uv | 0.13.7 | 2026-11-05 | mandatory package/env manager (D-010) |
| Python | 3.12.8 | 2026-11-05 | installed by uv; `.python-version` = 3.12 |
| pydantic | 2.14.2 | 2026-11-05 | |
| typer | 0.27.3 | 2026-11-05 | memcalc CLI |
| rich | 15.0.0 | 2026-11-05 | |
| httpx | 0.28.1 | 2026-11-05 | Hub access, range requests (later: servebench client) |
| pyyaml | 6.0.3 | 2026-11-05 | |
| ruff | 0.17.3 | 2026-11-05 | |
| mypy | 2.4.1 | 2026-11-05 | `--strict` |
| pytest | 9.2.0 | 2026-11-05 | |
| pre-commit | 4.7.0 | 2026-11-05 | gitleaks hook v8.24.2 |

## Model revisions (fixtures)

| Model | Revision (HF commit) | License |
|---|---|---|
| Qwen/Qwen3-0.6B | c1899de289a04d12100db370d81485cdf75e47ca | Apache-2.0 |
| Qwen/Qwen3-1.7B | 70d244cc86ccca08cf5af4e1e306ecf908b1ad5e | Apache-2.0 |
| Qwen/Qwen3-4B-Instruct-2507 | cdbee75f17c01a7cc42f958dc650907174af0554 | Apache-2.0 |
| Qwen/Qwen3-30B-A3B | ad44e777bcd18fa416d9da3bd8f70d33ebb85d39 | Apache-2.0 |
| Qwen/Qwen3.5-4B | 851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a | Apache-2.0 |
| google/gemma-4-E2B-it | 4adf15748de9e1a0097b6274cee9c15836c41c37 | Apache-2.0 |
| google/gemma-4-E4B-it | 4ecc2ba63bcac23ba4a767fc87025b1450f84c9c | Apache-2.0 |
