# M0 · Foundations: monorepo, CI, `memcalc`, learning docs: Spec

> **Status:** `AWAITING APPROVAL` · **Branch:** `serve/m0-foundations` · **Date:** 2026-11-04 · **Modal $:** 0
> Drafted here; copied to `cortex/serve/docs/milestones/M0_SPEC.md` at scaffold time.
> **Decisions in force:** D-004 (monorepo), D-010 (uv mandatory, Python 3.12), D-011 (memcalc design), D-012 (private repo until M2), D-014 (model lineup + E15).

## 1. Goal
Lay the engineering foundation every later milestone builds on, and ship the first real tool: **`memcalc`**. It predicts weight memory, KV cache, max concurrency and decode ceilings for any HF model on any device. It must be correct for the **three memory architectures in our lineup** (dense GQA, hybrid linear attention, sliding window + KV sharing) plus MoE. Its predictions become the hypotheses that M1, M2 and E15 test.

## 2. Scope
**In**
1. `cortex` monorepo scaffold (local `git init` at `P2/01_Implementation/cortex/`), uv workspace, `serve` package, quality tooling, CI, `CLAUDE.md`.
2. `memcalc` package + CLI + device table + golden-vector tests.
3. 8 learning docs (spec §3) in `serve/docs/learn/`.
4. ADRs 0001–0004, `VERSIONS.md`, `SPEC.md` (spec copy + adaptation notes).
5. Create the **private** GitHub repo and push at the end, after you confirm (D-012).

**Out:** benchmarking (M1), measured validation (M1/M2), TS port (M6), measuring this CPU's memory bandwidth (M1: the device-table value starts as "unmeasured").

## 3. Research summary (2026-11-04)
- Tool versions: uv **0.13.7** (upgraded, D-010), Python **3.12.8** (via uv), typer **0.27.3**. ruff, mypy, pytest, pydantic and huggingface_hub get pinned at scaffold, recorded in `VERSIONS.md`.
- Architecture facts for all lineup models were read from HF configs (`evidence/M0/02_*`). Gemma 4's KV-sharing semantics were verified in transformers' `configuration_gemma4.py::kv_sharing_modes`: the last `num_kv_shared_layers` layers reuse the KV of the last non-shared layer of the same type.
- Qwen3.5's linear-attention state dtype is `mamba_ssm_dtype: float32` (config).
- Hand-calculated expectations: `evidence/M0/04_kv-cache-corrected.txt`.

## 4. Design

### 4.1 Repo layout created in M0
```
cortex/
  CLAUDE.md  README.md  LICENSE  Makefile  pyproject.toml  uv.lock  .python-version (3.12)
  .pre-commit-config.yaml  .gitignore  .github/workflows/ci.yml
  serve/
    pyproject.toml                     # distribution `cortex-serve`; scripts: memcalc
    README.md  RESULTS.md (stub)
    src/cortex_serve/
      __init__.py
      memcalc/
        __init__.py
        config.py      # load HF config (repo@revision or file) → normalised ModelSpec
        families.py    # adapters: dense (llama/qwen3/phi3/smollm3), qwen3_moe, qwen3_5 (hybrid), gemma4
        model.py       # pydantic: ModelSpec, LayerSpec, ParamBreakdown
        dtypes.py      # bytes/param: bf16, fp16, fp8, int8, int4(+group-scale overhead), GGUF bpw table
        devices.py     # load devices.yaml → Device
        devices.yaml   # GPUs + CPU/unified-memory devices, each with source URL + approximate flag
        calc.py        # pure functions: weights, kv/token, per-seq, free mem, max seqs, decode ceiling, prefill FLOPs
        cli.py         # typer app
    tests/
      fixtures/configs/*.json          # vendored configs pinned by HF commit SHA (tests run offline)
      golden.json                      # inputs → expected outputs (shared with the TS port in M6)
      test_calc.py  test_families.py  test_cli.py  test_golden.py
    docs/ SPEC.md  VERSIONS.md  learn/01..08_*.md  adr/0001..0004  milestones/M0_SPEC.md
```

### 4.2 The memory model (the core idea)
memcalc normalises any config into a **list of per-layer specs**, so architecture differences become data, not special cases:

| Layer kind | Own KV? | KV bytes for a sequence of length `ctx` | Example |
|---|---|---|---|
| `full` | yes | `2 · kv_heads · head_dim · kv_bytes · ctx` | Qwen3 (all 36 layers); Gemma 4 global layers (head_dim = `global_head_dim`) |
| `sliding` | yes | `2 · kv_heads · head_dim · kv_bytes · min(ctx, window)` | Gemma 4 (window 512) |
| `shared` | no | 0 (reads the producer layer's KV) | Gemma 4's last 18 / 20 layers |
| `linear` | no KV; fixed state | `v_heads · k_dim · v_dim · state_bytes` + conv state `(2·k_heads·k_dim + v_heads·v_dim) · (kernel−1) · act_bytes` | Qwen3.5 Gated DeltaNet layers |

```
per_seq_bytes(ctx)  = Σ_layers layer_bytes(ctx)
weights_bytes       = Σ_tensors params × bytes(dtype)          # stored params
free_kv_bytes       = device_mem × util − weights_bytes − reserve
max_concurrent_seqs = floor(free_kv_bytes / per_seq_bytes(ctx))
decode_ceiling_tok_s ≈ bandwidth / bytes_read_per_token        # active weights (MoE: shared + top-k experts) + KV read at ctx
prefill_flops_per_1k ≈ 2 · compute_params · 1000 (+ attention term reported separately)
```

**Params are reported three ways:** *stored* (everything in the checkpoint, including vision/audio towers and Gemma's Per-Layer Embeddings), *compute* (what each token passes through: excludes PLE lookups and unused towers), and *active* (MoE: shared + top-k experts). Source of truth: the HF safetensors metadata count when available, cross-checked against an analytic count from the config (the analytic formula is the learning artifact).

**Reserve:** fixed default of 1.5 GiB on GPU (CUDA context, CUDA graphs, activations), 0.5 GiB on CPU. Overridable with `--reserve-gib`, and refit from measurements in M2 (D-011).

### 4.3 CLI
```bash
memcalc estimate --model Qwen/Qwen3-4B-Instruct-2507 --device L4 --dtype bf16 --kv-dtype fp8 --ctx 8192 --util 0.9
memcalc estimate --config path/to/config.json --device local-epyc-7571 --dtype q4_k_m --json
memcalc compare  --model Qwen/Qwen3-4B-Instruct-2507 --model Qwen/Qwen3.5-4B --model google/gemma-4-E4B-it --device L4 --ctx 8192 --ctx 32768
memcalc devices
```
Output: a readable table (rich) by default, `--json` for the report generator and the web site. Every output states its assumptions (reserve, util, which layers own KV).

### 4.4 Devices (`devices.yaml`)
T4, L4, A10G, A100-40/80, H100-80, H200, RTX 4090, Apple M-series, plus **`local-epyc-7571`** (15 GiB, bandwidth `null` until measured in M1). Each entry has `memory_gib`, `bandwidth_gbs`, `bf16_tflops`, `fp8_tflops` (null where unsupported), `source_url`, `approximate: true`. Values are taken from vendor spec pages at build time.

## 5. Decision gates
All gates for M0 are decided. **Small choices made directly (veto any):**
| Choice | Default |
|---|---|
| Python import package / distribution | `cortex_serve` / `cortex-serve`, src layout |
| License | **Apache-2.0** (matches Qwen/Gemma; includes a patent grant) |
| Pretty output | `rich` (ships with typer) |
| Config/data models | pydantic v2 |
| HF access | `huggingface_hub` (with timeouts + retries); network tests marked `@pytest.mark.network`, skipped in CI |
| Pre-commit hooks | ruff (lint + format), mypy, gitleaks, end-of-file/trailing-whitespace |
| CI | GitHub Actions: `uv sync --locked` → ruff → mypy --strict → pytest → gitleaks |

## 6. Acceptance criteria
- [ ] **AC1** From a clean clone: `make setup && make lint && make test` pass; CI green on the PR.
- [ ] **AC2** `memcalc estimate` prints all spec §4 outputs for Qwen3-4B on L4; `--json` validates against a pydantic schema.
- [ ] **AC3** Golden vectors match **exactly** (integer bytes): Llama-3.1-8B 131,072 B/token and 1 GiB per 8K sequence (spec §3.2); Qwen3-4B 1,152 MiB, Qwen3.5-4B 305 MiB (±1 MiB for conv-state rounding), Gemma 4 E4B 148 MiB, Gemma 4 E2B 54 MiB per 8K sequence at bf16 KV; Qwen3-30B-A3B 96 KiB/token.
- [ ] **AC4** Decode ceilings match spec §3.3 within ±5%: 8B bf16 ≈ 200 tok/s on H100, ≈ 19 tok/s on L4.
- [ ] **AC5** Analytic param count is within ±2% of safetensors metadata for every dense/MoE fixture. For Gemma 4 and Qwen3.5, the stored/compute/active breakdown is documented, with a per-component table.
- [ ] **AC6** `memcalc compare` prints the E15 table for the three ~4B models at 8K and 32K.
- [ ] **AC7** 8 learning docs exist, each ≤2 pages, with a worked example generated by memcalc and an "interview questions this answers" section.
- [ ] **AC8** ADRs 0001–0004 written; `VERSIONS.md` lists every pinned tool; gitleaks clean.
- [ ] **AC9** Private GitHub repo `intragalactic-stranger/cortex` created and pushed (**after your confirmation**).

## 7. Test plan
- **Unit:** each `calc.py` function with hand-computed cases; each family adapter against its vendored fixture (layer counts by kind, shared-layer mapping, state sizes).
- **Golden:** `golden.json`, a list of `{inputs, expected}` cases, also consumed by the TS port in M6.
- **CLI:** typer `CliRunner` snapshot tests for table and JSON output.
- **Property tests (light):** monotonicity, e.g. more context means more memory, fp8 KV means less memory, a larger window never reduces KV.
- **Network (opt-in):** fetch live configs for the lineup and diff against the fixtures, which flags upstream config changes.

## 8. Cost & time
$0. Estimated ~6–9 hours of Claude Code work, split into reviewable commits: scaffold → memcalc core → families → CLI → docs → ADRs.

## 9. Risks
| Risk | Mitigation |
|---|---|
| Upstream config schemas change (Gemma 4 / Qwen3.5 keys) | Fixtures pinned to commit SHAs; the opt-in network test detects drift |
| Engines implement these architectures differently from the theory (e.g. not exploiting sliding windows) | Expected. memcalc reports *theoretical* numbers, and E15 measures what engines actually do. That gap is the result. |
| PLE / multimodal param accounting is ambiguous | Report all three param views explicitly and state the assumptions in the output |

## 10. "Do it yourself" steps
1. Before running memcalc, compute by hand Qwen3-4B's KV per token and per 8K sequence from its `config.json` (expected: 144 KiB, 1,152 MiB).
2. Explain in your own words why Gemma 4 E4B needs only ~148 MiB per 8K sequence: which layers own KV, and why the window caps 20 of them.

## 11. Evidence to capture → `evidence/M0/`
- memcalc `estimate` outputs for the lineup; the `compare` E15 table.
- Hand calculation vs memcalc comparison.
- First CI run log, plus any failing-then-fixed test output.
- Param-count reconciliation table (analytic vs metadata).

## 12. Explain-back questions (preview)
1. Derive the KV-per-token formula. Why the factor of 2?
2. Why does GQA shrink KV, and by how much for Qwen3-4B vs full MHA?
3. What is a linear-attention layer's "state", and why doesn't it grow with context?
4. What do Gemma 4's KV-shared layers compute, and what do they skip?
5. Why does an MoE model load all experts but read only some per token, and what does that do to the decode ceiling?
6. Why is decode bandwidth-bound? Compute the ceiling for Qwen3-4B bf16 on an L4.
7. What does the reserve cover, and why might vLLM's real KV capacity differ from memcalc's?
