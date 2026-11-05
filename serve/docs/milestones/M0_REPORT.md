# M0 · Foundations: Report

> **Closed on:** 2026-11-05 · **Tag:** `serve-m0` · **Merge commit:** `86b55a4` (PR #1) · **Modal spend:** $0
> **Mode:** autonomous (D-015). Calls marked ⚑ are worth your review.

## 1. Summary of what was done

| # | Deliverable | Status | Evidence |
|---|---|---|---|
| 1 | Toolchain: uv 0.5.9 → **0.13.7**, Python **3.12.8** via uv | ✅ | `docs/VERSIONS.md` |
| 2 | Private GitHub repo `intragalactic-stranger/cortex`, uv workspace, `serve` package | ✅ | PR #1, tag `serve-m0` |
| 3 | Quality gates: ruff, `mypy --strict`, pytest, pre-commit + gitleaks, GitHub Actions CI | ✅ green | `evidence/M0/08_*` |
| 4 | **memcalc** core: per-layer memory model (full / sliding / KV-shared / linear-state), MoE | ✅ | `src/cortex_serve/memcalc/` |
| 5 | Exact params from **safetensors headers** via HTTP range requests (no weight download) | ✅ | `hub.py`, `evidence/M0/11_*` |
| 6 | CLI: `estimate`, `compare`, `devices`, `--json` | ✅ | `make demo` |
| 7 | Device table (10 devices incl. this CPU box), sources cited | ✅ | `devices.yaml` |
| 8 | **67 tests**: golden hand calculations, config ↔ tensor-shape cross-checks, properties, CLI (+1 opt-in network test) | ✅ 67/67 | `evidence/M0/09_*` |
| 9 | 8 learning docs with memcalc-generated worked examples + interview questions | ✅ | `docs/learn/` |
| 10 | ADRs 0001–0004, `SPEC.md` (with adaptation notes), `VERSIONS.md`, README, RESULTS stub | ✅ | `docs/` |
| 11 | Fresh-clone check: `make setup && make check && make demo` | ✅ 17 s | `evidence/M0/10_*` |

## 2. Acceptance criteria

| AC | Result | Proof |
|---|---|---|
| AC1 clean clone: setup, lint, test pass; CI green | ✅ | fresh clone 17 s; PR #1 checks green |
| AC2 `estimate` prints all spec §4 outputs; `--json` | ✅ | `test_estimate_json`, `evidence/M0/06_*` |
| AC3 golden vectors exact | ✅ **all exact to the byte** | `golden.json`, `evidence/M0/12_*` |
| AC4 decode ceilings vs spec §3.3 (±5%) | ✅ with a documented difference: L4 19.99 tok/s (spec ≈19). H100 **223** tok/s vs spec ≈200, because memcalc doesn't count the input-embedding table as read per token; counting it gives 208.6 (within 5%) | `golden.json` note |
| AC5 analytic params within ±2% of header | ✅ **exact** (after excluding Qwen3-0.6B/1.7B's redundant `lm_head`); hybrids broken down by component | `evidence/M0/11_*` |
| AC6 `compare` prints the E15 table | ✅ | `make demo` |
| AC7 8 learning docs | ✅ | `docs/learn/01..08` |
| AC8 ADRs, VERSIONS, gitleaks clean | ✅ | `docs/adr/`, CI `secrets` job |
| AC9 private repo created and pushed | ✅ | github.com/intragalactic-stranger/cortex (private) |

## 3. Key numbers (theoretical; M1/M2/E15 will measure)

| Model | Device | Weights | Memory / 8K seq | Max 8K seqs | 1-stream decode ceiling |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 bf16 | L4 | 7.49 GiB | 1,152 MiB | 9 (19 with FP8 KV) | 37 tok/s |
| Qwen3-4B int4 W4A16 | L4 | 2.47 GiB | 1,152 MiB | 14 | 113 tok/s |
| Qwen3.5-4B bf16 | L4 | 7.83 GiB | 305 MiB | 36 | — |
| Gemma 4 E4B bf16 | L4 | 14.00 GiB (8.75 with PLE offload) | 148 MiB | 32 (69 with PLE offload) | 32 tok/s |
| Qwen3-30B-A3B bf16 | H100 | 56.87 GiB | 768 MiB | 17 | 551 tok/s |

## 4. Deviations from the M0 spec
- **No `huggingface_hub` dependency.** I used `httpx` directly, because range requests on safetensors headers were needed anyway (one fewer dependency).
- **Fixtures** are a directory per model (`config.json` + gzipped tensor table + `meta.json` with the pinned SHA) rather than flat config files, because exact params need the tensor table.
- **The AC4 spec number is off.** The spec says ≈200 tok/s; memcalc reports 223 for the reason in §2. It's documented rather than forced to match.

## 5. Decisions I made (veto any)

| | Decision | Why |
|---|---|---|
| ⚑ | Git identity is **repo-local**, using your GitHub **noreply** address (`63484928+intragalactic-stranger@users.noreply.github.com`) | Commits are attributed to your account without publishing your personal email |
| ⚑ | Gemma 4 PLE tables are counted as **device memory by default** (`--ple-offload` to change) | Conservative. It swings Gemma's L4 capacity 32 ↔ 69 sequences, and E15/M2 will measure what engines actually do |
| ⚑ | memcalc's decode ceiling excludes input-embedding lookups (223 vs spec's ~200 tok/s for 8B on H100) | More accurate model of bytes read per token; difference documented |
| | Params from safetensors headers; tied-but-stored `lm_head` excluded from loaded params | Exact and verifiable |
| | llm-compressor formats keep embeddings/lm_head in 16-bit; GGUF quantizes them | Matches how those tools behave |
| | Default runtime reserve 1.5 GiB (GPU) / 0.5 GiB (CPU), refit in M2 | D-011 |
| | Apache-2.0 · hatchling · src layout · `cortex_serve` · merge commits (history kept) | Conventional |

## 6. Challenges & surprises (journal: 2026-11-05)
1. **Analytic params failed for Qwen3-0.6B/1.7B.** The checkpoints store `lm_head` even though the config says the embeddings are tied, so HF's "752M params" includes a 156M duplicate (596M unique). memcalc now detects stored-but-unused duplicates.
2. **Gemma 4 has fewer free sequences than its tiny per-sequence KV suggests.** Its 2.8B-param PLE table competes for the same memory, so where the engine places PLE matters more than the KV design.
3. **First CI run failed on the secrets job.** It wasn't a leak: gitleaks-action got HTTP 403 because I'd restricted the token to `contents: read`, and the action needs `pull-requests: read` to list PR commits. Fixed with least privilege kept (`05dced5`).
4. **A10G/T4/A100 have no FP8 compute.** memcalc now says so explicitly instead of silently giving FP8 speed-ups those GPUs can't deliver.

## 7. Explain-back checklist (answer without notes)
1. Derive KV bytes per token. Why the factor 2? Compute Qwen3-4B's value (144 KiB).
2. Why does GQA (8 KV heads vs 32) cut KV 4×, and what does it cost?
3. Why does Gemma 4 E4B need only 148 MiB per 8K sequence? Which layers own KV, and why does the window cap 20 of them?
4. What is the Gated DeltaNet "state", why doesn't it grow, and why is it fp32?
5. Why does Qwen3-30B-A3B need ~57 GiB but read only ~3B params per token?
6. Compute the decode ceiling for Qwen3-4B bf16 on an L4. Why does int4 roughly triple it?
7. What is the roofline crossover batch, and why does int4 reach it at batch ~104 instead of ~403?
8. Why does FP8 KV double max concurrency while FP8 weights don't?
9. How does memcalc count params without downloading weights?
10. Why did Qwen3-0.6B's analytic count disagree with Hugging Face's number?

## 8. Next: M1 (servebench + CPU baseline, local, $0)
Async streaming client (TTFT/ITL/E2E), workloads, load patterns, manifests, report generator; llama.cpp and Ollama launchers (Docker, pinned digests); E13 on Qwen3-0.6B / 1.7B / Gemma 4 E2B (incl. Gemma official QAT Q4_0 vs Q4_K_M); measure this CPU's memory bandwidth; first memcalc validation on llama.cpp.
