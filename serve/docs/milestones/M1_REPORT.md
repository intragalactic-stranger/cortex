# M1 · servebench + CPU baseline (E13, E6 preview): Report

> **Status:** `DONE` · **Dates:** 2026-11-05 → 2026-11-16 · **Branch:** `serve/m1-servebench` · **Cost:** $0 (local CPU)
> **Results:** [`../../RESULTS.md`](../../RESULTS.md) §M1 · **Design:** ADR [0005](../adr/0005-servebench-measurement-design.md), [0006](../adr/0006-cpu-tier-choices.md) · **Research:** [`../research/M1_engines-and-cpu-tier.md`](../research/M1_engines-and-cpu-tier.md)

## 1. Summary of what was done

| Deliverable | Status | Evidence |
|---|---|---|
| `servebench run`: client-side streaming timing (TTFT/ITL/TPOT), closed-loop concurrency + Poisson load, goodput, cgroup sampler, Prometheus scrape, per-run manifest | Done | ADR 0005; `serve/tests/test_servebench_*.py` |
| Engine adapters: llama.cpp server (Docker, pinned digest) and Ollama | Done | `servebench/engines/` |
| `servebench quality`: WikiText-2 perplexity (`llama-perplexity`) + ARC-Challenge via the serving API, Wilson CIs | Done | `servebench/quality/` |
| `servebench report`: median-of-repeats tables, CSV, plots | Done | `results/reports/m1-*` |
| memcalc reads GGUF headers (exact weight bytes per tensor type) | Done | `memcalc/gguf.py`; `evidence/M1/04_*` |
| Machine bandwidth probe (`scripts/membw.c`) | Done | `evidence/M1/01_*`, `02_*` |
| E13: 8 GGUF artifacts × llama.cpp (+ 2 on Ollama), llama-bench, quality | Done | RESULTS §1–4 |
| E6 CPU preview: speculative decoding | Done (after a harness fix) | RESULTS §5 |
| Speculative-decoding harness bug fixed + guard against silent no-ops | Done | commit `75d6f73` |
| Tests | 93 passed (+1 network-only deselected) | `make check` |

## 2. Acceptance criteria (spec §13, M1)

| Criterion | Met? | Notes |
|---|---|---|
| servebench runs against llama.cpp and Ollama on CPU | ✅ | 0 request errors across all points |
| Manifests | ✅ | Every run dir has `manifest.json` with image digest, artifact sha256, tokenizer revision, git SHA (clean) |
| Report generator | ✅ | `servebench report` |
| E13 done | ✅ (adapted) | Spec asked for 1.7B and 8B; D-014 replaced 8B with Gemma 4 E2B and added 0.6B for this 15 GiB CPU box |

## 3. Key numbers

| | Value |
|---|---|
| Measured memory bandwidth (4 threads) | 20.3 GB/s |
| Decode bandwidth efficiency, Q4 / Q8 | 45–57% / 66–83% of 20.3 GB/s |
| Q4_K_M vs Q8_0 decode speed-up (Qwen) | 1.15–1.19× (byte ratio 1.65×) |
| Q4_K_M vs Q8_0 perplexity cost (Qwen) | +3.7–4.4% |
| Only config meeting CPU SLO (TTFT ≤ 3 s, TPOT ≤ 125 ms) | Qwen3-0.6B Q4_K_M, 1 user: TTFT 2.35 s, TPOT 43 ms |
| Concurrency 1 → 4 | +6–13% throughput, TPOT ×3.0–3.7 |
| Speculative decoding (1.7B + 0.6B draft) | TPOT 93 → 147 ms (1.57× slower) at 99.2% acceptance |
| memcalc vs llama.cpp KV | Qwen exact (896 MiB); Gemma full-attention layers exact, sliding layers 2× (window + ubatch) |
| Gemma 4 prefill with 4 slots vs 1 | 13 vs 39 tok/s |

## 4. Deviations from plan
1. **No separate M1 spec file was written before building.** Under autonomous mode the design went straight into ADRs 0005/0006 and the research note. They served as the spec; this report checks against spec §13.
2. **The measurement run was interrupted and resumed.** The 2026-11-05 pipeline died at quality artifact 4/8 when the owning Claude session ended (log stops with no error). Resumed 2026-11-16 at the same commit `ac60f58`, re-running only the 5 missing artifacts (D-023). Quality is therefore split across two run dirs with identical config and SHA.
3. **E6 was run twice.** The first run is invalid (draft never drafted) and is kept with an `INVALID.md` marker. The valid baseline + draft pair ran back to back at `75d6f73`.
4. **Peak-RAM column not used** for memory claims (ADR 0005 amendment).

## 5. Decisions I made (veto any)

| | Decision | Why |
|---|---|---|
| ⚑ | CPU SLO: TTFT ≤ 3 s, TPOT ≤ 125 ms (closes G-04) | ≥ 8 tok/s outpaces reading; the GPU SLO (500 ms / 50 ms) is meaningless on this box |
| ⚑ | ARC answer naming several options = wrong | Prevents rewarding hedging; inflates "unparsed" for 0.6B, which is reported |
| ⚑ | Resume only the missing quality artifacts instead of a full rerun (D-023) | Same SHA/config; quality metrics aren't timing-sensitive; saves ~1 h |
| ⚑ | servebench fails a run when a configured draft model drafts 0 tokens (D-024) | A silent no-op produced plausible-looking but wrong numbers once |
| | Memory claims from engine-reported sizes, not cgroup peak (D-025) | mmap + shared page cache makes cgroup peak unreliable |
| | Invalid runs kept on disk with a marker, not deleted | Honest record; the failure is part of the story |
| | SGLang pinned to v0.5.22, vLLM v0.32.0 for M2 (made during M1 prep) | Avoid a 3-day-old release for the first GPU runs |

## 6. Challenges & surprises
1. **The benchmark tool reported a feature that never ran.** With current llama.cpp, `-md` loads a draft model but `--spec-type` defaults to `none`. The first E6 run looked plausible (slightly *faster*, within noise). The missing `draft acceptance` log line gave it away. Fixed, and servebench now verifies the feature is active.
2. **Speculative decoding made the CPU slower.** At 99% acceptance it still cost 1.57× per token: on a compute-bound CPU, verifying k tokens costs ~k/1.9 decode steps, not ~1.
3. **Gemma 4 E2B's QAT file scored 3× better perplexity than unsloth's near-lossless Q8_0.** Tested and ruled out the structural difference (60 unused K/V tensors): PPL identical to 4 decimals after removing them. The two files carry different weights; resolve in M3 with bf16 originals.
4. **Gemma 4 prefill is 3× slower with 4 server slots than with 1,** even for a single request; Qwen isn't affected. A serving config choice changed a model comparison.
5. **"E2B" decodes slower than a 1.7B model.** Per-layer embeddings are free per token, but Gemma still streams 1.31 GiB per token (incl. a 262k-vocab head) vs 1.03 GiB.
6. **cgroup memory lies about mmapped weights.** Q8_0 showed less peak RAM than Q4_0.
7. **First bandwidth probe under-reported by 40%** (unvectorized read loop). Measure the instrument before trusting it.
8. **The 4–5 h run died with the session.** Long runs are now launched detached (`setsid`).

## 7. Explain-back checklist (answer without notes)
1. Define TTFT, TPOT and ITL. Why measure them at the client, and why re-count tokens with the model's own tokenizer?
2. What is goodput (DistServe), and why is it per-request rather than an average?
3. Why do 4 concurrent users barely raise throughput on this CPU, when on a GPU they nearly quadruple it?
4. Why is Q4_K_M only ~1.17× faster than Q8_0 here, when it reads 1.65× fewer bytes?
5. Compute the decode ceiling for Qwen3-1.7B Q4_K_M from the 20.3 GB/s figure. Why is the measured 10.6 tok/s about half of it?
6. Why does Gemma 4 E2B (3 GB file) read fewer bytes per token than its file size, yet more than Qwen3-1.7B?
7. What must be true for speculative decoding to help? Why did it hurt here, even at 99% acceptance?
8. How did you find out the first E6 run never drafted? How does servebench prevent that now?
9. Why can't you compare Gemma's and Qwen's perplexity? What did you use instead?
10. Why is the Gemma QAT vs PTQ perplexity comparison confounded, and what experiment ruled out the K/V tensors?
11. Why is cgroup peak memory wrong for llama.cpp, and what did you use instead?
12. Derive Qwen3-1.7B's KV for 4 slots × 2,048 tokens (896 MiB). Why does llama.cpp give Gemma's sliding layers twice memcalc's number?
13. Why is ARC at n = 100 unable to separate Q4 from Q8? What is a Wilson interval?

## 8. Next: M2 (GPU engine baseline on Modal, cap $40)
Blocked only on the Modal login (`modal token new`). The M2 code is written and unit-tested on `serve/m2-modal-prep`; the speculative-decoding fix will be merged into it. First step: a ~$0.15 L4 smoke run with Qwen3-0.6B.
