# M4 · Feature experiments (E4–E7, E9, E10, E12, E15): Report

> **Status:** `DONE` · **Date:** 2026-11-24 → 25 · **Branch:** `serve/m3-quant` (M3 + M4) · **Modal spend:** ≈ $11 (credits; card $0.00)
> **Results:** [`../../RESULTS.md`](../../RESULTS.md) §M4

## 1. Summary of what was done

| Deliverable | Status | Evidence |
|---|---|---|
| servebench: real-prompt workloads (UltraChat), natural stopping, JSON-schema constraint + validity, truncation rate, vLLM spec-decode and prefix-cache counters | Done | `servebench/{workloads,runner,collect}.py`, tests |
| E4 prefix caching (vLLM APC, SGLang RadixAttention) | Done | `results/e4-*`, `reports/m4-e4/` |
| E5 chunked prefill on/off | Done | `results/e5-*`, `reports/m4-e5/` |
| E6 speculative decoding: Qwen3-0.6B draft and n-gram, real prompts | Done | `results/e6-*`, `reports/m4-e6/` |
| E7 FP8 KV cache: capacity, long-context speed, quality | Done | `results/e7-*`, `quality-gpu/qwen3-4b-bf16-kv-fp8/` |
| E9 MoE Qwen3-30B-A3B vs Qwen3-4B on H100: speed + quality | Done | `results/e9-*`, `quality-gpu/qwen3-30b-a3b-bf16/` |
| E10 structured output (prompted vs constrained JSON) | Done (rerun) | `results/e10-*` |
| E12 tensor parallel TP=2 | Done on 2 x H100 | `results/e12-*` |
| E15 architecture study: Qwen3-4B vs Qwen3.5-4B vs Gemma 4 E4B, vLLM + SGLang, 8K/32K, quality | Done | `results/e15-*`, `reports/m4-e15/` |
| E6 native MTP heads, llama.cpp prompt cache (local) | Not done | Optional in the plan; skipped for time |
| E11 multi-LoRA | Deferred (G-14) | — |

## 2. Acceptance criteria (Phase 1 plan, M4)

| Criterion | Met? | Notes |
|---|---|---|
| Each experiment has a plot | ✅ | `results/reports/m4-*` (E10/E12 tables in RESULTS) |
| Findings paragraph stating only what the data shows | ✅ | RESULTS §M4 |
| Manifest-backed config | ✅ | `configs/gpu/e{4,5,6,7,9,10,12,15}_*.yaml`; every run dir has `manifest.json` with the real GPU |

## 3. Key numbers

| | Value |
|---|---|
| Prefix caching, 75% shared prompt | TTFT 3× faster at 1 user; +73% throughput at 16 |
| Chunked prefill, 4 long-prompt users | ITL p99 365 → 47 ms |
| Speculative decoding (0.6B draft) | 1.27× faster at 1 user; −6% throughput at 16; n-gram 2% acceptance on chat |
| FP8 KV | 1.92× KV tokens; +46% long-context throughput; no measurable quality cost |
| MoE 30B-A3B vs 4B dense (H100) | 4.7 vs 4.2 ms/token at 1 user; 0.48× throughput at 64; MMLU-Pro +12.4, GSM8K +11.1 |
| JSON schema | 100% valid prompted and constrained; ≤0.5% TPOT overhead |
| TP=2 on 2 × H100 | 1.37× faster tokens, +47% goodput at 64, −27% goodput per dollar |
| E15 memory per 8K sequence (vLLM) | Qwen3-4B 1,152 MiB · Qwen3.5-4B 363 MiB · Gemma 4 E4B 309 MiB |
| E15 long-context throughput at 32 users | 68 · 109 · 116 tok/s |

## 4. Deviations from plan
1. **E12 ran on 2 x H100**, not 2 x L4: Modal never scheduled the L4 pair (1 h 45 min queue, not billed) (D-042).
2. **E6 used real UltraChat prompts with natural stopping** instead of the synthetic word-list workload, which would make draft acceptance meaningless.
3. **E10 was rerun** with 400-token answers; the first run's 12.5% truncation made both modes look invalid (D-043).
4. **E15 and FP8-KV quality evals added** beyond the plan (D-045): a decision guide needs quality next to speed.
5. **Optional items skipped:** native MTP heads, llama.cpp prompt cache locally.

## 5. Decisions I made (veto any)

| | Decision | Why |
|---|---|---|
| ⚑ | E15 compares engine-reported max concurrency at a given context, not "KV tokens" | Hybrid/SWA engines report token capacities that depend on `max_model_len` |
| ⚑ | E12 on 2 x H100 compared with E2's H100 TP=1 run | Same workload and image; no extra baseline run needed |
| ⚑ | Prefix-cache hit rate and spec-decode acceptance read from vLLM's Prometheus counters per load point | Shows the feature actually engaged (lesson from M1's silent spec-decode no-op) |
| | One container per experiment batch, batches in parallel | Credits deadline; reservations keep the gate check conservative |

## 6. Challenges & surprises
1. **Speculative decoding flips sign between CPU and GPU:** 1.57× slower on M1's CPU, 1.27× faster on the L4 at one user, and gone again by 16 users.
2. **MoE is a latency play, not a throughput play:** at batch, the union of routed experts approaches the whole model.
3. **Engines realise only part of the new architectures' KV savings,** and Gemma's 14.5 GiB of weights eat its saving on a 24 GB card.
4. **SGLang lags vLLM on the new architectures** in some cells (Qwen3.5 32-user chat; Gemma long context). Not investigated.
5. **Modal capacity:** ~12 parallel jobs queued others; multi-GPU L4 never scheduled.
6. **Disabling chunked prefill is "not officially supported"** for this model in vLLM 0.32, but it ran cleanly.

## 7. Explain-back checklist
1. Why does prefix caching cut TTFT but not TPOT? What hit rate did you expect and why?
2. APC vs RadixAttention: how does each find a reusable prefix?
3. Why does chunked prefill protect the inter-token tail, and why does it stop helping at 16 long-prompt users?
4. When does speculative decoding help, and why does the gain vanish at batch 16? Why is n-gram useless for chat?
5. What does FP8 KV change in memory and compute, and what quality risk did you not measure?
6. Why is the MoE almost as fast as the dense 4B at one user but half as fast at 64?
7. What does TP=2 add per layer, and why is it −27% per dollar for a 4B model?
8. Why is constrained JSON decoding nearly free here, and what was the real failure mode?
9. Derive Qwen3.5-4B's per-sequence memory (fixed recurrent state + full-attention KV) and compare with vLLM's.
10. Why does Gemma 4 E4B fit fewer 8K sequences than Qwen3.5-4B despite smaller KV per token?

## 8. Next
M5 (local k3d) is built on `serve/m5-k3d`; remaining GPU item: the Modal vLLM endpoint as the gateway's fallback demo. Then M6 (site) and M7 (decision guide) after the credits expire.
