# M2 · GPU engine baseline on Modal (E1, E8, memcalc validation): Report

> **Status:** `DONE` (repo-public step pending your yes) · **Date:** 2026-11-24 · **Branch:** `serve/m2-modal-prep` · **Modal spend:** ≈ $3.80 (credits; $0.00 billed to card)
> **Results:** [`../../RESULTS.md`](../../RESULTS.md) §M2 · **Spec:** [`M2_SPEC.md`](M2_SPEC.md) · **Research:** [`../research/M2_gpu-engines-and-modal.md`](../research/M2_gpu-engines-and-modal.md)

## 1. Summary of what was done

| Deliverable | Status | Evidence |
|---|---|---|
| Modal runner: engine + load client in one container, pinned images, per-run manifest with real GPU | Done | `deploy/modal/bench_app.py` |
| Cost safety: hard timeout per job, staged gates + $245 ceiling, refuse if anything is billed to the card, in-flight reservations | Done | `cortex_serve/spend.py`, `budget_check`, tests |
| CPU-only flag pre-flight for all three engine images | Done | `deploy/modal/check_flags.py` |
| Smoke run (Qwen3-0.6B, vLLM, L4) | Done | `results/gpu-smoke-vllm-l4/` |
| E1 L4: vLLM, SGLang, llama.cpp; 2 workloads; 3 repeats | Done | RESULTS §M2.1, `reports/m2-e1-l4/` |
| E1 H100: vLLM, SGLang (exact H100) + an H200 point | Done | RESULTS §M2.2, `reports/m2-e1-h100/` |
| E8 memory knobs (8 configs in one container) | Done | RESULTS §M2.3, `reports/m2-e8/` |
| memcalc vs engine KV capacity (vLLM + SGLang, 3 GPUs) | Done | RESULTS §M2.4 |
| Cross-check vs `vllm bench serve` | Done | RESULTS §M2.5 |
| Tests | 106 passed (+1 network-only deselected) | `make check` |

## 2. Acceptance criteria (M2_SPEC §5)

| Criterion | Met? | Notes |
|---|---|---|
| Smoke run green (manifest, KV parse, GPU telemetry) | ✅ | KV tokens 166,864 vs memcalc 166,836 |
| E1 curves: 3 engines on L4 + 2 on H100, 3 repeats, manifests | ✅ | 0 request errors in every point |
| memcalc vs vLLM KV table, ±20% on max concurrency | ✅ | −7% to +13% on KV; weights −2% |
| Cross-check within ~5–10% | ✅ | 1–7% |
| Spend logged per run | ✅ | `03_COMPUTE_AND_BUDGET.md` |
| Repo flipped to public | ⏳ | Waiting for your yes (D-022) |

## 3. Key numbers

| | Value |
|---|---|
| Single-user decode, all engines, L4 | 33 ms/token = 80% of the 300 GB/s bandwidth ceiling |
| Single-user decode, H100 | ~4.2 ms/token = 57% of ceiling |
| vLLM vs SGLang, equal hardware | within ~5% (L4), ~2% (H100) on throughput |
| llama.cpp vs vLLM at 16 users, L4 | −25% to −33% throughput |
| Users meeting the GPU SLO (TTFT p95 ≤ 0.5 s, TPOT ≤ 50 ms), chat | L4 ~4 · H100 ~64 |
| SLO-meeting requests per dollar, chat | L4 ~1,900 · H100 ~22,000–25,000 (11–13×) |
| H200 vs H100, same engine | +20–23% throughput |
| memcalc KV error (7 engine/GPU/util combos) | −7% to +13% |
| Harness vs `vllm bench serve` | within 1–7% |

## 4. Deviations from plan
1. **The H100 subset was run twice.** Modal served the first vLLM "H100" request on an H200; vLLM was re-run with `gpu: "H100!"`. The H200 run is kept, labelled.
2. **llama.cpp E1 ran twice.** The first attempt failed at startup (GGUF path mapped to the Docker mount `/models` instead of the Modal volume); fixed in `9cc9e30` with a test.
3. **llama.cpp used an F16 GGUF (unsloth)**, not bf16: no bf16 GGUF of this model is published. Same bytes per weight, so speed is comparable; M3 produces our own conversion.
4. **Native cross-check used vLLM only** (SGLang's bench tool not run): one vendor reference is enough to validate the client.
5. **GPU budget model changed** (D-030): $250 credits expiring 2026-11-30 replaced per-milestone caps with staged gates; M3 GPU work started while M2 was finishing.

## 5. Decisions I made (veto any)

| | Decision | Why |
|---|---|---|
| ⚑ | Configs request **exact GPU types** (`H100!`), and runs whose real GPU differs are flagged | An automatic H200 upgrade nearly produced a false "vLLM is 20% faster" conclusion |
| ⚑ | Requests per load level = max(16, 3 × concurrency) | Keeps high-concurrency points saturated without wasting time at 1 user |
| ⚑ | Several configs in one container (E8) | Saves 3–5 min of cold start per config |
| ⚑ | Parallel GPU jobs allowed, each reserving its worst case | Credits expire in 6 days; the gate check stays conservative |
| | llama.cpp grid capped at 16 users (your choice) | It is not built for high concurrency; 16 already shows the gap |
| | KV capacity reported as a ±5% quantity | vLLM's profiling varies between identical runs |

## 6. Challenges & surprises
1. **Modal swapped an H100 for an H200 without saying so** (same price). Only the manifest's `nvidia-smi` line revealed it. The lesson: record the hardware you *got*, not the hardware you asked for.
2. **Engine CLIs drift between releases:** vLLM 0.32 removed `--disable-log-requests`; the SGLang image has no `python`. Each cost a failed GPU start before the CPU-only flag pre-flight existed.
3. **vLLM's CLI cannot even print `--help` without CUDA**, so the pre-flight checks vLLM flags against its source instead.
4. **An L4 is "cheap" per hour but expensive per answer:** at this model size it is only ~4 users from the latency SLO.
5. **One sequence uses about half of an H100's bandwidth;** the big GPU only pays off with batching.
6. **Modal preempted a job once** (restarted automatically); cost, not correctness.

## 7. Explain-back checklist
1. Why is single-user decode the same on vLLM, SGLang and llama.cpp? What sets it?
2. Compute the L4 decode ceiling for Qwen3-4B bf16. Why is the measured value 80% of it, and why only ~57% on an H100?
3. What is goodput, and why does the L4 lose it at only ~4 users when its throughput keeps rising to 64?
4. Why is the H100 4.9× more expensive per hour but ~12× cheaper per SLO-meeting request?
5. What do continuous batching and chunked prefill change, and why does llama.cpp fall behind at 16 users?
6. Derive Qwen3-4B's KV bytes per token (147,456) and the max sequences at 4,096 tokens on an L4 at util 0.9.
7. In E8, why do util 0.7 and 0.9 give identical throughput at 64 users, while 0.5 halves it?
8. Why does `max_model_len` not change memory use in vLLM? What does it change?
9. What does `max_num_seqs` trade, and when would you set it low?
10. Why was memcalc's error largest at util 0.5?
11. How did you discover the H200, and what would have gone wrong without the manifest?
12. Why measure the client in the same container as the engine? What would an internet hop add?
13. Why does agreement with `vllm bench serve` matter, and why aren't the numbers identical?

## 8. Next: M3 (quantization + quality), already running
FP8 (ours and Qwen's official), GPTQ and AWQ 4-bit, W8A8 INT8, BFSI-calibrated GPTQ, GGUF Q8/Q5/Q4 ± imatrix, and the Gemma QAT four-way test are produced or in progress; quality evals run as artifacts land.
