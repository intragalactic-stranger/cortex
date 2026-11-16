# M2 · GPU engine baseline on Modal (E1, E8, memcalc validation): Spec

> **Status:** `READY, blocked on credentials` (Modal token) · **Branch:** `serve/m2-gpu-baseline` · **Cap:** $40 (D-019 pending your approval)

## 1. Goal
Measure vLLM vs SGLang vs llama.cpp-CUDA serving **Qwen3-4B-Instruct-2507 (bf16)** on an L4, plus a short H100 subset: throughput–latency curves, goodput under the GPU SLO, and memcalc's predicted vs engine-reported KV capacity.

## 2. Scope
**In:** E1 (engine baseline), E8 (memory knobs: `gpu_memory_utilization`, `max_num_seqs`, `max_model_len`), memcalc validation table, cross-check against `vllm bench serve` on 2 configs.
**Out:** quantized artifacts (M3), feature experiments (M4).

## 3. Plan and budget (rates verified on modal.com/pricing before each run)
| Step | GPU | Est. minutes | Est. $ |
|---|---|---|---|
| 0. Smoke: Qwen3-0.6B, vLLM, 2 load levels | L4 | 10 (cap 20) | ~0.15 |
| 1. Fix whatever the smoke run exposes (flags, log formats, image quirks) | L4 | 2 × 10 | ~0.30 |
| 2. E1 L4: 3 engines × `chat-balanced` (512/256) + `rag-long-prefill` (2048/128) × concurrency {1, 4, 16, 32, 64} × 3 repeats | L4 | 3 × ~60 | ~2.50 |
| 3. E8 + memcalc validation (vLLM, startup-only runs + 1 load point each) | L4 | ~30 | ~0.40 |
| 4. E1 H100 subset: vLLM + SGLang, `chat-balanced`, concurrency {1, 16, 64, 128} × 3 | H100 | 2 × ~40 | ~5.30 |
| 5. Native cross-check: `vllm bench serve` on 2 configs | L4 | ~20 | ~0.30 |
| Weight downloads, image pulls, container starts (≈ 3–5 min per job) | — | ~60 | ~1.50 |
| **Total** | | | **~$10–15 expected, $40 cap** |

The 4B primary makes this much cheaper than the spec's 8B plan. The unspent cap rolls into the reserve.

## 4. GPU SLO
TTFT p95 ≤ 500 ms and TPOT ≤ 50 ms (spec default; per-request DistServe goodput).

## 5. Acceptance criteria
- [ ] Smoke run green: manifest, KV-capacity parse, GPU telemetry all present.
- [ ] E1 curves for 3 engines on L4 + 2 engines on H100, 3 repeats, manifests.
- [ ] memcalc vs vLLM KV capacity table with error % and explanation (target ±20% on max concurrency).
- [ ] Cross-check: servebench vs `vllm bench serve` within ~5–10% on 2 configs, or the discrepancy explained.
- [ ] Spend logged per run in `03_COMPUTE_AND_BUDGET.md`.
- [ ] Repo flipped to public (D-012), after you confirm.

## 6. What's needed from you
1. `! uv tool install modal` then `! modal token new` (browser login).
2. The Modal credit balance, and approval of the **$40 M2 cap**.
