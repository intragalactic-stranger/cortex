# Results

Every number here must be reproducible from a committed config, with a manifest. Measured results start in M1.

## M0: memcalc predictions (theoretical, not yet measured)

| Model | Device | Weights | Memory / 8K seq | Max seqs @8K | Source |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | L4 | 7.49 GiB | 1,152 MiB | 9 | `make demo` |
| Qwen3.5-4B | L4 | 7.83 GiB | 305 MiB | 36 | `make demo` |
| Gemma 4 E4B | L4 | 14.00 GiB (8.75 with PLE offload) | 148 MiB | 32 (69 with PLE offload) | `memcalc estimate` |
| Qwen3-30B-A3B | H100-80GB | 56.87 GiB | 768 MiB | 17 | `memcalc estimate` |

These become hypotheses for M1 (llama.cpp), M2 (vLLM capacity at startup) and E15.

---

## M1: CPU tier (E13) and speculative-decoding preview (E6), measured

**Hardware:** AWS VM, AMD EPYC 7571 (Zen 1), 2 cores / 4 threads, AVX2+FMA, 15 GiB RAM; measured memory bandwidth **20.3 GB/s** (`serve/scripts/membw.c`, 4 threads).
**Engines:** llama.cpp `server`/`full` images pinned by digest; Ollama 0.40.14 (see `docs/VERSIONS.md`). 4 threads everywhere.
**Workload:** `chat-short`, 128 input / 64 output tokens (exact, `ignore_eos`), concurrency 1 and 4, 8 requests per point, 3 repeats, median shown.
**CPU SLO (G-04):** TTFT ≤ 3 s and TPOT ≤ 125 ms (≥ 8 tok/s).
**Runs:** `results/e13-llama-bench/20261105T060207Z`, `results/e13-cpu-llamacpp/20261105T061823Z`, `results/e13-cpu-ollama/20261105T081019Z`, `results/e13-cpu-llamacpp-quality/{20261105T082630Z,20261116T043401Z}`, `results/e6-cpu-baseline/20261116T072042Z`, `results/e6-cpu-spec-decoding/20261116T072642Z`. Full tables: `results/reports/m1-e13/`, `results/reports/m1-e6/`.

### 1. Engine-internal speed (llama-bench, 3 reps)

| GGUF | File GiB | Streamed GiB/token¹ | pp128 tok/s | tg64 tok/s | Effective GB/s² |
|---|---|---|---|---|---|
| Qwen3-0.6B Q4_K_M | 0.36 | 0.36 | 61.6 | 25.3 | 9.9 |
| Qwen3-0.6B Q8_0 | 0.59 | 0.59 | 45.4 | 21.2 | 13.4 |
| Qwen3-1.7B Q4_K_M | 1.03 | 1.03 | 18.6 | 10.6 | 11.6 |
| Qwen3-1.7B Q8_0 | 1.70 | 1.70 | 15.0 | 9.2 | 16.8 |
| Gemma 4 E2B QAT Q4_0 (Google) | 3.10 | 1.31 | 40.1 | 6.6 | 9.3 |
| Gemma 4 E2B PTQ Q4_0 (unsloth) | 2.82 | 1.31 | 40.7 | 7.3 | 10.2 |
| Gemma 4 E2B Q4_K_M | 2.88 | 1.37 | 47.2 | 7.7 | 11.4 |
| Gemma 4 E2B Q8_0 | 4.69 | 2.36 | 34.1 | 5.9 | 15.1 |

¹ All tensor bytes except embedding tables that are only looked up (Gemma's token and per-layer embeddings; Qwen3-0.6B's tied embedding is also the output head, so it is streamed). From GGUF headers.
² Streamed bytes × tg64 tok/s.

**Findings**
- **Decode does not reach the bandwidth roof on this CPU.** Q4 files reach 9–12 GB/s (45–57% of 20.3 GB/s); Q8_0 reaches 13–17 GB/s (66–83%). Dequantizing 4-bit blocks costs compute that this 2-core AVX2 CPU doesn't have to spare.
- **So Q4 buys less speed than its size suggests:** Q4_K_M is 1.15–1.19× faster than Q8_0 for Qwen (decode), against a 1.65× byte ratio.
- **Gemma 4 E2B decodes slower than Qwen3-1.7B (6.6–7.7 vs 10.6 tok/s) despite "2B effective".** Its 1.5 GiB of per-layer embeddings are lookups and cost nothing per token, but the rest still streams 1.31 GiB per token (incl. a 262k-vocab output head) vs 1.03 GiB for Qwen3-1.7B. Prefill is faster on Gemma (40 vs 19 tok/s); we do not yet have a verified explanation for that and make no claim.

### 2. Served latency and throughput (servebench, client-side)

| Artifact | Engine | Conc. | Out tok/s | TTFT p50 (s) | TPOT p50 (ms) | Meets CPU SLO |
|---|---|---|---|---|---|---|
| Qwen3-0.6B Q4_K_M | llama.cpp | 1 | 12.6 | 2.35 | 43 | **yes (100%)** |
| Qwen3-0.6B Q4_K_M | Ollama | 1 | 12.7 | 2.33 | 43 | **yes (100%)** |
| Qwen3-0.6B Q4_K_M | llama.cpp | 4 | 14.1 | 8.95 | 147 | no |
| Qwen3-0.6B Q8_0 | llama.cpp | 1 | 9.9 | 3.13 | 52 | no (TTFT) |
| Qwen3-1.7B Q4_K_M | llama.cpp | 1 | 4.7 | 7.02 | 100 | no |
| Qwen3-1.7B Q4_K_M | Ollama | 1 | 5.0 | 6.74 | 96 | no |
| Qwen3-1.7B Q4_K_M | llama.cpp | 4 | 5.0 | 27.33 | 368 | no |
| Qwen3-1.7B Q8_0 | llama.cpp | 1 | 3.8 | 9.07 | 126 | no |
| Gemma 4 E2B QAT Q4_0 | llama.cpp | 1 | 3.2 | 10.31 | 148 | no |
| Gemma 4 E2B Q4_K_M | llama.cpp | 1 | 3.6 | 9.22 | 138 | no |
| Gemma 4 E2B Q8_0 | llama.cpp | 1 | 2.8 | 12.06 | 170 | no |

(Every artifact × concurrency × engine: `results/reports/m1-e13/report.md`. Zero request errors across all points.)

**Findings**
- **Only Qwen3-0.6B Q4_K_M at one user meets the interactive CPU SLO.** Qwen3-1.7B misses on TTFT (prefill ~20 tok/s → 7 s for a 141-token prompt) and Gemma 4 E2B on both TTFT and TPOT.
- **Concurrency barely helps on this CPU:** 4 users raise total throughput by only 6–13% while TPOT rises 3.0–3.7×. On a GPU, batching is nearly free during decode because it is bandwidth-bound; here the extra sequences compete for the same compute.
- **Gemma 4 prefill is 3× slower when llama-server has 4 slots configured,** even with one active request: 13 tok/s with `-np 4` vs 39 tok/s with `-np 1` (llama-bench: 40). Context size and `--swa-full` make no difference, and Qwen3 is unaffected (`evidence/M1/06_*`). Gemma's TTFT above (9–12 s) is therefore a 4-slot number; single-slot it would be ≈ 3.8 s, and TPOT (148 ms) would still miss the SLO. Root cause inside llama.cpp not yet identified. Lesson: **the serving config can flip a model comparison**, so benchmark each model in the config you'd deploy.
- **Ollama ≈ llama.cpp** in speed (within 7%; Ollama embeds llama.cpp). It's a packaging choice, not a performance one.

### 3. Quality (WikiText-2 perplexity, 20 × 512 tokens; ARC-Challenge, n = 100, Wilson 95% CI)

| Artifact | PPL ± err | ARC acc [95% CI] | ARC unparsed |
|---|---|---|---|
| Qwen3-0.6B Q4_K_M | 24.32 ± 1.17 | 0.35 [0.26, 0.45] | 23 |
| Qwen3-0.6B Q8_0 | 23.30 ± 1.11 | 0.28 [0.20, 0.37] | 15 |
| Qwen3-1.7B Q4_K_M | 18.57 ± 0.96 | 0.65 [0.55, 0.74] | 11 |
| Qwen3-1.7B Q8_0 | 17.90 ± 0.92 | 0.68 [0.58, 0.76] | 8 |
| Gemma 4 E2B QAT Q4_0 (Google) | 65.48 ± 3.85 | 0.67 [0.57, 0.75] | 5 |
| Gemma 4 E2B PTQ Q4_0 (unsloth) | 194.90 ± 14.27 | 0.63 [0.53, 0.72] | 9 |
| Gemma 4 E2B Q4_K_M | 214.74 ± 16.19 | 0.68 [0.58, 0.76] | 6 |
| Gemma 4 E2B Q8_0 | 211.98 ± 16.12 | 0.68 [0.58, 0.76] | 5 |

**Findings**
- **Q4_K_M vs Q8_0 costs Qwen 3.7–4.4% perplexity;** ARC differences are inside the confidence intervals at n = 100. A multiple-choice benchmark at this size cannot see a Q4 vs Q8 difference.
- **Perplexity is not comparable across tokenizers** (Gemma 262k vocab vs Qwen 152k), and both are instruction-tuned models scored on raw text. Use ARC across families.
- **Gemma QAT vs PTQ by perplexity is confounded.** Google's QAT file scores 65 while unsloth's Q8_0 (near-lossless) scores 212, so the gap is not quantization error. We tested the obvious structural difference: unsloth's file stores 60 extra K/V tensors for the 20 KV-shared layers; removing them leaves PPL at 211.9807, identical to 4 decimals (`evidence/M1/05_*`). The remaining explanation is that the two files come from different weights (Google's QAT checkpoint vs the base instruct checkpoint). Settling it needs the bf16 originals (M3). On ARC, QAT Q4_0 0.67 vs PTQ Q4_0 0.63 is within noise.
- **Qwen3-0.6B ARC is mostly a format-following score:** 15–23% of answers were unparseable (an answer naming several options counts as wrong).

### 4. Memory: memcalc vs llama.cpp (4 slots × 2,048 tokens, f16 KV)

| Model | memcalc KV | llama.cpp KV | Note |
|---|---|---|---|
| Qwen3-0.6B / 1.7B | 4 × 224 = 896 MiB | 896 MiB | exact (28 layers × 8 KV heads × 128 × 2 × 2 B × 8,192 cells) |
| Gemma 4 E2B | 48 (full) + 24 (sliding) = 72 MiB | 48 + 48 = 96 MiB | full layers exact; sliding-window layers get window + one 512-token ubatch per sequence |

Model sizes from GGUF headers match llama.cpp's reported file size to the MiB. The container cgroup's *peak RAM* is **not** used for memory claims: weights are `mmap`ed, and page-cache pages already resident on the host are not charged to the new container (Gemma Q8_0 showed 657 MiB peak vs 1,847 MiB for the smaller Q4_0).

### 5. E6 preview: speculative decoding on CPU (Qwen3-1.7B Q4_K_M target, Qwen3-0.6B Q4_K_M draft, 1 user)

| Config | Out tok/s | TTFT p50 (s) | TPOT p50 (ms) | Draft acceptance |
|---|---|---|---|---|
| Baseline | 5.0 | 6.87 | 93 | — |
| + draft (`--spec-type draft-simple`) | 3.5 | 9.21 | 147 | 99.2% |

**Findings**
- **Speculative decoding is 1.57× *slower* per token on this CPU, even at 99% acceptance.** It relies on verifying k drafted tokens for about the price of one decode step, which holds when decode is bandwidth-bound (GPU). Here a batched forward pass is only ~1.9× cheaper per token than single-token decode (prefill 18.6 vs decode 10.6 tok/s), so verifying 4 tokens costs ~2 decode steps, plus the draft model's own passes. TTFT also rises 2.3 s (the draft prefills the prompt too).
- The 99.2% acceptance is likely inflated by the synthetic prompts (`ignore_eos` output becomes repetitive); a natural-language prompt gave 94% (46/49). High acceptance is the best case for speculative decoding, so the slowdown conclusion holds.
- **A first E6 run (`20261116T065339Z`) is invalid:** current llama-server defaults `--spec-type` to `none`, so `-md` loaded the draft model and never used it. Fixed in `75d6f73`; servebench now rejects a draft model without `spec_type` and fails a run that drafts 0 tokens.

---

## M2: GPU engine baseline on Modal (E1, E8, memcalc validation), measured

**Model:** Qwen3-4B-Instruct-2507 @ `cdbee75f`, bf16 (vLLM, SGLang) / F16 GGUF from unsloth @ `a06e946b` (llama.cpp; same bytes per weight).
**Engines (pinned by digest, `deploy/modal/bench_app.py`):** vLLM 0.32.0, SGLang 0.5.22, llama.cpp `server-cuda-b12110`. Load client in the **same container** as the engine (G-05), so TTFT has no internet in it.
**GPUs (from `nvidia-smi` in each manifest):** L4 24 GB ($0.80/h); H100 80GB HBM3 ($3.95/h); one H200 run (see the correction below).
**Workloads:** `chat-balanced` 512 in / 256 out, `rag-long-prefill` 2,048 in / 128 out (exact lengths, `ignore_eos`). Closed-loop concurrency; requests per level = max(16, 3 × concurrency); 3 repeats; medians shown.
**GPU SLO:** TTFT ≤ 500 ms and TPOT ≤ 50 ms per request; **goodput** = requests/s that met both (DistServe).
**Runs:** `results/e1-l4-{vllm,sglang,llamacpp}/`, `results/e1-h100-{vllm,sglang}/`, `results/e8-l4-vllm-*/`, `results/m2-crosscheck/`. Full tables and plots: `results/reports/m2-e1-l4/`, `m2-e1-h100/`, `m2-e8/`.

### 1. E1 on L4: three engines

| Workload | Users | Out tok/s vLLM / SGLang / llama.cpp | TTFT p95 (ms) | TPOT p50 (ms) | Goodput req/s |
|---|---|---|---|---|---|
| chat | 1 | 30 / 30 / 30 | 106 / 115 / 187 | 33 / 33 / 33 | 0.12 / 0.12 / 0.12 |
| chat | 4 | 110 / 109 / 97 | 366 / 394 / 700 | 35 / 35 / 39 | **0.43 / 0.43** / 0 |
| chat | 16 | 353 / 348 / 268 | 1,301 / 1,441 / 2,325 | 42 / 42 / 55 | 0.34 / 0.09 / 0 |
| chat | 64 | 792 / 749 / — | 4,614 / 5,535 / — | 76 / 73 / — | 0 / 0 / — |
| rag | 1 | 27 / 27 / 26 | 370 / 399 / 600 | 34 / 34 / 34 | **0.21 / 0.21** / 0 |
| rag | 16 | 171 / 162 / 115 | 4,801 / 5,699 / 8,517 | 83 / 71 / 123 | 0 / 0 / 0 |

(llama.cpp ran 1–16 users by design: 16 slots × 4,096 context.) 0 errors in all points.

**Findings**
- **Single-user decode is the same on all three engines (33 ms/token)** and is **80% of the L4's bandwidth ceiling**: 300 GB/s ÷ 8.05 GB of weights read per token = 37 tok/s; measured 30.
- **vLLM and SGLang are within ~5% on L4 at every load.** vLLM is slightly ahead on throughput at 64 users (792 vs 749 tok/s); SGLang keeps TPOT lower under long-prompt load (71 vs 83 ms at 16 users).
- **llama.cpp falls 25–33% behind at 16 users** and its TTFT grows fastest: no continuous batching of prefill with decode on the same scale.
- **An L4 serving this 4B model in bf16 meets the GPU SLO for only ~4 chat users (or 1 RAG user).** The SLO's 50 ms TPOT leaves little room above the 33 ms single-user floor, and TTFT p95 breaks 500 ms first.

### 2. E1 on H100: vLLM vs SGLang

| Workload | Users | Out tok/s vLLM / SGLang | TTFT p95 (ms) | TPOT p50 (ms) | Goodput req/s |
|---|---|---|---|---|---|
| chat | 1 | 232 / 244 | 48 / 21 | 4.2 / 4.0 | 0.90 / 0.95 |
| chat | 16 | 2,947 / 2,994 | 171 / 147 | 4.8 / 4.8 | 11.5 / 11.7 |
| chat | 64 | 7,271 / 7,345 | 582 / 514 | 7.2 / 7.4 | 24.0 / **27.1** |
| chat | 128 | 9,508 / 9,614 | 1,066 / 926 | 10.7 / 11.1 | 10.6 / 14.6 |
| rag | 16 | 1,643 / 1,630 | 510 / 489 | 7.0 / 7.4 | 11.8 / **12.7** |
| rag | 128 | 2,698 / 2,656 | 3,619 / 3,614 | 38 / 33 | 1.4 / 2.0 |

**Findings**
- **On a real H100 the two engines tie on throughput (within ~2%).** SGLang has lower TTFT and so slightly higher goodput at 64–128 users.
- **Single-user decode reaches only ~57% of the H100's bandwidth ceiling** (3,350 GB/s ÷ 8.05 GB = 416 tok/s; measured ~238). One sequence of a 4B model cannot keep a large GPU busy; batching is where the H100 pays off.
- **Cost per useful answer (goodput per dollar, chat):** L4 0.43 req/s ÷ $0.80/h = **~1,900 SLO-meeting requests per $**; H100 SGLang 27.1 req/s ÷ $3.95/h = **~24,700 per $** (vLLM ~21,800). The H100 costs 4.9× more per hour and is **~11–13× cheaper per SLO-meeting request** for this model.

**Correction (caught by the manifest):** the first vLLM "H100" run (`e1-h100-vllm/20261124T081904Z`) actually ran on an **H200** — Modal upgraded the request, billing the H100 rate. On that run vLLM looked 15–22% faster than SGLang. Re-run on an exact H100 (`gpu: "H100!"`), the gap disappears. The H200 run is kept as an H200 data point (`GPU_NOTE.md`): vLLM on H200 vs H100 is **+20–23% throughput** (memory bandwidth 4.8 vs 3.35 TB/s = 1.43×). `bench_app.py` now flags any run whose real GPU differs from the request.

### 3. E8: vLLM memory knobs (L4, chat-balanced, 64 users, 1 repeat)

| Knob (others default: util 0.9, max_len 4096) | KV cache (GiB) | KV tokens | Out tok/s | TTFT p95 (s) | TPOT p50 (ms) |
|---|---|---|---|---|---|
| `gpu_memory_utilization` 0.5 | 2.00 | 14,544 | 409 | 28.3 | 50 |
| 0.7 | 6.99 | 50,864 | 794 | 4.8 | 75 |
| 0.9 | 11.40 | 83,024 | 794 | 4.8 | 75 |
| `max_model_len` 2,048 | 10.83 | 78,864 | 794 | 4.8 | 75 |
| `max_model_len` 16,384 | 10.83 | 78,864 | 794 | 4.9 | 75 |
| `max_num_seqs` 8 | 11.34 | 82,544 | 204 | 71.0 | 38 |
| `max_num_seqs` 32 | 11.28 | 82,112 | 558 | 16.8 | 55 |
| `max_num_seqs` 128 | 11.05 | 80,432 | 796 | 4.8 | 75 |

**Findings**
- **Too little KV cache halves throughput:** at util 0.5 only ~19 of 64 requests (768 tokens each) fit, the rest queue (TTFT p95 28 s).
- **Past "enough", more KV buys nothing at a given load:** 0.7 and 0.9 are identical at 64 users (64 × 768 = 49K tokens fits in both).
- **`max_model_len` does not change memory or speed** with paged attention; it only changes how many *maximum-length* requests could fit (vLLM's reported max concurrency: 38.5× at 2K vs 4.8× at 16K).
- **`max_num_seqs` trades per-user speed for total throughput:** capping at 8 gives each user 38 ms/token but cuts throughput 4× and makes others wait 71 s.
- **vLLM's KV size varies ~5% between identical configs** (10.83 GiB in E1 vs 11.40 GiB in E8 at util 0.9; smaller `max_num_seqs` also frees a little activation memory). Treat KV capacity as a ±5% quantity.

### 4. memcalc vs engines (S5: weights ±10%, max concurrency ±20%)

| Engine / GPU / util | memcalc KV (GiB) | Engine-reported KV (GiB) | Error | Max seqs @4,096: memcalc / engine |
|---|---|---|---|---|
| vLLM / L4 / 0.9 | 11.25 | 10.83 – 11.40 | −1% to +4% | 19 / 19.3 – 20.3 |
| vLLM / L4 / 0.7 | 6.75 | 6.99 | −3% | 12 / 12.4 |
| vLLM / L4 / 0.5 | 2.25 | 2.00 | +13% | 4 / 3.6 |
| vLLM / H100 / 0.9 | 62.69 | 61.06 | +3% | — |
| vLLM / H200 / 0.9 | 117.01 | 115.62 | +1% | — |
| SGLang / L4 / 0.85 | 10.12 | 10.83 | −7% | — |
| SGLang / H100 / 0.85 | 58.71 | 59.01 | −1% | — |

Weights: memcalc 7.49 GiB vs vLLM "Model loading took 7.64 GiB" (−2%). KV bytes per token (147,456 = 36 layers × 8 KV heads × 128 × 2 × 2 B) reproduce every engine's token count exactly from its GiB. **Both S5 targets are met.** The largest error (+13% at util 0.5) comes from memcalc's fixed 1.5 GiB reserve: at low utilisation the real activation/CUDA-graph reserve is a larger share of a small budget.

### 5. Cross-check against `vllm bench serve` (same server config, L4)

| Users | TTFT p95 (ms) servebench / vllm bench | TPOT p50 (ms) | Out tok/s |
|---|---|---|---|
| 1 | 106 / 114 | 33.3 / 33.5 | 30.0 / 29.6 |
| 16 | 1,301 / 1,323 | 41.9 / 42.0 | 353 / 359 |

**The harness agrees with the vendor's benchmark within 1–7%** (`results/m2-crosscheck/20261124T091327Z`). Differences are within run-to-run noise (vllm bench uses random token IDs on `/v1/completions`; servebench uses tokenizer-verified text on chat completions).

---

## M3: Quantization artifacts, quality (E3) and speed (E2), measured

**Base model:** Qwen3-4B-Instruct-2507 @ `cdbee75f`. **Quantizer:** llm-compressor 0.15.0 / compressed-tensors 0.20.1 in its own Modal image (it pins compressed-tensors 0.20.1; vLLM 0.32.0 pins 0.17.0). **GGUF:** llama.cpp `full-cuda-b12110`. Calibration: 512 samples × 2,048 tokens, generic = UltraChat 200k (`8049631c`), BFSI = eCFR Titles 12 + 17 (`Azzindani/US_Regulation_ECFR_20260101` @ `04148fee`), split by CFR part so the BFSI held-out set never overlaps calibration (D-026).
**Quality (vLLM 0.32.0 + lm-eval 0.4.13, same container):** ARC-Challenge (1,172, acc_norm), HellaSwag (first 2,000, acc_norm), MMLU-Pro leaderboard variant (first 1,000, logprob MC), GSM8K (1,319, 5-shot, strict + flexible extract), IFEval (541, prompt-level strict), WikiText word perplexity, BFSI held-out perplexity (200 sections). Every variant sees the same items; ± = 95% CI from lm-eval's standard error.
**Runs:** `results/quantize/*/cortex_manifest.json`, `results/quality-gpu/<variant>/`, `results/quality-gguf-gpu/`, `results/e2-{l4,h100}-*`.

### 1. Artifacts

| Variant | Method | Size | Time (GPU) |
|---|---|---|---|
| bf16 | original | 7.49 GiB | — |
| FP8-dynamic | FP8 E4M3 W per-channel, A per-token dynamic (data-free) | 4.12 GiB | 18 s (L4) |
| W8A8-INT8 | SmoothQuant 0.8 + GPTQ int8 | 4.12 GiB | 22 min (L4) |
| W4A16 GPTQ g128 | GPTQ int4, group 128, generic calib | 2.48 GiB | 16 min (L4) |
| W4A16 GPTQ g128 BFSI | same, BFSI calibration | 2.48 GiB | 12 min (L4) |
| W4A16 AWQ | AWQ scales + RTN int4 asym g128 | 2.49 GiB | 19 min (L40S; OOM on L4) |
| Qwen official FP8 | block-128 FP8 (Qwen) | 5.19 GB | — |
| GGUF Q8_0 / Q5_K_M / Q4_K_M (± imatrix generic / BFSI) | llama.cpp | 3.99 / 2.69 / 2.33 GiB | minutes (CPU) |

### 2. Quality vs bf16 (vLLM formats)

| Variant | ARC | HellaSwag | MMLU-Pro | GSM8K strict / flex | IFEval | WikiText PPL | BFSI PPL |
|---|---|---|---|---|---|---|---|
| bf16 | 58.3 ±2.8 | 58.5 ±2.2 | 34.5 ±2.9 | 72.8 / 78.8 | 83.4 ±3.1 | 14.27 | 5.97 |
| FP8-dynamic (ours) | +0.4 | −0.3 | −0.4 | +1.3 / +0.8 | −2.0 | +1.3% | +0.2% |
| FP8 official (Qwen) | +0.8 | −0.1 | −0.8 | −0.7 / +0.8 | +0.4 | +0.1% | +0.2% |
| W8A8-INT8 | +0.7 | +0.2 | 0.0 | −1.7 / +0.5 | −0.9 | +0.5% | +0.2% |
| W4A16 AWQ | −1.6 | −0.4 | −1.9 | +3.2 / +1.7 | −0.7 | +4.7% | +5.6% |
| W4A16 GPTQ (generic) | +0.1 | −1.6 | −2.3 | +0.7 / +0.8 | −0.9 | +5.5% | +7.3% |
| W4A16 GPTQ (BFSI calib) | +0.4 | −0.9 | −2.8 | **−38.4** / −3.2 | −3.9 | +8.3% | **+3.0%** |

**Findings**
- **8-bit (FP8 or INT8) is quality-neutral:** every task within its CI, perplexity +0.1–1.3%. Our 18-second data-free FP8 matches Qwen's official FP8.
- **4-bit costs a measurable but small amount:** perplexity +4.7–5.5%, MMLU-Pro −1.9 to −2.3 (inside the CI), reasoning and instruction tasks unchanged within noise. AWQ ≥ GPTQ on most metrics, all inside the CIs.
- **Domain-only calibration is a trade-off, not a free win.** BFSI calibration halves the in-domain perplexity damage (+3.0% vs +7.3%) but the model stops following the few-shot answer format: only 45% of GSM8K answers end in `#### N` (80% with generic calibration), so strict GSM8K drops 38 points while arithmetic itself (flexible extract) drops 3.2. IFEval −3.9 agrees. Recommendation: calibrate on a domain + general-chat mix.

### 3. GGUF quality (llama.cpp; KL divergence vs the bf16 GGUF on 100 WikiText-2 chunks, PPL on all chunks)

| Variant | Size | Mean KLD | Same top-1 | WikiText PPL | BFSI PPL |
|---|---|---|---|---|---|
| bf16 | 7.50 GiB | — | — | 10.338 | 5.444 |
| Q8_0 | 3.99 GiB | 0.0012 | 98.2% | +0.1% | +0.2% |
| Q5_K_M | 2.69 GiB | 0.0115 | 94.8% | +1.1% | +0.7% |
| Q4_K_M | 2.33 GiB | 0.0366 | 91.7% | +2.0% | +2.9% |
| Q4_K_M + imatrix (generic) | 2.33 GiB | 0.0250 | 92.4% | +2.9% | +2.4% |
| Q4_K_M + imatrix (BFSI) | 2.33 GiB | 0.0255 | 92.5% | +1.8% | +2.0% |

- **Each step down in bits roughly triples KL divergence** (8 → 5 → 4 bit); **an importance matrix cuts 4-bit KLD by ~31%** at the same size. Generic and BFSI imatrices are equivalent on KLD; BFSI is slightly better on BFSI text. PPL and KLD disagree on the generic imatrix (PPL slightly worse): KLD against the parent is the more reliable damage measure.

### 4. E2 speed: what the bytes buy (vLLM, chat-balanced 512/256)

| Format | L4 TPOT @1 (ms) | L4 goodput @4 (req/s) | L4 tok/s @64 | H100 TPOT @1 (ms) | H100 goodput @64 (req/s) | H100 tok/s @128 |
|---|---|---|---|---|---|---|
| bf16 | 33.3 | 0.43 | 792 | 4.15 | 27.2 | 10,079 |
| FP8-dynamic | 21.1 | 0.67 | 996 | 3.12 | 35.3 | 11,710 |
| FP8 official | 19.1 | 0.71 | 880 | 3.20 | 35.7 | 11,796 |
| W8A8-INT8 | 20.8 | 0.68 | 994 | 3.30 | 35.1 | 11,673 |
| W4A16 GPTQ | **12.2** | **1.07** | **1,045** | 3.66 | 25.1 | 9,619 |
| W4A16 AWQ | 12.3 | 1.06 | 1,038 | — | — | — |

llama.cpp on L4 (1 user, TPOT ms): bf16 33.2 · Q8_0 19.8 · Q5_K_M 14.3 · Q4_K_M 12.7 (imatrix: identical speed). At 16 users: 247 / 312 / 349 / 365 tok/s.

**Findings**
- **The best format depends on the GPU.** On the L4 (300 GB/s) decode stays memory-bound even at 64 users, so 4-bit wins everywhere: 2.7× faster per token at 1 user, 2.5× goodput at 4 users, +32% throughput at 64. On the H100 (3.35 TB/s) a busy server becomes compute-bound: **W4A16 is faster at 1 user but slower than bf16 at 64–128 users** (weight-only int4 must dequantize to bf16 before every matmul), while **FP8/INT8 (true 8-bit compute) give +30% goodput** at 64 users.
- **Qwen's official block-FP8 has worse TTFT on the L4** (1.55 s vs 0.99 s p95 at 16 users): sm_89 has no CUTLASS block-FP8 path (research M3 §6). On the H100 it equals our FP8.
- The engine-reported KV capacity grows as weights shrink (H100: 444K tokens bf16 → 469K FP8 → 481K W4A16).

### 5. Gemma 4 E2B perplexity anomaly from M1: resolved (llama.cpp, WikiText-2, ctx 512)

| Model | PPL, M1 setting (20 chunks) | PPL, all chunks | KLD vs own bf16 parent | Same top-1 |
|---|---|---|---|---|
| Google QAT checkpoint, bf16 (`…-qat-q4_0-unquantized`) | 61.7 | 54.0 | — | — |
| Google QAT Q4_0 GGUF | 65.5 | 58.2 | 0.050 | 89.4% |
| Original IT, bf16 | 213.7 | 185.4 | — | — |
| unsloth IT Q8_0 | 213.4 | 185.4 | 0.0065 | 96.8% |

**M1's "QAT 65 vs Q8_0 212" was never a quantization effect:** the QAT checkpoint is a different model whose raw-text perplexity is ~3.5× lower *before* quantization. Quantization damage measured against each model's own parent is normal (Q8_0 KLD 0.0065; QAT Q4_0 0.050). Lesson: compare quantization damage within one parent (KLD), never raw PPL across parents.

### 6. QAT vs PTQ: Gemma 4 E4B at the same W4A16 g32 symmetric scheme (vLLM, L40S)

| Variant | GSM8K strict / flex | IFEval | ARC, reasoning-first (n = 500, Wilson 95%) | Size |
|---|---|---|---|---|
| bf16 (original IT) | 71.4 / 76.5 | 85.2 ±3.0 | 93.6 (91.1–95.4) | — |
| Google QAT (`…-qat-w4a16-ct`) | −1.7 / −1.2 | −3.0 | 93.2 (90.6–95.1) | 11.5 GB |
| Our GPTQ PTQ (generic calib, basic pipeline) | +0.1 / −0.2 | −2.0 | 92.6 (90.0–94.6) | 10.75 GiB |
| Our RTN (data-free) | **−40.4 / −13.6** | **−6.8** | 90.6 (87.7–92.9) | 9.50 GiB |

**Findings**
- **Careful PTQ (GPTQ) matched both the original and Google's QAT checkpoint on every generative task** (all within CI). **Naive RTN** at the same bit width lost 13.6 points of GSM8K (flexible) and 6.8 of IFEval: *how* you quantize matters more than whether the model was trained for it, at this size and on these tasks.
- **QAT's large advantage shows up only in raw-text perplexity** (WikiText word PPL 86 vs 252 for the original; BFSI 52 vs 127), the same "different model" effect as in §5. It did not translate into better chat-task scores.
- **Loglikelihood benchmarks are not meaningful for Gemma 4 IT:** ARC/HellaSwag/MMLU-Pro sit at chance even for bf16 (25.2 / 37.4 / 24.8), with or without `<bos>` (`results/quality-gpu-invalid/README.md`). Gemma is compared on generative tasks only. lm-eval's `arc_challenge_chat` scores 0 for every Gemma variant (it expects a bare letter; Gemma reasons first), so reasoning-first ARC uses our own scorer (`quality/arc.py`, `cot=True`).
- Size difference explained by a header audit (`deploy/modal/artifact_audit.py`): both artifacts hold identical quantized text layers (1.84 GiB int4 + scales); the GPTQ artifact additionally stores `lm_head` as a separate bf16 tensor (1.25 GiB) where RTN kept it tied to `embed_tokens`. Same weights, different storage; 10.72 GiB (11.5 GB) equals Google's QAT checkpoint size.

---

## M4: Feature experiments (E4–E7, E9, E10, E12, E15), measured

All on Modal with the pinned vLLM 0.32.0 / SGLang 0.5.22 images, Qwen3-4B-Instruct-2507 bf16 unless stated, L4 unless stated, closed-loop concurrency, medians of 1–2 repeats (stated per config in `configs/gpu/e*.yaml`). Runs: `results/e{4,5,6,7,9,10,12,15}-*`.

### E4. Prefix caching (2,048-token prompts, 128 out)

| Workload | Users | TTFT p50 (s): vLLM APC off / on · SGLang radix off / on | Out tok/s: vLLM off / on · SGLang off / on |
|---|---|---|---|
| shared-prefix (1,536 of 2,048 tokens shared) | 1 | 0.37 / **0.12** · 0.36 / **0.12** | 27 / 29 · 27 / 29 |
| shared-prefix | 16 | 1.63 / **1.03** · 3.34 / **1.15** | 165 / **285** · 168 / **296** |
| rag-long-prefill (unique prompts) | 16 | 1.43 / 1.43 · 3.47 / 3.51 | 166 / 165 · 165 / 164 |

- **With 75% of the prompt shared, caching cuts TTFT 3× at 1 user and raises throughput 73–76% at 16 users** (TTFT p95 5.3 s → 1.5 s). vLLM's measured cache hit rate was 72–75%, matching the 75% shared share.
- **No cost on unique prompts** (within 1%). vLLM APC and SGLang RadixAttention are equivalent on this workload.

### E5. Chunked prefill (vLLM)

| Workload | Users | ITL p99 (ms): chunked off / on | TPOT p50 (ms) | Out tok/s |
|---|---|---|---|---|
| rag-long-prefill (2,048 in) | 4 | **365 / 47** | 42.8 / 41.0 | 80 / 81 |
| rag-long-prefill | 16 | 397 / 370 | 90 / 86 | 159 / 163 |
| chat-balanced (512 in) | 16 | 44 / 49 | 42 / 42 | 347 / 349 |

- **Chunked prefill cuts the inter-token tail 8× when long prompts arrive while others decode** (4 users): without it each new 2,048-token prefill stalls every running stream for one long step.
- At 16 long-prompt users both settings stall (~370–400 ms): prefill work fills every step, so the next knob is the chunk size (`max_num_batched_tokens`). vLLM warns that disabling chunked prefill is not officially supported for this model.

### E6. Speculative decoding (real UltraChat prompts, natural stop, ≤256 out)

| Config | TPOT p50 @1 (ms) | Out tok/s @1 / @4 / @16 | Acceptance | TTFT p95 @1 (s) |
|---|---|---|---|---|
| Baseline | 33 | 30 / 101 / **397** | — | 0.11 |
| Draft model Qwen3-0.6B, k = 4 | **26** | **38 / 119** / 374 | 30–33% | 0.22 |
| n-gram prompt lookup, k = 4 | 34 | 30 / 96 / 327 | 2% | 0.10 |

- **A small draft model speeds up a single user 1.27× on the GPU** (the opposite of M1's CPU result, where verifying k tokens was not cheap) **and the gain is gone by 16 users** (−6% throughput): with a busy batch, verification is no longer free.
- **n-gram lookup is useless on open-ended chat** (2% acceptance; answers rarely copy the prompt) and costs 18% throughput at 16 users. It is a tool for edit/extract/RAG-quote workloads, not chat.
- The draft model doubles TTFT at 1 user (it prefills too).

### E7. FP8 KV cache (long-context: 7,680 in / 256 out, max_model_len 8,192)

| | KV tokens | Max seqs @8K | Out tok/s @16 / @32 | TTFT p95 @16 (s) | TPOT p50 @16 (ms) |
|---|---|---|---|---|---|
| bf16 KV | 78,864 | 9.6 | 69 / 71 | 41.2 | 120 |
| FP8 KV | **151,184** | **18.5** | **101 / 104** | **21.0** | 142 |

- **FP8 KV doubles capacity (1.92× tokens) and gives +46% throughput with half the TTFT tail** once long contexts are KV-bound; each stream decodes a little slower because twice as many share the GPU.
- **Quality cost of FP8 KV: none measurable** (same suite as M3, `--kv-cache-dtype fp8`): ARC +0.4, HellaSwag −0.6, MMLU-Pro −1.8 (CI ±2.9), GSM8K +1.1, IFEval −0.6, WikiText PPL +0.1%, BFSI PPL +0.2%. These tasks use short contexts; long-context retrieval quality under FP8 KV was not measured.

### E9. MoE: Qwen3-30B-A3B-Instruct-2507 vs Qwen3-4B dense (bf16, exact H100)

| | 4B dense | 30B-A3B MoE (3.3B active) |
|---|---|---|
| Weights | 7.6 GiB | 56.9 GiB |
| KV tokens | 444,592 | 124,128 |
| TPOT p50 @1 / @16 / @64 (ms) | 4.15 / 4.77 / 7.15 | **4.68** / 10.36 / 16.47 |
| Out tok/s @64 / @128 | 7,584 / 10,079 | 3,607 / 5,398 |
| ARC / HellaSwag / MMLU-Pro | 58.3 / 58.5 / 34.5 | **62.5 / 67.8 / 46.9** |
| GSM8K strict / flex · IFEval | 72.8 / 78.8 · 83.4 | **83.9 / 86.1** · 82.6 |
| WikiText word PPL | 14.27 | 9.54 |

- **At 1 user the MoE decodes almost as fast as the 4B dense model** (4.7 vs 4.2 ms): each token reads only the active experts.
- **Under batching the advantage disappears:** different requests route to different experts, so the batch reads most of the 30B weights; TPOT doubles by 16 users and throughput is half the dense model's at 64–128.
- **Quality is much higher** (MMLU-Pro +12.4, GSM8K +11, HellaSwag +9.3) for 7.5× the weight memory. MoE buys a bigger model's quality at a small model's single-stream latency, not at its batch throughput or memory.

### E10. Structured output (JSON schema, ≤400 out)

| | JSON valid | Truncated | TPOT p50 @1 / @16 (ms) | TTFT p95 @16 (s) |
|---|---|---|---|---|
| Instruction only | 100% | 0% | 33.5 / 44.6 | 1.36 |
| `response_format: json_schema` | 100% | 0% | 33.5 / 44.8 | 1.37 |

- **Constrained decoding costs nothing measurable here** (≤0.5% TPOT), and **this model already produced valid JSON 100% of the time when asked** for a simple 4-key schema. The constraint is free insurance. A first run with a 200-token cap (superseded) showed the real failure mode: 12.5% of answers cut off by `max_tokens`, invalid in both modes.

### E15. Architecture study: Qwen3-4B (dense) vs Qwen3.5-4B (hybrid Gated DeltaNet) vs Gemma 4 E4B (SWA + KV sharing + PLE), L4

| | Qwen3-4B | Qwen3.5-4B | Gemma 4 E4B |
|---|---|---|---|
| Weights on GPU (vLLM) | 7.64 GiB | 7.94 GiB | 14.5 GiB (PLE + vision/audio towers) |
| KV pool (vLLM, util 0.9) | 10.81 GiB | 9.86 GiB | 4.16 GiB |
| Memory per 8K sequence: memcalc / vLLM | 1,152 / 1,152 MiB | 305 / 363 MiB | 148 / 309 MiB |
| Memory per 32K sequence: memcalc / vLLM | 4,608 / 4,612 MiB | 1,073 / 1,140 MiB | 532 / 692 MiB |
| Max concurrent seqs @8K / @32K (vLLM) | 9.6 / 2.4 | **27.8 / 8.9** | 13.8 / 6.2 |
| TPOT p50 @1, chat (ms) | **33.5** | 35.0 | 38.8 |
| Out tok/s, chat @32 (vLLM / SGLang) | **528 / 538** | 484 / 313 | 512 / 502 |
| Out tok/s, long-context @32 (vLLM / SGLang) | 68 / 69 | **109 / 111** | **116** / 57 |
| TTFT p95, long-context @32 (vLLM) | 100 s | **50 s** | 56 s |

- **The new architectures cut memory per sequence 3.2–3.7× at 8K and 4.0–6.7× at 32K, and on long contexts that becomes 1.6–1.7× throughput and half the TTFT tail.** For short chats the dense model is still fastest (it streams the fewest bytes per token).
- **How much of the theoretical saving vLLM realises:** Qwen3.5 84% at 8K (363 vs 305 MiB per sequence) and 94% at 32K; Gemma only ~48% at 8K and 77% at 32K. Gemma's sliding-window layers carry a fixed per-sequence overhead beyond the window (the same pattern llama.cpp showed in M1). memcalc is exact for the dense model.
- **Gemma's per-sequence saving is partly eaten by its 14.5 GiB of weights on a 24 GB card:** its KV pool is only 4.2 GiB, so it fits fewer 8K sequences than Qwen3.5 despite smaller KV per token.
- **Quality, Qwen3.5-4B vs Qwen3-4B (same suite, thinking off):** HellaSwag +5.8, MMLU-Pro +3.4, GSM8K +7.7 strict / +3.2 flexible, WikiText word PPL 12.2 vs 14.3; ARC −3.9, IFEval −2.3; reasoning-first ARC 93.8 vs 94.0. **The hybrid model gives equal-or-better quality with 3–4× less KV memory per sequence.** (Gemma 4 E4B is not on this scale: its loglikelihood benchmarks are unusable, §M3.6; on generative tasks it scores GSM8K 71.4/76.5, IFEval 85.2, reasoning-first ARC 93.6.)
- **Engine support differs for the new designs:** SGLang matches vLLM on Qwen3.5 long contexts but not on 32-user chat (313 vs 484 tok/s, TTFT p95 15 s), and on Gemma long contexts it reaches only half of vLLM's throughput. Not investigated further.

### E12. Tensor parallel: TP=2 on 2 x H100 vs TP=1 on 1 x H100 (Qwen3-4B bf16, chat-balanced)

| Users | TPOT p50 (ms) TP1 / TP2 | Out tok/s TP1 / TP2 | Goodput req/s TP1 / TP2 |
|---|---|---|---|
| 1 | 4.15 / **3.04** | 237 / 305 | 0.92 / 1.19 |
| 16 | 4.77 / 3.69 | 3,034 / 3,883 | 11.9 / 15.2 |
| 64 | 7.15 / 5.13 | 7,584 / 10,244 | 27.2 / 40.0 |
| 128 | 10.67 / 7.05 | 10,079 / 14,433 | 20.5 / 31.4 |

- **TP=2 cuts per-token latency 1.37× and adds 43–47% capacity, for 2× the price:** goodput per dollar at 64 users falls from 6.9 to 5.1 req/s per $/h (−27%). Each layer ends in an all-reduce across the two GPUs; for a 4B model that fits on one GPU, the communication is a large share of each step.
- **For throughput, two independent replicas (data parallel) are the cheaper way to use a second GPU** for a model this size (not measured; follows from the TP=1 numbers). TP pays off when latency is the constraint or the model does not fit on one GPU (E9's 57 GiB MoE would need TP on an 80 GB card at long contexts).
- 2 x L4 was requested first but Modal never scheduled it (1 h 45 min in queue, not billed), so the pair is H100s.

---

## M5: production-style serving on local k3d (CPU), measured

**Setup:** single-node k3d (k3s v1.35.10) on the M1 VM (2 cores / 4 threads, 15 GiB). Helm chart `deploy/helm/cortex-serve`.
LiteLLM v1.108.16 gateway, KEDA 2.21.1, kube-prometheus-stack 93.0.0, llama.cpp server image from M1.
**Primary pool:** Qwen3-0.6B Q4_K_M, **1 thread × 2 slots per replica**, 1–3 replicas, scaled by KEDA on (running + queued) / slots.
**Canary:** Qwen3-0.6B Q8_0 (10%). **Fallback (D-021):** Qwen3-1.7B Q4_K_M, 2 threads.
**Workload:** 64 input / 32 output tokens through the gateway (client-side timing, `gatewaybench`), every request logged.
**Runs:** `results/m5-autoscale/{20261124T131550Z,20261124T135406Z}`, `results/m5-rollout/20261124T133518Z`,
`results/m5-fallback/20261124T134537Z`, `results/m5-netpol/20261124T135251Z`, `results/m5-canary/20261124T140843Z`.
Details: `docs/milestones/M5_REPORT.md`.

### 1. Reliability

| Run | Requests | Failed | Gateway retries | What happened during the run |
|---|---|---|---|---|
| Autoscaling ramp (final query) | 133 | 0 | 1 | pool 1 → 2 → 3 → 1 replicas |
| Autoscaling ramp (first query) | 155 | 0 | 0 | pool 1 → 2 → 3 → 1 (over-scaled at 4 users) |
| Rolling update, 4 users | 127 | 0 | 0 | two full rollouts of a 3-replica pool (`maxUnavailable 0`) |
| Fallback demo, 2 users | 46 | 0 | 0 | primary at 0 replicas for ~150 s; 16 requests served by the fallback model |
| Canary window, 4 users | 169 | 0 | 0 | 90/10 split; 13 requests (7.7%) went to the canary |
| **Total** | **630** | **0** | **1** | |

### 2. Autoscaling (final ramp)

| Users | Replicas | TTFT p50 / p95 (s) | Pool decode tok/s |
|---|---|---|---|
| 1 | 1 | 3.33 / 3.45 | 5.0 |
| 4 | 1 → 2 | 8.31 / 15.92 | 7.3 |
| 6 | 2 → 3 | 12.19 / 28.36 | 8.9 |
| 2 | 3 → 1 | 3.43 / 6.48 | 7.0 |

**Findings**
- **Replicas tracked demand / slots** (4 users → 2, 6 → 3). Load step → new replica Ready in ≈ 20–25 s; pod create → Ready from the weight cache ≈ 6 s. First scale-down ≈ 65 s after the load dropped (60 s stabilisation).
- **Scaling did not restore the CPU SLO (TTFT ≤ 3 s) beyond 1 user.** Three single-thread replicas on 2 physical cores gave 1.8× the decode throughput of one, and busy replicas were CPU-throttled in ~19% of periods.
- **Requests were not balanced across replicas.** The gateway's keep-alive connections plus per-connection ClusterIP balancing left per-pod demand at 2.84 / 1.56 / 0.94 in the 6-user stage (first ramp). Queues persisted although the slot count matched the load. Not yet measured: the opt-in per-request connection setting in the chart.
- **Two scaling-query bugs, measured and fixed** (ADR 0007): summing separate `max_over_time` terms, and the peak of the summed series, both over-read 4 users as 5 and held 3 replicas instead of 2.

### 3. Operations findings

| Finding | Evidence | Fix in the chart |
|---|---|---|
| llama.cpp's default host prompt cache (8,192 MiB) OOM-killed a primary pod with a 1.5 GiB limit after ~100 min of load (RSS 484 → 1,507 MiB) | cAdvisor series, pod `OOMKilled` | `--cache-ram 256` |
| A terminating pod kept serving for its whole 120 s grace period because pooled gateway connections bypassed endpoint removal | fallback demo outcome log | opt-in `gateway.backendConnectionClose` (unmeasured) |
| NetworkPolicies + backend key: 10 / 10 access checks behave as designed | `m5-netpol/*/result.md` | n/a |
| Canary gate returned HOLD (13 canary requests < 20); probe accuracy 12/20 for both versions | `m5-canary/*/decision.json` | longer window needed on CPU |

### M5 addendum: GPU fallback to a Modal endpoint (2026-11-25)

The gateway's prepared Modal slot was enabled with our M3 FP8 Qwen3-4B served by vLLM 0.32.0 on a Modal L4 (`deploy/modal/vllm_endpoint.py`: API key in a Modal secret, max 1 container, scale-to-zero after 2 min; unauthenticated requests get 401). Load: 2 users, 64 in / 32 out, 410 s; the whole local `cortex-default` pool (primary and canary) taken to 0 for 200 s, then restored (`scripts/m5_modal_fallback.sh`, `results/m5-modal-fallback/20261125T073502Z`).

| Served by | Requests | TTFT p50 / p95 (s) | End-to-end p50 / p95 (s) |
|---|---|---|---|
| Modal L4 GPU (fallback) | 214 | **0.27** / 5.59 | **0.97** / 6.30 |
| Local CPU pool (before/after the outage) | 38 | 5.96 / 6.36 | 11.2 / 11.3 |

- **250 of 252 requests succeeded; 214 were served by the GPU fallback** while the local pool was down. The 2 failures were streams already in progress on local pods when they were scaled to 0 ("response payload is not completed"), the same in-flight cut-off seen in the M5 CPU fallback demo.
- The GPU p95 TTFT (5.6 s) is the first requests of the outage, which waited for the failed local attempt and the gateway cooldown before falling back.
- **A first attempt served 0 GPU requests** (`…-gpu-unauthorized/INVALID.md`): the load-test virtual key did not list `cortex-gpu`, so LiteLLM skipped that fallback ("caller is not authorized") and went straight to the second CPU model; and the canary deployment kept serving `cortex-default`. Per-key model allow-lists apply to fallbacks too.
