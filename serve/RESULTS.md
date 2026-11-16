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
