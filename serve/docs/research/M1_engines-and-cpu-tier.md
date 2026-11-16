# M1 research: CPU-tier engines, artifacts and measurement design (2026-11-05)

## Engines (pinned)
| Engine | Image (digest) | Version |
|---|---|---|
| llama.cpp server | `ghcr.io/ggml-org/llama.cpp:server@sha256:1145e8de3111dc9358c6d8daffb755c4cf8a862d93391c26ba521b9a810bbe3e` | 0.9.0-dev, build 12110, commit `cd959bae9` |
| llama.cpp tools (bench, perplexity, quantize, imatrix) | `ghcr.io/ggml-org/llama.cpp:full@sha256:93573d882ea21d1b77712a33162934daaabcdf3a076dfff91a9ed3c2aed8cd41` | same build |
| Ollama | `ollama/ollama:0.40.14@sha256:ebaef4f4c5d93660e1f25f8ad085a112f51ccfb0e8458073e71793959dba91c9` | 0.40.14 |

**llama-server flags verified from `--help` inside the pinned image** (not assumed):
`-t`, `-np/--parallel`, `--kv-unified-per-slot N` (sizes the KV pool to `n_parallel × N`), `--ignore-eos` (server-wide), `--metrics`, `--jinja` (default on), `--chat-template-kwargs`, `-md/--spec-draft-model`, `--spec-draft-n-max` (default 3), `--spec-type`, `-ctk/-ctv`, `--swa-full`, `--cache-reuse`. **Changed vs older builds:** `--draft-max/--draft-min` are removed (use `--spec-draft-n-*`).

## Machine
AMD EPYC 7571 (Zen 1), 2 cores / 4 threads, AVX2+FMA (no AVX-512/AMX), 15 GiB. llama.cpp selects its `haswell` CPU backend.
- **Memory bandwidth (measured, `scripts/membw.c`):** read 10.2 / 17.0 / **20.3 GB/s** at 1 / 2 / 4 threads. *First attempt under-reported 1-thread read at 6.1 GB/s, because a non-`simd` FP reduction is latency-bound, not memory-bound.*
- **Threads:** llama-bench on Qwen3-0.6B Q4_K_M: 4 threads beat 2 (pp128 63 vs 42 tok/s, tg64 27 vs 17 tok/s), so `-t 4`.
- **This CPU is compute-bound even in decode:** 0.6B Q4_K_M decodes at ~27 tok/s ≈ 10 GB/s, about half the measured bandwidth. Concurrency 2 barely raises aggregate throughput (12.6 → 13.4 tok/s in the smoke run) while TPOT doubles. Prefill at ~63 tok/s ≈ 75 GFLOP/s, about the peak of 2 Zen-1 cores. Unlike a GPU, batching buys little here.

## Artifacts (all Apache-2.0; exact bytes from GGUF headers, `memcalc/gguf.py`)
| Artifact | Repo @ revision | Params | Weight MiB | BPW | Type mix |
|---|---|---|---|---|---|
| Qwen3-0.6B Q4_K_M | unsloth/Qwen3-0.6B-GGUF @ 50968a44 | 596.0M | 372.7 | 5.24 | Q4_K 205 + Q6_K 168 |
| Qwen3-0.6B Q8_0 | same | 596.0M | 604.1 | 8.50 | Q8_0 |
| Qwen3-1.7B Q4_K_M | unsloth/Qwen3-1.7B-GGUF @ d7f544ee | 1720.6M | 1050.4 | 5.12 | Q4_K 646 + Q6_K 404 |
| Qwen3-1.7B Q8_0 | same | 1720.6M | 1743.8 | 8.50 | Q8_0 |
| Gemma 4 E2B **QAT** Q4_0 | google/gemma-4-E2B-it-qat-q4_0-gguf @ 675cff42 | 4628.6M | 3179.3 | 5.76 | **Q6_K 2152 (PLE + embeddings)** + Q4_0 999 |
| Gemma 4 E2B PTQ Q4_0 | unsloth/gemma-4-E2B-it-GGUF @ 0314792d | 4647.5M | 2885.4 | 5.21 | **Q5_K 1540 (PLE)** + Q4_0 975 + Q4_K 216 + … |
| Gemma 4 E2B Q4_K_M | same | 4647.5M | 2947.7 | 5.32 | Q5_K 1540 + Q4_K 1070 + Q6_K 205 |
| Gemma 4 E2B Q8_0 | same | 4647.5M | 4799.4 | 8.66 | Q8_0 |

Findings:
- **llama.cpp confirms memcalc's Qwen3-0.6B count: 596.05M**, not HF's 752M (duplicate `lm_head` not converted).
- **Gemma "Q4_0" files are mostly Q5/Q6 by bytes:** the per-layer-embedding table (≈1.5–1.8 GiB) is kept at higher precision.
- **QAT vs PTQ is confounded:** Google's QAT file stores PLE + embeddings at Q6_K, unsloth's PTQ at Q5_K/Q4_K, while the decoder linears are Q4_0 in both. Google's file also omits **18.9M params**: exactly memcalc's `unused_shared_kv` for E2B (k/v weights of KV-shared layers). Any quality difference mixes training method and PLE precision.
- Average-BPW models of GGUF are off by up to −21% (Gemma QAT: 4.5 assumed vs 5.76 real). memcalc now reads GGUF headers (`--gguf`) for exact weight bytes.

## KV cache: memcalc vs llama.cpp (first CPU validation)
| Model, 4 slots × 2048 ctx | memcalc (theory) | llama.cpp allocates | Note |
|---|---|---|---|
| Qwen3-0.6B | 4 × 224 = 896 MiB | 896.00 MiB | exact |
| Gemma 4 E2B, global layers | 3 layers: 48 MiB | 48.00 MiB (2048 cells, 3 layers) | exact; **KV sharing fully exploited** (only 15 of 35 layers own KV) |
| Gemma 4 E2B, sliding layers | 12 layers × 512 window: 24 MiB | 48.00 MiB (**1024 cells**, 12 layers) | llama.cpp keeps window + one 512-token ubatch → 2× theory |

## Measurement design decisions
- **CPU SLO (G-04):** TTFT ≤ 3 s, TPOT ≤ 125 ms (≥ 8 tok/s). The spec's 500 ms / 50 ms GPU SLO is unreachable here.
- **Workload:** `chat-short` 128 in / 64 out at concurrency 1 and 4, 8 measured requests per point, 1 warmup, 3 repeats. Prefill-heavy shapes are covered by `llama-bench pp512` (a 512-token prompt costs ~8–25 s of TTFT on this CPU).
- **Prompts:** seeded English-word text with exact token counts in the model's own tokenizer. The instruction is placed *after* the unique text so no two prompts share a cacheable prefix.
- **Thinking disabled** (`enable_thinking: false` / Ollama `think: false`) and `--ignore-eos` for fixed output length. Ollama can't ignore EOS, so its output lengths are recorded per request.
- **Quality:** WikiText-2 perplexity (20 × 512 tokens, `llama-perplexity`; within-family comparison) + ARC-Challenge n=100 via the chat API (letter answer, greedy; **hedged answers naming several options count as wrong**), Wilson 95% CI.
