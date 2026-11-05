# G-01 Research · Model selection (small models first)

> **Date:** 2026-11-04 · **Your brief:** prefer Gemma or other small models, because they're quicker and need less compute. Suggest others too.
> **Method:** architecture facts read directly from each model's Hugging Face `config.json` and API metadata (`evidence/M0/02_candidate-model-configs.txt`), not from blogs. Engine support checked against vLLM's model registry and SGLang's model directory on their main branches. GGUF availability checked on the HF Hub. **Re-verify versions at M1/M2 build time.**

---

## 1. Key finding: small 2026 models use different memory architectures

The spec assumes a plain dense GQA transformer, where KV cache grows linearly with every token in every layer. The newer small families don't work that way:

| Family | Architecture (from config) | What it does to memory |
|---|---|---|
| **Qwen3** (Apr 2025) | Dense, all layers full attention, GQA (8 KV heads, head_dim 128) | Textbook case: the spec's formulas apply unchanged |
| **Qwen3.5** (early 2026) | **Hybrid:** 1 in 4 layers full attention (GQA, head_dim 256); 3 in 4 layers **Gated DeltaNet** (linear attention, fixed-size recurrent state). Ships as a multimodal (`ConditionalGeneration`) checkpoint. | KV grows only in the full-attention layers, plus a fixed per-sequence state |
| **Gemma 4 E2B/E4B** (Apr 2026, now **Apache-2.0**) | Mostly **sliding-window** layers (window 512) + some global layers (head_dim 512); **KV sharing** across layers (20 of 35 / 18 of 42 layers reuse earlier KV); **Per-Layer Embeddings** (large stored params, small "effective" compute); 262K vocab; multimodal | "E2B" stores **5.1B params** but computes like ~2B. Memory math needs per-layer-type modelling. |
| **Granite 4.0-H** | Hybrid Mamba-2 + attention (36 of 40 layers Mamba) + MoE | Mostly fixed-state; MoE on top |

**KV cache + state per sequence, theoretical, bf16 KV** (`evidence/M0/04_kv-cache-corrected.txt`; supersedes `03_…`):

| Model | Memory mechanism | Per sequence @ 8K | Per sequence @ 32K | vs Qwen3-4B @ 8K |
|---|---|---|---|---|
| Qwen3-4B (dense GQA) | 144 KiB/token × all 36 layers | 1,152 MiB | 4,608 MiB | 1× |
| Qwen3.5-4B (hybrid) | 32 KiB/token in 8 full layers + 49 MiB fixed DeltaNet state (**fp32**, per `mamba_ssm_dtype`) | 305 MiB | 1,073 MiB | ~3.8× less |
| Gemma 4 E4B | Only 24 of 42 layers own KV (last 18 reuse it); of those, 20 are sliding-window (capped at 512 tokens); 4 full layers with head_dim 512 | 148 MiB | 532 MiB | ~7.8× less |
| Gemma 4 E2B | 15 of 35 layers own KV; 12 sliding + 3 full | 54 MiB | 198 MiB | — |
| Qwen3-1.7B (dense GQA) | 112 KiB/token | 896 MiB | 3,584 MiB | — |
| Qwen3-30B-A3B (MoE, GQA) | 96 KiB/token | 768 MiB | 3,072 MiB | — |

*Corrections logged:* (1) the first estimate assumed a bf16 DeltaNet state, but the config says fp32, which doubles it. (2) The Gemma KV-sharing assumption (that the *last* N layers are the consumers) was verified in transformers' `configuration_gemma4.py` → `kv_sharing_modes`.

**Why this matters for P2:** in theory a ~4B hybrid or KV-sharing model fits **4–8× more concurrent 8K sequences** in the same GPU memory as a dense 4B. **The open question for E15 is how much of that the engines (vLLM, SGLang, llama.cpp) actually realize.** That depends on hybrid-cache allocators and sliding-window support, and it's a result very few people have measured.

**Also:** Gemma 4's "E" (effective) sizes come from **Per-Layer Embedding** tables: 2.82B params in E4B and 2.35B in E2B, which engines can keep off-GPU or memory-mapped. memcalc has to report "stored params" and "compute params" separately.

---

## 2. Candidates checked

| Model | Params (stored) | License | Gated | vLLM | SGLang | GGUF (llama.cpp) | Notes |
|---|---|---|---|---|---|---|---|
| Qwen3-0.6B | 0.75B | Apache-2.0 | No | ✅ | ✅ | ✅ official | Ideal **draft model** for spec decoding (same tokenizer as Qwen3-4B) |
| Qwen3-1.7B | 2.0B | Apache-2.0 | No | ✅ | ✅ | ✅ official | Main CPU model; P4 base family |
| **Qwen3-4B** | 4.0B | Apache-2.0 | No | ✅ | ✅ | ✅ official | Dense baseline; **8 GB bf16 → fits L4 with ~12 GB KV room** |
| Qwen3.5-0.8B / 2B / 4B / 9B | 0.87 / 2.3 / 4.7 / 9.7B | Apache-2.0 | No | ✅ | ✅ (+ MTP heads) | ✅ unsloth/bartowski | Hybrid DeltaNet; native MTP heads allow built-in speculative decoding |
| **Gemma 4 E2B / E4B** | 5.1 / 8.0B (eff. 2B / 4B) | **Apache-2.0** | No | ✅ | ✅ (+ MTP) | ✅ ggml-org + **official Google QAT Q4_0** | Google also ships **QAT W4A16** (compressed-tensors) for vLLM → enables a clean **QAT vs PTQ** experiment |
| Gemma 4 26B-A4B | 25.8B total / ~4B active | Apache-2.0 | No | ✅ | ✅ | ✅ | MoE; needs H100 at bf16 (~52 GB) |
| Qwen3-30B-A3B | 30.5B total / ~3.3B active | Apache-2.0 | No | ✅ | ✅ | ✅ | MoE in the same family as Qwen3-4B: **similar active compute, ~7.5× the memory** |
| Granite-4.0-H-Tiny | 6.9B total / ~1B active | Apache-2.0 | No | ✅ | ✅ | ✅ | Cheapest MoE (fits L4 at bf16), but mixes MoE with Mamba-hybrid effects |
| SmolLM3-3B | 3.1B | Apache-2.0 | No | ✅ | — | ✅ | Fully open (data + recipe); more useful in P4 |
| Phi-4-mini | 3.8B | MIT | No | ✅ | ✅ | ✅ | Fine; adds little over Qwen3-4B here |
| ~~Llama-3.2-3B~~ | 3.2B | Llama license | **Yes** | | | | Rejected: gated + restrictive license |
| ~~LFM2-1.2B~~ | 1.2B | "other" (LFM license) | No | | | | Rejected: non-standard license for published artifacts |
| ~~Qwen3-8B~~ (spec default) | 8.2B | Apache-2.0 | No | | | | **Dropped per your brief**: 2× the compute of 4B for little extra learning |

---

## 3. Recommended lineup

> **Decided 2026-11-04 (D-014):** accepted as below, except the CPU tier is **Qwen3-0.6B, Qwen3-1.7B, Gemma 4 E2B** (Qwen3.5-2B and Qwen3-4B Q4_K_M dropped). Primary pinned to **Qwen3-4B-Instruct-2507**.

| Role | Model | Tier | Why |
|---|---|---|---|
| **Primary (E1–E8, E10, E14)** | **Qwen3-4B** (Instruct-2507 variant if present; verify) | L4 (+ short H100) | Textbook dense GQA, so the spec's math and every engine and quantization tool work unchanged. Half the cost of 8B per run. |
| Draft model (E6) | Qwen3-0.6B | L4 + CPU | Same tokenizer as the primary |
| CPU tier (E13) | Qwen3-0.6B, Qwen3-1.7B, **Gemma 4 E2B**, Qwen3.5-2B; Qwen3-4B Q4_K_M as the top point | Local CPU | All ≤ ~3 GB at 4-bit; quick on 4 threads |
| **New E15: architecture study** | **Qwen3-4B vs Qwen3.5-4B vs Gemma 4 E4B** | L4 | Three ~4B models with three memory designs. Measure KV/token, max concurrency at 8K/32K, throughput, quality. memcalc predicts all three. |
| QAT vs PTQ (added to E3) | Gemma 4 E4B: Google QAT Q4_0 / W4A16 vs our own PTQ (GGUF Q4_K_M, GPTQ/AWQ) | L4 + CPU | Spec §3.5 asks for QAT vs PTQ, and this is the cheapest clean way to measure it |
| MoE (E9) | **Qwen3-30B-A3B** | H100 short run (bf16) or L4 at 4-bit | Same family as the primary → a clean dense-vs-MoE comparison at similar active compute |

**Compute impact vs the spec's 8B plan:** primary runs need roughly half the GPU time and memory. Every CPU-tier model runs comfortably on this box, and the 8B GGUF conversion that didn't fit on local disk is no longer needed. E15 adds ~$8–12 on L4.

**Costs of this choice:**
- memcalc must model per-layer types (sliding window, KV sharing, linear-attention state). That makes memcalc stronger, but M0 gets a little larger.
- Hybrid models may have weaker quantization-tool support (llm-compressor recipes for DeltaNet layers). That gets checked in M3, and the quantization grid stays on Qwen3-4B (dense) so it doesn't depend on that support.

---

## 4. Sources
- HF model API + `config.json` for every model above (raw dump in evidence)
- vLLM `vllm/model_executor/models/registry.py` (main), SGLang `python/sglang/srt/models/` (main)
- [Gemma 4 model card](https://ai.google.dev/gemma/docs/core/model_card_4) · [Gemma 4 announcement](https://blog.google/innovation-and-ai/technology/developers-tools/gemma-4/)
- [Qwen3.5 small models overview](https://www.therundown.ai/tools/qwen3-5-small) · [BentoML SLM roundup 2026](https://www.bentoml.com/blog/the-best-open-source-small-language-models)
