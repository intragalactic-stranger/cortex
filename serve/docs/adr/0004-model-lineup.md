# 0004 · Model lineup: small, permissive, and one architecture study

- **Status:** Accepted · **Date:** 2026-11-04 · **Related:** D-014, research `G-01_model_selection.md`

## Context
Owner's brief: prefer small models (Gemma or other SLMs) for speed and lower compute. In 2026 the small open families use different memory designs (dense GQA, hybrid linear attention, sliding window + KV sharing + PLE). Everything published must be license-clean.

## Options considered (primary model)
1. **Qwen3-4B-Instruct-2507**: dense GQA, Apache-2.0, supported by every engine and quantization tool, official FP8 to cross-check against, non-thinking (stable output lengths).
2. Gemma 4 E4B: current and multimodal, with official QAT. But the memory math is complex and quantization-tool support uncertain.
3. Qwen3.5-4B: newest, ~4× less KV, but quantization support for DeltaNet layers is untested.
4. Spec default Qwen3-8B: ~2× the cost per run for little extra learning.

## Decision
- Primary **Qwen3-4B-Instruct-2507** · draft **Qwen3-0.6B** · CPU tier **Qwen3-0.6B, Qwen3-1.7B, Gemma 4 E2B** · MoE **Qwen3-30B-A3B**.
- New **E15**: Qwen3-4B vs Qwen3.5-4B vs Gemma 4 E4B: theoretical (memcalc) vs engine-realised KV capacity, throughput, quality.
- E3 adds **Gemma 4 QAT vs our PTQ**.
- Rejected: Llama 3.x (gated, restrictive license), LFM2 (custom license), Granite-H-Tiny / Gemma 26B-A4B as MoE (less clean comparison).

## Consequences
+ About half the GPU cost of the spec's 8B plan; every CPU-tier model fits the local box.
+ A distinctive, timely result (E15) that exercises memcalc's per-layer model.
− memcalc became larger (per-layer kinds, PLE, state dtype). − Results on a 4B model may not transfer to 70B-class serving; stated as a limitation.
