# 0002 · memcalc: per-layer memory model, exact params from safetensors headers

- **Status:** Accepted · **Date:** 2026-11-05 · **Related:** D-011, D-014, milestone M0

## Context
The spec's formula (`2 × layers × kv_heads × head_dim × bytes`) assumes every layer is a full-attention layer. Our lineup includes Qwen3.5 (24 of 32 layers Gated DeltaNet with fp32 state), Gemma 4 (sliding window 512, last 18/20 layers reuse KV, global layers with head_dim 512, 2.8B-param per-layer-embedding tables) and MoE. Parameter counts from model cards or metadata are ambiguous: they include vision/audio towers, MTP heads, and (Qwen3-0.6B/1.7B) a redundant `lm_head` stored despite tied embeddings.

## Options considered
1. **Closed-form formula per family**: simple, but every new architecture is a special case; wrong for hybrids.
2. **Per-layer spec list** built from `layer_types`, `num_kv_shared_layers`, `global_head_dim`, linear-attention fields; one arithmetic loop over layers.
3. Params from **HF metadata totals** (one number, no breakdown) vs **safetensors headers** (every tensor's shape, via two HTTP range requests per shard, no weight download) vs **analytic formula**.

## Decision
- Option 2 for memory. Layer kinds: `full`, `sliding`, `shared`, `linear`.
- Params come from **safetensors headers**, classified by tensor name into decoder / experts / embeddings / lm_head / PLE / vision / audio / MTP / unused-shared-KV / tied-duplicate. The analytic formula is kept for llama-style models and must match the header **exactly** (tests).
- Report **stored / loaded / active** params separately. Defaults: text components only, PLE on device (flag `--ple-offload`), MTP not loaded (flag `--with-mtp`).
- llm-compressor formats keep embeddings/lm_head in 16-bit; GGUF quantizes them.
- Python is the reference implementation; the M6 TypeScript port must pass the same `golden.json`.
- Fixed runtime reserve (1.5 GiB GPU / 0.5 GiB CPU), refit from M2 measurements.

## Consequences
+ Correct theoretical numbers for all lineup architectures; config interpretation is cross-checked against real tensor shapes in tests.
+ Surfaced two non-obvious facts: Qwen3-0.6B's 156M duplicate `lm_head`; Gemma 4 E4B's concurrency on an L4 swings 32 → 69 depending on PLE placement.
− Predictions are *theoretical*: engines may not exploit sliding windows or KV sharing fully, and block rounding/overheads are not modelled. E15 and M2 measure the gap.
− Unknown layer types (e.g. Mamba) raise an explicit error rather than guess.
