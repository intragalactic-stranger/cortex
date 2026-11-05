# 01 · Model anatomy: what's actually inside a decoder LLM

Everything in inference follows from a handful of `config.json` fields. Here they are, read off our primary model, **Qwen3-4B-Instruct-2507**.

| Field | Qwen3-4B | Meaning |
|---|---|---|
| `num_hidden_layers` | 36 | Transformer blocks stacked in sequence |
| `hidden_size` (d) | 2560 | Width of the residual stream |
| `num_attention_heads` | 32 | Query heads |
| `num_key_value_heads` | 8 | K/V heads (GQA: 4 query heads share each K/V head) |
| `head_dim` | 128 | Per-head width. **Note 32 × 128 = 4096 ≠ 2560**: Qwen3 decouples head_dim from d |
| `intermediate_size` | 9728 | FFN width (SwiGLU: gate, up, down) |
| `vocab_size` | 151,936 | Token embedding rows |
| `tie_word_embeddings` | true | The output projection reuses the embedding matrix |

## One block, counted

```
attention : q 2560×4096 + k 2560×1024 + v 2560×1024 + o 4096×2560 + q/k norms (2×128) = 26.2M
FFN       : 3 × 2560 × 9728 (SwiGLU gate/up/down)                                     = 74.7M
norms     : 2 × 2560                                                                  = 0.005M
block                                                                                 = 100.93M
× 36 blocks = 3.634B  +  embeddings 151,936 × 2560 = 0.389B  +  final norm  =  4,022,468,096
```

memcalc's analytic formula reproduces the checkpoint's safetensors header **to the parameter** (test `test_analytic_matches_header_exactly`). Two thirds of every block is FFN, and the embedding table is ~10% of this small model. That's why small models with huge vocabularies (Gemma's 262K) carry a lot of "dead weight" for compute.

## Attention variants: MHA → MQA → GQA
- **MHA:** every query head has its own K/V head (`kv_heads = q_heads`).
- **MQA:** one shared K/V head. Smallest KV cache, but some quality loss.
- **GQA:** groups of query heads share a K/V head. Qwen3-4B uses 8 K/V heads for 32 query heads, which gives **4× less KV cache than MHA** with near-MHA quality (doc 02).

## RoPE
Positions are injected by rotating q and k vectors by an angle proportional to position (`rope_theta` sets the frequency base; Qwen3-4B-2507 uses 5,000,000 for long context). RoPE scaling (YaRN, etc.) stretches those frequencies to extend context beyond training length.

## Beyond the textbook: what 2026 small models change
| Model | Twist | Memory consequence |
|---|---|---|
| **Qwen3-30B-A3B** (MoE) | 128 experts per layer, 8 routed per token | 30.5B stored, **3.04B active**: memory scales with total, per-token compute with active |
| **Qwen3.5-4B** (hybrid) | 24 of 32 layers are Gated DeltaNet (linear attention) | Those layers keep a fixed-size state, not a growing KV cache |
| **Gemma 4 E4B** | Sliding-window layers, cross-layer KV sharing, Per-Layer Embeddings (PLE) | "E4B" = ~4.6B *active* params, but **8.0B stored**: 2.8B is a PLE lookup table |

**MoE in one sentence:** every expert must be loaded (memory = all 30.5B), but each token only multiplies through 8 of 128 (compute and bytes read ≈ 3B). So MoE models are memory-hungry but fast per token.

## Try it
```bash
uv run memcalc estimate -m Qwen/Qwen3-30B-A3B -d H100-80GB   # stored vs active params
```

## Interview questions this answers
1. Where do the parameters of a 4B model live? (blocks ≈ 90%, two thirds of each is FFN)
2. MHA vs MQA vs GQA: what's the tradeoff?
3. Why does an MoE model with 3B active params need 60 GB of memory?
4. What is RoPE, and how is context extended beyond the training length?
5. What does "E4B" mean for Gemma, and why does it matter for serving?
