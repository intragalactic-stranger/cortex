# 02 · Memory math: weights, KV cache, and how many users fit

GPU memory during serving = **weights + KV cache + runtime reserve**. Whatever's left after weights and reserve decides how many sequences you can serve at once.

## 1. Weights
`weights ≈ params × bytes_per_param`

| Format | Bytes/param | Qwen3-4B weights (memcalc) |
|---|---|---|
| bf16 | 2 | 7.49 GiB |
| fp8 / int8 | 1 (embeddings stay bf16) | 4.11 GiB |
| int4 GPTQ (W4A16, group 128) | 0.5 + 2/128 scale overhead | 2.47 GiB |
| GGUF Q4_K_M (~4.89 bits avg, embeddings quantized too) | ~0.61 | 2.29 GiB |

Note: llm-compressor-style formats leave the embedding/lm_head in 16-bit. For a 4B model that's 0.78 GB that *doesn't* shrink, which is why int4 isn't a clean 4× reduction.

## 2. KV cache
Each attention layer stores a key and a value vector per KV head for every past token:

```
KV bytes per token (per layer) = 2 (K and V) × kv_heads × head_dim × bytes
```

**Worked example (spec §3.2, Llama-3.1-8B, bf16):** 32 layers × 2 × 8 KV heads × 128 × 2 B = **131,072 B = 128 KiB/token**. At 8K tokens that's exactly **1 GiB per sequence**.

**Qwen3-4B:** 36 × 2 × 8 × 128 × 2 = **144 KiB/token → 1,152 MiB per 8K sequence**.
- With MHA (32 KV heads) it would be 576 KiB/token. **GQA saves 4×.**
- With an **FP8 KV cache** it's 72 KiB/token, which halves it again.

## 3. How many sequences fit (Qwen3-4B, bf16, L4, 8K context)
```
budget   = 22.49 GiB × 0.90 utilisation = 20.24 GiB
weights  =  7.49 GiB
reserve  =  1.50 GiB   (CUDA context, CUDA graphs, activations; refit from measurements in M2)
free KV  = 11.25 GiB  →  11.25 / 1.125 GiB per sequence  =  9 concurrent 8K sequences
```
Change one knob at a time: FP8 KV → **19**; int4 weights → **14**; 32K context → **2**. Context length is the most expensive knob.

## 4. The per-layer view (why memcalc iterates layers)
The formula above assumes every layer is a full-attention layer. Newer models break that assumption:

| Layer kind | Memory per sequence | Model |
|---|---|---|
| full attention | grows with every token | all Qwen3 layers |
| sliding window | capped at `window` tokens (512) | 20 of Gemma 4 E4B's 24 KV-owning layers |
| KV-shared | **zero**: reads another layer's KV | Gemma 4 E4B's last 18 layers |
| linear (Gated DeltaNet) | fixed state, 128×128 per head, **fp32** | 24 of Qwen3.5-4B's 32 layers |

Per 8K sequence: **Qwen3-4B 1,152 MiB · Qwen3.5-4B 305 MiB · Gemma 4 E4B 148 MiB.**

But Gemma's weights are 14.0 GiB, because its 2.8B-param PLE table counts as device memory. So on an L4 it fits *fewer* 8K sequences than Qwen3.5 (32 vs 36), unless the engine keeps the PLE table off the GPU (then 69). **Per-sequence memory isn't the whole story; weights compete for the same budget.**

## Try it
```bash
make demo
uv run memcalc estimate -m Qwen/Qwen3-4B-Instruct-2507 -d L4 --kv-dtype fp8
```

## Interview questions this answers
1. Derive the KV-cache-per-token formula. Why the factor 2?
2. How many concurrent 8K-token users fit on an L4 for a 4B model, and what are the levers?
3. Why does GQA matter more for serving than for training?
4. Why doesn't int4 shrink a small model's memory by a clean 4×?
5. How do sliding-window attention and linear attention change KV memory?
