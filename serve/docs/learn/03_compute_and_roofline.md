# 03 · Compute and the roofline: why decode is memory-bound

## Two phases with different bottlenecks
- **Prefill** processes the whole prompt in one forward pass. Thousands of tokens share each weight read, so it's **compute-bound**.
- **Decode** generates one token per sequence per step, and every step re-reads all active weights (plus the KV cache) from memory. With one sequence, the GPU does ~2 FLOPs per weight byte loaded, so it's **memory-bandwidth-bound**.

## The single-stream decode ceiling
```
max tokens/s (one sequence) ≈ memory bandwidth / bytes read per token
```
**Spec §3.3 example (8B, bf16, ~16 GB):** H100 3.35 TB/s → ~200 tok/s; L4 300 GB/s → ~19 tok/s.

memcalc gives **223 tok/s** on H100 for Llama-3.1-8B, because the *input* embedding table (0.5B params) is a row lookup, not a full read. Counting all weights gives 208.6 tok/s, within 5% of the spec. *Small modelling choices move these numbers by 10%. State your assumptions.*

**Qwen3-4B on L4:** bf16 **37 tok/s** · fp8 **68** · int4 **113**. Lower-bit weights speed up decode almost linearly because fewer bytes are read per token. That's the main reason quantization matters for serving.

## Batching moves you along the roofline
With batch B, one weight read serves B tokens:
```
FLOPs per step = 2 × P × B        bytes per step ≈ P × b  (+ KV)
arithmetic intensity = 2B / b  FLOPs/byte
compute-bound once  B > (peak FLOPS / bandwidth) × b / 2
```
| Device | Weights | Crossover batch B* |
|---|---|---|
| L4 (121 TFLOPS bf16, 300 GB/s) | bf16 | **403** |
| L4 | int4 (W4A16) | **104** |
| H100 (989 TFLOPS, 3.35 TB/s) | bf16 | **295** |

Below B*, adding sequences is almost free throughput. That's why continuous batching (doc 04) is the biggest single win in serving. Quantized weights reach the compute roof sooner. In practice, KV memory usually runs out first: Qwen3-4B on an L4 fits only 9 × 8K sequences, far below 403. **Memory capacity, not FLOPs, caps decode throughput for long contexts.**

## Prefill cost
`FLOPs ≈ 2 × active_params × tokens` (+ attention ≈ 4 × heads × head_dim × context per token per layer).
Qwen3-4B: **8.0 TFLOP per 1K prompt tokens**. At 121 TFLOPS peak that's ~66 ms ideal, so expect ~130 ms at a realistic ~50% utilisation. A 2K-token RAG prompt therefore starts with ~¼ s of TTFT on an L4 before any queueing.

## Latency metrics (what servebench will measure)
| Metric | Definition | Dominated by |
|---|---|---|
| **TTFT** | time to first token | queueing + prefill |
| **ITL / TPOT** | gap between output tokens | decode step time (batch size, bandwidth) |
| **E2E** | request start → last token | TTFT + output_tokens × ITL |
| **Throughput** | output tokens/s across all requests | batch size |
| **Goodput** | throughput counting only requests that meet the SLO (e.g. TTFT p95 ≤ 500 ms *and* ITL p95 ≤ 50 ms) | the real business metric |

Throughput keeps rising with load while latency degrades. Goodput peaks and then falls. **The operating point you sell is max goodput, not max throughput.**

## Interview questions this answers
1. Why is decode memory-bound and prefill compute-bound?
2. Compute the decode ceiling for a 4B bf16 model on an L4.
3. Why does batching help decode, and where does it stop helping?
4. Why do lower-bit weights speed up decode even without faster math?
5. Throughput vs goodput: which do you optimise and why?
