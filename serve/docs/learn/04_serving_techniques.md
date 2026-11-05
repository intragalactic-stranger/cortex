# 04 · Serving techniques: how engines turn the math into throughput

| Technique | Problem it solves | Mechanism | Experiment |
|---|---|---|---|
| **Continuous batching** | Static batches wait for the slowest request | Scheduler adds/removes sequences every decode step | E1 |
| **PagedAttention** | KV cache fragmentation wastes 60–80% of memory with contiguous per-request buffers | KV stored in fixed-size blocks (e.g. 16 tokens) mapped through a block table, like OS virtual-memory pages | E1, E8 |
| **Prefix caching** (vLLM APC, SGLang **RadixAttention**) | Shared system prompts are re-prefilled for every request | Hash/radix-tree of KV blocks; a request whose prefix matches reuses cached blocks | E4 |
| **Chunked prefill** | A long prompt's prefill stalls everyone's decode (ITL spikes) | Split prefill into chunks interleaved with decode steps | E5 |
| **Speculative decoding** | Decode is memory-bound, so FLOPs are idle | A cheap drafter proposes γ tokens; the target verifies them in one pass | E6 |
| **Disaggregated prefill/decode** | Prefill and decode want different hardware/batching | Separate pools; ship KV between them | (reading) |
| **Tensor / pipeline / expert parallelism** | Model or KV too big for one GPU | Split matrices across GPUs / layers across GPUs / experts across GPUs | E12 (optional) |
| **Structured decoding** (XGrammar) | Need guaranteed-valid JSON | Mask logits with a grammar automaton at each step | E10 |
| **Multi-LoRA** | Many fine-tunes of one base | One base in memory, small adapters batched together | E11 (after P4) |

## PagedAttention ↔ OS paging
| OS virtual memory | PagedAttention |
|---|---|
| Page (4 KiB) | KV block (e.g. 16 tokens × all KV heads) |
| Page table | Block table per sequence |
| Physical frames allocated on demand | Blocks allocated as the sequence grows |
| Copy-on-write for forked processes | Shared blocks for parallel sampling / shared prefixes |

Result: near-zero fragmentation, so more sequences fit, so the batch is bigger, so throughput is higher.

## Speculative decoding math
With per-token acceptance probability α and draft length γ, the expected number of tokens accepted per target forward pass is

```
E[tokens] = (1 − α^(γ+1)) / (1 − α)
```
α = 0.7, γ = 4 → **2.77 tokens per pass**. At batch 1 the target pass costs about the same as a single-token step (memory-bound), so that's close to a 2.7× speedup minus draft cost.

**Why gains shrink at high concurrency:** at large batch the target is already closer to compute-bound (doc 03). Verifying γ extra tokens per sequence then costs real FLOPs, and rejected drafts are pure waste. E6 measures this at concurrency 1 vs high. Our draft for Qwen3-4B is **Qwen3-0.6B** (same tokenizer). Qwen3.5 and Gemma 4 also ship native MTP heads (memcalc counts Qwen3.5-4B's at 121M params).

## Interview questions this answers
1. Explain PagedAttention using OS paging.
2. Continuous vs static batching: why is the difference so large?
3. When does prefix caching help a RAG system, and how should you order the prompt to maximise hits?
4. Derive the expected speedup of speculative decoding. Why does it fade under load?
5. What does chunked prefill trade off?
