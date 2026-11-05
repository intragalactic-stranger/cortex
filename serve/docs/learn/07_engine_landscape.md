# 07 · The engine landscape: which runtime, when

| Engine | Sweet spot | Key ideas | Our use |
|---|---|---|---|
| **vLLM** | Datacenter GPU throughput, broadest model support | PagedAttention, continuous batching, Automatic Prefix Caching, chunked prefill, speculative decoding, FP8/INT4 Marlin kernels, hybrid KV-cache manager for sliding-window/linear layers | Primary GPU engine (M2+), production (M5) |
| **SGLang** | Prefix-heavy and structured workloads, agents | **RadixAttention** (radix tree of cached prefixes), fast structured decoding, strong MoE support | GPU comparison (E1, E4, E15) |
| **llama.cpp** (`llama-server`) | CPU, edge, laptops, mixed CPU/GPU | GGUF k-quants, importance matrices, mmap loading, SWA cache, draft-model speculation | CPU tier (M1), GPU comparison (E1) |
| **Ollama** | Developer UX on a laptop | Wraps llama.cpp with a model registry and an API | CPU tier comparison (E13): what does convenience cost? |
| **TensorRT-LLM** | Maximum NVIDIA performance | Compiled engines, in-flight batching | Optional (not planned) |
| **MLX / LM Studio** | Apple Silicon | Unified-memory-native kernels | Out of scope (no Mac) |

## How to choose
```
Need many concurrent users on NVIDIA GPUs?           → vLLM or SGLang (measure both: E1)
  Heavy shared prefixes / agents / JSON outputs?      → test SGLang's RadixAttention and grammar speed (E4, E10)
  New architecture (hybrid, sliding window)?          → check which engine actually exploits it (E15)
Single user, offline, CPU or laptop?                  → llama.cpp (or Ollama for convenience), Q4_K_M-class GGUF
Absolute peak on fixed hardware and a frozen model?   → consider TensorRT-LLM
```
These are working hypotheses. The decision guide (M7) will contain only conclusions our measurements support.

## Our hardware reality
- **Local box:** AMD EPYC 7571, 4 threads, **AVX2 only (no AVX-512/AMX)**. SGLang's CPU backend needs AMX, and vLLM's CPU backend targets AVX-512, so locally it's llama.cpp and Ollama.
- **Modal GPUs:** L4 (main tier), H100 (FP8, MoE, high concurrency). Same engines, pinned container digests.

## Interview questions this answers
1. vLLM vs SGLang vs llama.cpp: when would you pick each?
2. Why does Ollama exist if llama.cpp does the work, and what might it cost you?
3. How would you decide whether to adopt a new engine version in production?
