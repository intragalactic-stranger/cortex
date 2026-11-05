# Cortex Serve

> **Question:** for a given model, latency SLO and budget, which engine + quantization + configuration
> delivers the most goodput per dollar without unacceptable quality loss?

Cortex Serve is a reproducible inference lab plus production-style serving for the rest of Cortex.
It is being built milestone by milestone (see [`docs/SPEC.md`](docs/SPEC.md)).

## M0: `memcalc`

`memcalc` predicts, for any Hugging Face model on any device:

- weight memory (per format: bf16, fp8, int4 GPTQ/AWQ, GGUF k-quants), counted **exactly** from the
  checkpoint's safetensors header, fetched with HTTP range requests and **without downloading the weights**
- KV cache per token and per sequence, **layer by layer**, so it handles more than the textbook
  dense transformer:
  - **sliding-window** layers (KV capped at the window)
  - **cross-layer KV sharing** (Gemma 4: 18 of 42 layers own no KV)
  - **linear-attention / Gated DeltaNet** layers (Qwen3.5: fixed fp32 state instead of a growing cache)
  - **MoE** (all experts loaded, only the routed ones read per token)
- max concurrent sequences, single-stream decode ceiling (memory-bandwidth bound), the batch size at
  which decode turns compute-bound, and prefill FLOPs

```text
$ make demo
Memory per sequence and max concurrency on L4 (bf16 weights, auto KV)
 model              layers                        weights GiB  KV/token KiB  MiB/seq @8K  max seqs @8K  MiB/seq @32K  max seqs @32K
 Qwen3-4B-2507      36 full                              7.49         144.0        1,152             9         4,608              2
 Qwen3.5-4B         8 full, 24 linear                    7.83          32.0          305            36         1,073             10
 gemma-4-E4B-it     4 full, 20 sliding, 18 shared       14.00          16.0          148            32           532              9
```

Three ~4B models need **1,152 vs 305 vs 148 MiB** of memory per 8K-token sequence. Experiment E15 will
measure how much of that theoretical advantage vLLM, SGLang and llama.cpp actually deliver.

### Things memcalc caught that the usual formulas miss
- **Qwen3-0.6B's "752M params" includes a 156M duplicate:** the config ties `lm_head` to the embeddings,
  but the checkpoint also stores it. The real unique count is 596M.
- **Gemma 4 E4B stores 8.0B params but computes with ~4.6B.** 2.8B of it is a per-layer-embedding lookup
  table. Whether the engine keeps that table on the GPU changes max concurrency at 8K from **32 to 69**.
- Every config interpretation is cross-checked against real tensor shapes in the tests.

### Usage
```bash
uv run memcalc devices
uv run memcalc estimate -m Qwen/Qwen3-4B-Instruct-2507 -d L4 --dtype bf16 --kv-dtype fp8 --ctx 8192
uv run memcalc estimate -m google/gemma-4-E4B-it -d L4 --ple-offload --json
uv run memcalc compare -m Qwen/Qwen3-4B-Instruct-2507 -m Qwen/Qwen3.5-4B --ctx 8192 --ctx 32768
```

## Docs
- [`docs/learn/`](docs/learn/): 8 short learning docs on inference from first principles
- [`docs/adr/`](docs/adr/): architecture decision records
- [`docs/VERSIONS.md`](docs/VERSIONS.md) · [`RESULTS.md`](RESULTS.md)
