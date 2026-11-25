---
license: apache-2.0
base_model: Qwen/Qwen3-4B-Instruct-2507
base_model_relation: quantized
tags:
  - qwen3
  - quantized
  - compressed-tensors
  - llm-compressor
  - vllm
---
# Qwen3-4B-Instruct-2507 · W8A8 INT8 (SmoothQuant + GPTQ)

INT8 weights (per channel) and dynamic INT8 activations (per token), SmoothQuant strength 0.8 then GPTQ.

Produced and evaluated in the [Cortex Serve](https://github.com/intragalactic-stranger/cortex) project (milestone M3). Every number below is
reproducible from a committed config and manifest; full tables are in `serve/RESULTS.md` §M3.

## How it was made

| | |
|---|---|
| Base model | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) @ `cdbee75f` |
| Method | SmoothQuant 0.8 + GPTQ int8 W per-channel / A per-token dynamic |
| Calibration | `HuggingFaceH4/ultrachat_200k` @ `8049631c`, 512 samples × 2048 tokens (sample-id digest `66bc04b0f6e4882a`) |
| Tools | llmcompressor 0.15.0, compressed-tensors 0.20.1, torch 2.14.1, transformers 5.21.0 |
| Hardware / time | L4, 1327 s |
| Size on disk | 4.12 GiB |
| Source commit | [`d05daa83`](https://github.com/intragalactic-stranger/cortex/commit/d05daa83829b1a0fddb135de64ed74f0f100b6a9) |

## Usage

```bash
vllm serve intragalactic-stranger/Qwen3-4B-Instruct-2507-W8A8-INT8
```

## Quality (same items for every variant; ± = 95% CI from lm-eval's standard error)

Served with vLLM 0.32.0, scored with lm-evaluation-harness 0.4.13 in the same container.

| Benchmark | This model | bf16 | Δ |
|---|---|---|---|
| ARC-Challenge (acc_norm, 1,172) | 59.0 ±2.8 | 58.3 | +0.7 |
| HellaSwag (acc_norm, first 2,000) | 58.7 ±2.2 | 58.5 | +0.2 |
| MMLU-Pro (logprob MC, first 1,000) | 34.5 ±2.9 | 34.5 | +0.0 |
| GSM8K 5-shot (strict) | 71.0 ±2.4 | 72.8 | -1.7 |
| GSM8K 5-shot (flexible extract) | 79.2 ±2.2 | 78.8 | +0.5 |
| IFEval (prompt-level strict) | 82.4 ±3.2 | 83.4 | -0.9 |
| WikiText-2 word perplexity (lower is better) | 14.338 | 14.266 | +0.5% |
| Held-out US financial regulation perplexity (lower is better) | 5.977 | 5.965 | +0.2% |

## Speed (vLLM 0.32.0, Qwen chat workload 512 in / 256 out, closed-loop concurrency)

| GPU | Metric | bf16 | This model |
|---|---|---|---|
| NVIDIA L4 | ms per output token, 1 user | 33.3 | 20.8 |
| NVIDIA L4 | output tokens/s, 64 users | 792 | 994 |
| NVIDIA H100 80GB | ms per output token, 1 user | 4.15 | 3.30 |
| NVIDIA H100 80GB | output tokens/s, 128 users | 10,079 | 11,673 |

## Caveats

- Runs on CUTLASS INT8 kernels in vLLM (Ampere and newer).

## License

Apache-2.0, inherited from [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
