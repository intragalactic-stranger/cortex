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
# Qwen3-4B-Instruct-2507 · W4A16 GPTQ (group 128)

INT4 weights, group size 128, symmetric, static activation order; activations stay bf16.

Produced and evaluated in the [Cortex Serve](https://github.com/intragalactic-stranger/cortex) project (milestone M3). Every number below is
reproducible from a committed config and manifest; full tables are in `serve/RESULTS.md` §M3.

## How it was made

| | |
|---|---|
| Base model | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) @ `cdbee75f` |
| Method | GPTQ int4 weights, group 128, symmetric, static actorder |
| Calibration | `HuggingFaceH4/ultrachat_200k` @ `8049631c`, 512 samples × 2048 tokens (sample-id digest `66bc04b0f6e4882a`) |
| Tools | llmcompressor 0.15.0, compressed-tensors 0.20.1, torch 2.14.1, transformers 5.21.0 |
| Hardware / time | L4, 986 s |
| Size on disk | 2.48 GiB |
| Source commit | [`9b0cc3fe`](https://github.com/intragalactic-stranger/cortex/commit/9b0cc3fe97efa0905b0b10d04e7f0745067ccd6f) |

## Usage

```bash
vllm serve intragalactic-stranger/Qwen3-4B-Instruct-2507-W4A16-GPTQ
```

## Quality (same items for every variant; ± = 95% CI from lm-eval's standard error)

Served with vLLM 0.32.0, scored with lm-evaluation-harness 0.4.13 in the same container.

| Benchmark | This model | bf16 | Δ |
|---|---|---|---|
| ARC-Challenge (acc_norm, 1,172) | 58.4 ±2.8 | 58.3 | +0.1 |
| HellaSwag (acc_norm, first 2,000) | 56.9 ±2.2 | 58.5 | -1.6 |
| MMLU-Pro (logprob MC, first 1,000) | 32.2 ±2.9 | 34.5 | -2.3 |
| GSM8K 5-shot (strict) | 73.5 ±2.4 | 72.8 | +0.7 |
| GSM8K 5-shot (flexible extract) | 79.6 ±2.2 | 78.8 | +0.8 |
| IFEval (prompt-level strict) | 82.4 ±3.2 | 83.4 | -0.9 |
| WikiText-2 word perplexity (lower is better) | 15.048 | 14.266 | +5.5% |
| Held-out US financial regulation perplexity (lower is better) | 6.402 | 5.965 | +7.3% |

## Speed (vLLM 0.32.0, Qwen chat workload 512 in / 256 out, closed-loop concurrency)

| GPU | Metric | bf16 | This model |
|---|---|---|---|
| NVIDIA L4 | ms per output token, 1 user | 33.3 | 12.2 |
| NVIDIA L4 | output tokens/s, 64 users | 792 | 1,045 |
| NVIDIA H100 80GB | ms per output token, 1 user | 4.15 | 3.66 |
| NVIDIA H100 80GB | output tokens/s, 128 users | 10,079 | 9,619 |

## Caveats

- Weight-only INT4 is 2.7× faster per token than bf16 on an L4, but slower than bf16 on a busy H100 (dequantization becomes the bottleneck at high batch). Prefer FP8 for high-concurrency H100 serving.
- Perplexity +5.5%; MMLU-Pro −2.3 (inside the CI); reasoning and instruction tasks within noise.

## License

Apache-2.0, inherited from [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
