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
# Qwen3-4B-Instruct-2507 · W4A16 AWQ (asymmetric, group 128)

AWQ activation-aware scales followed by round-to-nearest INT4, group 128, asymmetric (zero point).

Produced and evaluated in the [Cortex Serve](https://github.com/intragalactic-stranger/cortex) project (milestone M3). Every number below is
reproducible from a committed config and manifest; full tables are in `serve/RESULTS.md` §M3.

## How it was made

| | |
|---|---|
| Base model | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) @ `cdbee75f` |
| Method | AWQ scales + RTN int4 weights, group 128, asymmetric |
| Calibration | `HuggingFaceH4/ultrachat_200k` @ `8049631c`, 512 samples × 2048 tokens (sample-id digest `66bc04b0f6e4882a`) |
| Tools | llmcompressor 0.15.0, compressed-tensors 0.20.1, torch 2.14.1, transformers 5.21.0 |
| Hardware / time | L40S, 1127 s |
| Size on disk | 2.49 GiB |
| Source commit | [`d796b1bf`](https://github.com/intragalactic-stranger/cortex/commit/d796b1bf555e37e64f8de57ca993e5a2c74bdcaa) |

## Usage

```bash
vllm serve intragalactic-stranger/Qwen3-4B-Instruct-2507-W4A16-AWQ
```

## Quality (same items for every variant; ± = 95% CI from lm-eval's standard error)

Served with vLLM 0.32.0, scored with lm-evaluation-harness 0.4.13 in the same container.

| Benchmark | This model | bf16 | Δ |
|---|---|---|---|
| ARC-Challenge (acc_norm, 1,172) | 56.7 ±2.8 | 58.3 | -1.6 |
| HellaSwag (acc_norm, first 2,000) | 58.1 ±2.2 | 58.5 | -0.4 |
| MMLU-Pro (logprob MC, first 1,000) | 32.6 ±2.9 | 34.5 | -1.9 |
| GSM8K 5-shot (strict) | 76.0 ±2.3 | 72.8 | +3.2 |
| GSM8K 5-shot (flexible extract) | 80.5 ±2.1 | 78.8 | +1.7 |
| IFEval (prompt-level strict) | 82.6 ±3.2 | 83.4 | -0.7 |
| WikiText-2 word perplexity (lower is better) | 14.944 | 14.266 | +4.7% |
| Held-out US financial regulation perplexity (lower is better) | 6.299 | 5.965 | +5.6% |

## Speed (vLLM 0.32.0, Qwen chat workload 512 in / 256 out, closed-loop concurrency)

| GPU | Metric | bf16 | This model |
|---|---|---|---|
| NVIDIA L4 | ms per output token, 1 user | 33.3 | 12.3 |
| NVIDIA L4 | output tokens/s, 64 users | 792 | 1,038 |

## Caveats

- H100 speed not measured for this variant.
- Quantized on an L40S: AWQ's scale search ran out of memory on a 24 GB L4 at 512 × 2,048 calibration.

## License

Apache-2.0, inherited from [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
