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
# Qwen3-4B-Instruct-2507 · FP8 dynamic (W8A8)

FP8 E4M3 weights (per channel) and dynamically quantized FP8 activations (per token). Data-free.

Produced and evaluated in the [Cortex Serve](https://github.com/intragalactic-stranger/cortex) project (milestone M3). Every number below is
reproducible from a committed config and manifest; full tables are in `serve/RESULTS.md` §M3.

## How it was made

| | |
|---|---|
| Base model | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) @ `cdbee75f` |
| Method | FP8 E4M3 weights per-channel + dynamic per-token activations (data-free) |
| Calibration | none (data-free) |
| Tools | llmcompressor 0.15.0, compressed-tensors 0.20.1, torch 2.14.1, transformers 5.21.0 |
| Hardware / time | L4, 18 s |
| Size on disk | 4.12 GiB |
| Source commit | [`1c526987`](https://github.com/intragalactic-stranger/cortex/commit/1c526987b4ada4368b490e8a30eb130d13fef352) |

## Usage

```bash
vllm serve intragalactic-stranger/Qwen3-4B-Instruct-2507-FP8-Dynamic
```

## Quality (same items for every variant; ± = 95% CI from lm-eval's standard error)

Served with vLLM 0.32.0, scored with lm-evaluation-harness 0.4.13 in the same container.

| Benchmark | This model | bf16 | Δ |
|---|---|---|---|
| ARC-Challenge (acc_norm, 1,172) | 58.7 ±2.8 | 58.3 | +0.4 |
| HellaSwag (acc_norm, first 2,000) | 58.1 ±2.2 | 58.5 | -0.3 |
| MMLU-Pro (logprob MC, first 1,000) | 34.1 ±2.9 | 34.5 | -0.4 |
| GSM8K 5-shot (strict) | 74.1 ±2.4 | 72.8 | +1.3 |
| GSM8K 5-shot (flexible extract) | 79.6 ±2.2 | 78.8 | +0.8 |
| IFEval (prompt-level strict) | 81.3 ±3.3 | 83.4 | -2.0 |
| WikiText-2 word perplexity (lower is better) | 14.446 | 14.266 | +1.3% |
| Held-out US financial regulation perplexity (lower is better) | 5.977 | 5.965 | +0.2% |

## Speed (vLLM 0.32.0, Qwen chat workload 512 in / 256 out, closed-loop concurrency)

| GPU | Metric | bf16 | This model |
|---|---|---|---|
| NVIDIA L4 | ms per output token, 1 user | 33.3 | 21.1 |
| NVIDIA L4 | output tokens/s, 64 users | 792 | 996 |
| NVIDIA H100 80GB | ms per output token, 1 user | 4.15 | 3.12 |
| NVIDIA H100 80GB | output tokens/s, 128 users | 10,079 | 11,710 |

## Caveats

- FP8 compute needs compute capability ≥ 8.9 (Ada L4/L40S, Hopper H100). Older GPUs fall back to weight-only FP8.
- Quality matched both bf16 and Qwen's own FP8 release within the confidence intervals.

## License

Apache-2.0, inherited from [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
