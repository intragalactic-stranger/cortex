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
# Qwen3-4B-Instruct-2507 · W4A16 GPTQ calibrated on US financial regulation (research artifact)

Same recipe as the generic GPTQ release, calibrated only on eCFR Titles 12 and 17 (banking, securities).

Produced and evaluated in the [Cortex Serve](https://github.com/intragalactic-stranger/cortex) project (milestone M3). Every number below is
reproducible from a committed config and manifest; full tables are in `serve/RESULTS.md` §M3.

## How it was made

| | |
|---|---|
| Base model | [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507) @ `cdbee75f` |
| Method | GPTQ int4 weights, group 128, symmetric, static actorder |
| Calibration | `Azzindani/US_Regulation_ECFR_20260101` @ `04148fee`, 512 samples × 2048 tokens (sample-id digest `4257b9acdf26e457`) |
| Tools | llmcompressor 0.15.0, compressed-tensors 0.20.1, torch 2.14.1, transformers 5.21.0 |
| Hardware / time | L4, 698 s |
| Size on disk | 2.48 GiB |
| Source commit | [`77b83aec`](https://github.com/intragalactic-stranger/cortex/commit/77b83aec886c97c4875fa4ab458e4ec5c75fff23) |

## Usage

```bash
vllm serve intragalactic-stranger/Qwen3-4B-Instruct-2507-W4A16-GPTQ-BFSI
```

## Quality (same items for every variant; ± = 95% CI from lm-eval's standard error)

Served with vLLM 0.32.0, scored with lm-evaluation-harness 0.4.13 in the same container.

| Benchmark | This model | bf16 | Δ |
|---|---|---|---|
| ARC-Challenge (acc_norm, 1,172) | 58.7 ±2.8 | 58.3 | +0.4 |
| HellaSwag (acc_norm, first 2,000) | 57.6 ±2.2 | 58.5 | -0.9 |
| MMLU-Pro (logprob MC, first 1,000) | 31.7 ±2.9 | 34.5 | -2.8 |
| GSM8K 5-shot (strict) | 34.3 ±2.6 | 72.8 | -38.4 |
| GSM8K 5-shot (flexible extract) | 75.6 ±2.3 | 78.8 | -3.2 |
| IFEval (prompt-level strict) | 79.5 ±3.4 | 83.4 | -3.9 |
| WikiText-2 word perplexity (lower is better) | 15.454 | 14.266 | +8.3% |
| Held-out US financial regulation perplexity (lower is better) | 6.143 | 5.965 | +3.0% |

## Speed (vLLM 0.32.0, Qwen chat workload 512 in / 256 out, closed-loop concurrency)

Not measured for this artifact.

## Caveats

- **Research artifact, not a general-purpose model.** Domain-only calibration halved the in-domain perplexity damage (+3.0% vs +7.3%) but weakened format-following: only 45% of GSM8K answers keep the `#### N` format (80% for generic calibration), so strict GSM8K drops 38 points while arithmetic (flexible extract) drops 3.2. IFEval −3.9. Calibrate on a domain + general-chat mix instead.
- Speed is identical to the generic GPTQ release (same format).

## License

Apache-2.0, inherited from [`Qwen/Qwen3-4B-Instruct-2507`](https://huggingface.co/Qwen/Qwen3-4B-Instruct-2507).
