# 05 · Quantization: fewer bytes per weight, and what it costs

## The menu
| Family | Formats | What's quantized | Calibration | Where it runs |
|---|---|---|---|---|
| Weight-only PTQ | **GPTQ**, **AWQ** (int4, group 128), bitsandbytes NF4 | weights; activations stay 16-bit (W4A16) | ~512 samples | vLLM/SGLang (Marlin kernels) |
| Weight + activation | **FP8 (E4M3)** W8A8, **INT8 SmoothQuant** W8A8 | weights *and* activations | FP8 dynamic: none; SmoothQuant: yes | FP8 needs Ada/Hopper+ (L4 ✓, H100 ✓; A10G/A100 fall back to weight-only) |
| GGUF k-quants | Q8_0, Q6_K, Q5_K_M, **Q4_K_M**, IQ… | weights (incl. embeddings) | optional **importance matrix** | llama.cpp / Ollama (CPU + GPU) |
| KV-cache | FP8 KV, q8_0/q4_0 KV | the cache, not the weights | none | engine flag |
| Microscaling | **MXFP4**, NVFP4 | 4-bit values + shared 8-bit scale per 32 | — | Blackwell; gpt-oss ships MXFP4 |
| **QAT** | e.g. Gemma 4's official Q4_0 / W4A16 | trained *with* quantization in the loop | training data | same as above |

## How the main methods work
- **GPTQ:** quantize weights column by column, using second-order (Hessian) information from calibration activations to push each rounding error onto the not-yet-quantized columns.
- **AWQ:** a small fraction of weight channels matter most (those that see large activations). Scale them up before quantizing, which protects them, then fold the inverse scale into the previous op.
- **SmoothQuant (W8A8):** activations have **outlier channels** (some 100× larger), so per-tensor int8 activations clip badly. It migrates the difficulty from activations into weights with a per-channel scale.
- **FP8 E4M3:** 4 exponent bits give dynamic range, so outliers are far less painful than with int8. That's why FP8 is usually near-lossless.

## Group size
int4 with one scale per **group of 128 weights** costs 2 B / 128 = **0.016 B/param** extra (memcalc's `w4a16-g128` = 0.516 B/param). Smaller groups improve accuracy but add overhead and slow the kernels.

## Worked example: Qwen3-4B on an L4 (memcalc)
| Format | Weights | Decode ceiling (1 stream) | Max 8K seqs |
|---|---|---|---|
| bf16 | 7.49 GiB | 37 tok/s | 9 |
| fp8 | 4.11 GiB | 68 tok/s | 13 |
| int4 W4A16 | 2.47 GiB | 113 tok/s | 14 |
| bf16 + **FP8 KV** | 7.49 GiB | 37 tok/s | **19** |

For long-context capacity, **quantizing the KV cache beats quantizing the weights**, because KV is the larger consumer at 8K. That's the hypothesis for E7.

## Where quantization hurts
Commonly reported: multi-step reasoning (GSM8K), instruction-following (IFEval) and long-context retrieval degrade first at 4-bit, and small models are more sensitive than large ones. E3 measures this on Qwen3-4B, and compares **QAT vs PTQ** using Gemma 4's official QAT checkpoints against our own PTQ at the same bit width.

## Interview questions this answers
1. GPTQ vs AWQ: how does each decide what to protect?
2. Why is activation quantization harder than weight quantization?
3. Why is FP8 usually near-lossless while int8 W8A8 needs SmoothQuant?
4. When should you quantize the KV cache instead of the weights?
5. QAT vs PTQ: when is QAT worth it?
