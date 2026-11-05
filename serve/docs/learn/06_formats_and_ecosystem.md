# 06 · Formats and the model ecosystem

## Weight file formats
| Format | What it is | Safety | Used by |
|---|---|---|---|
| **pickle** (`.bin`, `.pt`) | Python object serialization | ⚠️ **Loading can execute arbitrary code** | legacy PyTorch |
| **safetensors** | 8-byte header length + JSON header (name, dtype, shape, offsets) + raw tensor bytes | Safe: data only, memory-mappable | HF, vLLM, SGLang |
| **GGUF** | Single file: metadata (architecture, tokenizer, chat template) + quantized tensors | Data only | llama.cpp, Ollama, LM Studio |

**memcalc uses the safetensors layout directly.** Two HTTP range requests (`bytes=0-7`, then the header) give every tensor's shape, so the parameter count is exact without downloading gigabytes. That's how it found Qwen3-0.6B's redundant `lm_head` (156M params stored but never loaded).

## A Hugging Face repo, file by file
| File | Why it matters for serving |
|---|---|
| `config.json` | Architecture: everything memcalc needs (doc 01) |
| `model.safetensors` / `model-0000x-of-0000y.safetensors` + `model.safetensors.index.json` | Weights, sharded |
| `tokenizer.json`, `tokenizer_config.json` | Tokenizer + **chat template** (Jinja) |
| `generation_config.json` | Default sampling (temperature, top-p, EOS ids). Engines may apply these silently. |

**Always pin a commit SHA.** Our fixtures are pinned (e.g. Qwen3-4B-Instruct-2507 @ `cdbee75f`). A `main` branch can change under you.

## Chat templates: the silent quality killer
Instruct models are trained on a specific turn format (`<|im_start|>user …<|im_end|>`). A wrong or missing template, an extra BOS token, or the wrong thinking/non-thinking mode **still produces fluent text, but measurably worse answers**. Benchmarks then compare engines and silently blame the wrong component. servebench will use each engine's `/v1/chat/completions` with the model's own template, and log the rendered prompt for one request per run.

## Licenses (check before publishing anything)
| Model family | License | Implication |
|---|---|---|
| Qwen3, Qwen3.5 | Apache-2.0 | Derivatives can be published freely |
| **Gemma 4** | **Apache-2.0** (changed from Gemma 3's custom terms) | Same |
| Llama 3.x | Llama Community License, gated | Attribution and naming rules; we avoid it |
| LFM2 | custom "LFM Open License" | Read before redistributing; rejected for our artifacts |

Our quantized artifacts (M3) inherit Apache-2.0.

## Interview questions this answers
1. Why is loading a pickle checkpoint a security risk, and what does safetensors change?
2. How would you get a model's exact parameter count without downloading it?
3. What can a wrong chat template do to an evaluation, and how would you detect it?
4. What license checks do you run before publishing a quantized model?
