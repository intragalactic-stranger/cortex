# 08 · The model lifecycle (overview; training details live in Project 4)

```
pretraining → mid-training → SFT → preference optimisation / RL → distillation → quantization → serving
   (P4)          (P4)        (P4)          (P4)                     (P4)         (P2 + P4)      (P2)
```

| Stage | What happens | Why a serving engineer cares |
|---|---|---|
| **Pretraining** | Next-token prediction on trillions of tokens | Fixes the architecture: layers, heads, KV design, vocab. That determines all of doc 02. |
| **Mid-training** | Higher-quality/long-context data, context extension (RoPE scaling) | Sets usable context length and long-context quality |
| **SFT** | Supervised instruction/chat data with a **chat template** | The template you *must* serve with (doc 06) |
| **Preference optimisation / RL** | DPO, PPO, GRPO/RLVR on preferences or verifiable rewards | Reasoning ("thinking") modes, which change output lengths, so they change cost and latency |
| **Distillation** | A small student learns from a large teacher (logits, sequences, or on-policy) | Produces the small models we serve (Qwen3-0.6B/1.7B) and draft models for speculative decoding |
| **Quantization** | PTQ (GPTQ, AWQ, FP8, GGUF) or QAT | Bytes per weight, so speed and capacity (doc 05) |
| **Serving** | Engines, batching, caching, scaling | This project |

## Connections inside Cortex
- **P4 (Cortex-SLM)** trains a small model from scratch and post-trains a Qwen3-family base into the **Cortex Router**.
- **P2 serves it:** P2 quantizes that model (M3 pipeline), serves it with multi-LoRA (E11), and puts it behind the gateway as `cortex-router` (M5).
- **Thinking vs non-thinking:** we serve **Qwen3-4B-Instruct-2507** (non-thinking) for benchmarks, so output lengths reflect the requested `max_tokens` rather than hidden reasoning tokens. A reasoning model's real cost per answer can be 5–20× higher. Benchmark both if the product needs reasoning.

## Interview questions this answers
1. Walk through the lifecycle of an open model from pretraining to production.
2. How does post-training (e.g. RL for reasoning) change serving cost?
3. Why are distilled small models central to speculative decoding?
