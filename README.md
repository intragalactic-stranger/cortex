# Cortex

A self-hosted **personal AI brain**, built as four projects that form one stack. Each project is a
reproducible lab: every number in it comes from a committed config, a pinned image digest and a run manifest.

| Package | Project | Status |
|---|---|---|
| [`serve/`](serve/) | **Cortex Serve**: memory math, benchmarking, quantization and production serving for LLM inference | ✅ M0–M5 done (memcalc, servebench, GPU engine baseline, quantization, feature experiments, k3d serving) · M6 web site and M7 decision guide next |
| `slm/` | Cortex-SLM: a small language model from tokenizer to RL to distillation | planned |
| `retrieval/` | Cortex Retrieval: hybrid / graph / multimodal RAG | planned |
| `agent/` | Cortex Agent: autonomous Goal → Plan → Act → Evaluate agent | planned |

## Cortex Serve in one screen

> **Question:** for a given model, latency target and budget, which engine + quantization + configuration
> delivers the most goodput per dollar without unacceptable quality loss?

Measured on Modal (L4, H100) and a 4-vCPU box, ~60 GPU runs, with every request timed client-side:

1. **A cheap GPU can mean expensive answers.** Qwen3-4B on an L4 meets the latency SLO for ~4 chat users, an H100 for ~64. Per SLO-meeting request the H100 is **~12× cheaper** at 4.9× the hourly price.
2. **vLLM and SGLang tie on equal hardware** (±5% L4, ±2% H100); llama.cpp is 25–33% behind at 16 users. (A silent H100 → H200 swap first made vLLM look 20% faster; every run now records the GPU it actually got.)
3. **The best quantization depends on the GPU.** 4-bit is 2.7× faster per token on an L4 at every load, but *slower* than bf16 on a busy H100, where FP8/INT8 add ~30% capacity.
4. **8-bit costs no measurable quality; 4-bit costs a little** (perplexity +5%, knowledge −2 points). Careful PTQ (GPTQ) matched Google's QAT Gemma 4 on chat tasks; naive rounding lost 13.6 points on GSM8K.
5. **Hybrid and sliding-window architectures need 3–4× less KV memory per sequence** and serve long contexts 1.7× faster, but engines realise only 48–94% of the theoretical saving.
6. **Production patterns, measured:** queue-depth autoscaling, zero-failure rolling updates (127/127), and gateway fallback to a GPU endpoint (214/252 requests served by the GPU while the CPU pool was down).

→ Details, charts and reproduction commands: [`serve/README.md`](serve/README.md) · all tables: [`serve/RESULTS.md`](serve/RESULTS.md)

## Quickstart
```bash
make setup          # uv sync + git hooks (requires uv; Python 3.12 is installed by uv)
make check          # ruff, mypy --strict, pytest
make demo           # KV-cache memory of three ~4B architectures on an L4
uv run memcalc estimate -m Qwen/Qwen3-4B-Instruct-2507 -d L4 --kv-dtype fp8
```

## Repository conventions
uv workspace, Python 3.12, `ruff` + `mypy --strict` + `pytest` (`make check` in CI), gitleaks, images pinned by
digest, model revisions pinned by commit SHA, architecture decisions in `serve/docs/adr/`.

License: Apache-2.0
