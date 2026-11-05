# Cortex

A self-hosted **personal AI brain**, built as four projects that form one stack:

| Package | Project | Status |
|---|---|---|
| [`serve/`](serve/) | **Cortex Serve**: LLM inference memory math, benchmarking, quantization and production serving | 🚧 M0 done: `memcalc` |
| `slm/` | Cortex-SLM: a small language model from tokenizer to RL to distillation | planned |
| `retrieval/` | Cortex Retrieval: hybrid / graph / multimodal RAG | planned |
| `agent/` | Cortex Agent: autonomous Goal→Plan→Act→Evaluate agent with GPA scoring | planned |

## Quickstart
```bash
make setup          # uv sync + git hooks (requires uv; Python 3.12 is installed by uv)
make check          # lint, strict type-check, tests
make demo           # compare KV-cache memory of three ~4B architectures on an L4
uv run memcalc estimate -m Qwen/Qwen3-4B-Instruct-2507 -d L4 --kv-dtype fp8
```

License: Apache-2.0
