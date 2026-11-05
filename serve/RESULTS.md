# Results

Every number here must be reproducible from a committed config, with a manifest. Measured results start in M1.

## M0: memcalc predictions (theoretical, not yet measured)

| Model | Device | Weights | Memory / 8K seq | Max seqs @8K | Source |
|---|---|---|---|---|---|
| Qwen3-4B-Instruct-2507 | L4 | 7.49 GiB | 1,152 MiB | 9 | `make demo` |
| Qwen3.5-4B | L4 | 7.83 GiB | 305 MiB | 36 | `make demo` |
| Gemma 4 E4B | L4 | 14.00 GiB (8.75 with PLE offload) | 148 MiB | 32 (69 with PLE offload) | `memcalc estimate` |
| Qwen3-30B-A3B | H100-80GB | 56.87 GiB | 768 MiB | 17 | `memcalc estimate` |

These become hypotheses for M1 (llama.cpp), M2 (vLLM capacity at startup) and E15.
