# 0006 · CPU tier: artifacts, threads, SLO

- **Status:** Accepted · **Date:** 2026-11-05 · **Related:** G-04, D-014, E13

## Context
The local box is a 2-core / 4-thread AVX2 EPYC with 15 GiB RAM and (after downloads) ~8 GB free disk.

## Decisions
- **Artifacts from one producer per family** (unsloth for Qwen3 and Gemma PTQ) so the quantization recipe is constant; **Google's official QAT Q4_0** for the QAT-vs-PTQ comparison. Q8_0 serves as the near-lossless reference (BF16 GGUFs exceed the disk budget).
- **4 threads** (measured 1.5–1.6× faster than 2 for both prefill and decode).
- **SLO for the CPU tier:** TTFT ≤ 3 s, TPOT ≤ 125 ms. Rationale: ≥ 8 tok/s outpaces reading speed; a 3-second wait is the edge of acceptable for an interactive local assistant.
- **Grid:** `chat-short` (128/64), concurrency 1 and 4, 8 requests × 3 repeats; `llama-bench` pp128/pp512/tg64 × 3 for engine-internal numbers.

## Consequences
+ The whole CPU tier (8 artifacts, 2 engines, quality, speculative-decoding preview) fits in one ~5-hour unattended run.
− The QAT-vs-PTQ comparison is confounded by PLE-table precision (Q6_K vs Q5_K); stated wherever the result is used.
− No BF16 reference on CPU; quality deltas are relative to Q8_0.
