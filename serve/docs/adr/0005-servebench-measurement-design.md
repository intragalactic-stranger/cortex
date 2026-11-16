# 0005 · servebench measurement design

- **Status:** Accepted · **Date:** 2026-11-05 · **Related:** spec §8, M1

## Context
Engine comparisons are easy to get wrong: engine-reported timings differ in definition, token counts come from different tokenizers, output lengths vary, prompts can hit prefix caches, and a single run hides variance.

## Decision
1. **Client-side timing from the stream.** TTFT = first chunk carrying generated text (content *or* reasoning); ITL = gaps between later text chunks; TPOT = (E2E − TTFT) / (tokens − 1). Parsing is separated from I/O with an injectable clock and unit-tested with a fake clock. Engine-side timings (`timings`, Ollama `eval_duration`) are recorded only as a cross-check.
2. **One ruler:** output tokens are re-counted with the model's own HF tokenizer (pinned revision); engine counts are logged alongside.
3. **Fixed output length:** `--ignore-eos` + `max_tokens` where supported. Engines that can't (Ollama) are documented, and per-token metrics are compared rather than E2E.
4. **No accidental cache hits:** prompts are seeded, unique, exact-length, and the shared instruction comes last. Prefix sharing is only present in the explicit `shared-prefix` workload.
5. **Load:** closed-loop fixed concurrency (users wait for their previous answer) and open-loop Poisson arrivals (for goodput-vs-rate sweeps in M2). Warmup requests are discarded.
6. **Goodput (DistServe definition):** a request counts if *its own* TTFT ≤ SLO and *its own* TPOT ≤ SLO. Goodput = counted requests (or their tokens) per second.
7. **Repeats:** 3 per point. Reports show the median and the min–max range.
8. **Manifest per run:** engine image digest + version, artifact revision + sha256, tokenizer revision, host (CPU, RAM, kernel, Docker, GPU), git SHA + dirty flag, full config, UTC time. Results without a manifest are invalid.
9. **Resources from cgroup v2 files** (CPU µs, memory bytes) at 1 Hz, not `docker stats` polling.
10. **Sequential execution:** measurement runs never overlap on the same machine.

## Consequences
+ Numbers are comparable across engines and reproducible from configs. + Timing logic is tested without servers.
− Client-side TTFT includes HTTP and loopback overhead (sub-millisecond locally; on Modal the client is co-located, G-05).
− Synthetic prose is not a real conversation distribution; ShareGPT-style traces are added on GPU (M2).

## Amendment (2026-11-16, M1 results)
- **Point 9 is limited for memory.** llama.cpp `mmap`s weights, and cgroup v2 charges a file page only to the cgroup that first faults it in. Weights already in the host page cache from an earlier container are not counted, so cgroup *peak memory* under-reports (Gemma 4 E2B Q8_0: 657 MiB peak vs 1,847 MiB for the smaller Q4_0). Memory claims use the engine-reported model/KV/compute-buffer sizes (`mem_*` columns) and memcalc; cgroup memory is kept only as a diagnostic. CPU time from cgroup is unaffected.
- **Engine features must be verified as active, not just configured.** The first E6 run loaded a draft model that never drafted (`--spec-type` defaults to `none`). servebench now records per-point draft counts from server `timings` and fails a run whose configured feature shows no effect (commit `75d6f73`).
