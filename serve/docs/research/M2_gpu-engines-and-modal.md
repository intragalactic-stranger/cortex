# M2 research: GPU engines, images and the Modal runner (2026-11-05)

## Pinned images (stable releases ≥ ~2 weeks old, digests from the registries)
| Engine | Image | Released | Why this version |
|---|---|---|---|
| vLLM | `vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34` | 2026-10-18 | Latest non-rc release. GitHub's "latest" points at `v0.33.0rc3`, which vLLM doesn't mark as a prerelease, so filter by tag name. |
| SGLang | `lmsysorg/sglang:v0.5.22@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae` | 2026-10-19 | v0.5.23 is 3 days old; prefer a release with two weeks of field use for a reproducible baseline |
| llama.cpp (CUDA) | `ghcr.io/ggml-org/llama.cpp:server-cuda-b12110@sha256:a2963b4817f03eea7e0d6065ea576252322792441ef97aaae178b924520547d6` | build 12110 | **Same build as the CPU tier**, so CPU vs GPU differences aren't engine-version differences |

**To verify inside each image at the first run (never assume flags):** `vllm serve --help`, `python -m sglang.launch_server --help`; the exact startup log lines that `parse_vllm_memory` / `parse_sglang_memory` match ("GPU KV cache size: N tokens", "Maximum concurrency for N tokens per request: Xx", "max_total_num_tokens="); the vLLM waiting-requests metric name (for M5 KEDA).

## Modal runner (`serve/deploy/modal/bench_app.py`)
- One Modal function per engine image; GPU and timeout set per call (`.with_options(gpu=..., timeout=...)`).
- The **engine runs as a subprocess in the same container as the load client** (G-05): TTFT excludes internet latency, so results compare with other published single-node numbers.
- **Cost gate:** configs must carry `budget: {gpu, usd_per_hour, max_minutes}`. The launcher prints the worst case and needs `--yes`. The function timeout equals `max_minutes`, so spend is hard-capped.
- Weights cached in the `cortex-hf-cache` Modal volume (`HF_HOME=/hf`), downloaded once.
- Results come back as a tar of the run directory and land in `serve/results/` locally. The git SHA and dirty flag are passed into the container for the manifest.
- GPU telemetry: `nvidia-smi --query-gpu=utilization.gpu,memory.used,power.draw -lms 1000` → mean utilisation, peak memory, mean power per point.

## Request semantics per engine
| | llama.cpp | vLLM | SGLang | Ollama |
|---|---|---|---|---|
| Fixed output length | server flag `--ignore-eos` | request `ignore_eos: true` | request `ignore_eos: true` | not supported (recorded) |
| Disable thinking | `chat_template_kwargs.enable_thinking=false` | same | same | `think: false` |
| KV capacity source | `llama_kv_cache: size` log | "GPU KV cache size" log | "max_total_num_tokens" log | — |
