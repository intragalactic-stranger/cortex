"""CPU-only pre-flight: confirm every CLI flag servebench passes exists in the pinned engine images.

    uv run --extra modal modal run serve/deploy/modal/check_flags.py

Runs each engine's ``--help`` on a CPU container (no GPU billed) so flag drift between engine releases fails
for fractions of a cent instead of after a paid GPU cold start.
"""

from __future__ import annotations

import subprocess

import modal
from bench_app import _image  # type: ignore[import-not-found]  # same (cached) images as the benchmark

# Flags servebench emits (engines/process.py, engines/llamacpp.py) or that M2 configs use.
# llama.cpp --no-mmap/--mlock are CPU-only and absent from the b12110 CUDA help, so not listed.
FLAGS = {
    "vllm": [
        "--revision", "--port", "--host", "--served-model-name", "--max-model-len", "--gpu-memory-utilization",
        "--max-num-seqs", "--enable-prefix-caching", "--no-enable-prefix-caching", "--enable-chunked-prefill",
        "--enforce-eager", "--max-num-batched-tokens", "--kv-cache-dtype", "--quantization",
    ],
    "sglang": [
        "--model-path", "--revision", "--port", "--host", "--served-model-name", "--enable-metrics",
        "--mem-fraction-static", "--context-length", "--max-running-requests", "--disable-radix-cache",
        "--disable-cuda-graph", "--chunked-prefill-size", "--enable-mixed-chunk", "--kv-cache-dtype",
    ],
    "llamacpp": [
        "--port", "--host", "-t", "-np", "-c", "-b", "-ub", "-ctk", "-ctv", "-fa", "-md", "--spec-type",
        "--spec-draft-n-max", "--cache-reuse", "--ignore-eos", "--swa-full", "-ngl",
        "--metrics",
    ],
}  # fmt: skip
HELP = {
    "sglang": ["python3", "-m", "sglang.launch_server", "--help"],
    "llamacpp": ["/app/llama-server", "--help"],
}

app = modal.App("cortex-flagcheck")


def _check_vllm_source() -> list[str]:
    """vLLM's CLI parser needs CUDA, so on CPU check that each flag's config field exists in the source."""
    import pathlib
    import re

    root = pathlib.Path(subprocess.run(
        ["python3", "-c", "import importlib.util as u; print(u.find_spec('vllm').submodule_search_locations[0])"],
        capture_output=True, text=True, check=True,
    ).stdout.strip())  # fmt: skip
    src = "\n".join(p.read_text(errors="ignore") for p in root.rglob("*.py") if "tests" not in p.parts)
    missing = []
    for flag in FLAGS["vllm"]:
        field = flag[2:].replace("-", "_").removeprefix("no_")
        if not re.search(rf"^\s+{field}\s*:", src, re.M):
            missing.append(flag)
    return missing


def _check(kind: str) -> list[str]:
    if kind == "vllm":
        return _check_vllm_source()
    out = subprocess.run(HELP[kind], capture_output=True, text=True, check=False)
    text = out.stdout + out.stderr
    tokens = set(text.replace(",", " ").replace("=", " ").split())
    missing = [f for f in FLAGS[kind] if f not in tokens]
    if not text.strip():
        return [f"<no help output, exit {out.returncode}>"]
    return missing + ([f"\n--- help head (exit {out.returncode}) ---\n{text[:1500]}"] if missing else [])


@app.function(image=_image("vllm").add_local_python_source("bench_app"), timeout=120, retries=0)
def check_vllm() -> list[str]:
    return _check("vllm")


@app.function(image=_image("sglang").add_local_python_source("bench_app"), timeout=120, retries=0)
def check_sglang() -> list[str]:
    return _check("sglang")


@app.function(image=_image("llamacpp").add_local_python_source("bench_app"), timeout=120, retries=0)
def check_llamacpp() -> list[str]:
    return _check("llamacpp")


@app.local_entrypoint()
def main() -> None:
    bad = False
    for kind, fn in (("vllm", check_vllm), ("sglang", check_sglang), ("llamacpp", check_llamacpp)):
        missing = fn.remote()
        print(f"[flags] {kind}: {'OK' if not missing else 'MISSING ' + ' '.join(missing)}")
        bad |= bool(missing)
    if bad:
        raise SystemExit(1)
