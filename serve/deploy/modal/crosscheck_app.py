"""M2 cross-check: vLLM's own ``vllm bench serve`` against the same server config as E1 L4 (chat-balanced 512/256).

    uv run --extra modal modal run serve/deploy/modal/crosscheck_app.py --yes

If servebench's client-side TTFT/TPOT agree with the engine vendor's benchmark at the same load, the harness is not
adding its own overhead. Same pinned image, model revision, server args and GPU as ``configs/gpu/e1_l4_vllm.yaml``.
"""

from __future__ import annotations

import io
import json
import subprocess
import tarfile
import time
from pathlib import Path

import modal

VLLM_IMAGE = "vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34"
MODEL, REV = "Qwen/Qwen3-4B-Instruct-2507", "cdbee75f17c01a7cc42f958dc650907174af0554"
LEVELS = (1, 16)
L4_USD_PER_HOUR = 0.80
MAX_MINUTES = 30

app = modal.App("cortex-crosscheck")
hf_cache = modal.Volume.from_name("cortex-hf-cache", create_if_missing=True)
image = (
    modal.Image.from_registry(VLLM_IMAGE)
    .entrypoint([])
    .run_commands('command -v python || ln -s "$(command -v python3)" /usr/local/bin/python')
    .uv_pip_install("pydantic>=2.10", "pyyaml>=6", "httpx>=0.28")
    .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
    .add_local_python_source("cortex_serve")
)


@app.function(image=image, gpu="L4", volumes={"/hf": hf_cache}, timeout=MAX_MINUTES * 60)
def crosscheck(git: dict[str, object]) -> bytes:
    from cortex_serve.servebench.config import ArtifactSpec
    from cortex_serve.servebench.engines.process import ProcessEngine

    out = Path("/tmp/crosscheck")
    out.mkdir(parents=True, exist_ok=True)
    art = ArtifactSpec(name="qwen3-4b-bf16", repo=MODEL, revision=REV, family=MODEL, quant="bf16")
    eng = ProcessEngine("vllm", VLLM_IMAGE, {"max_model_len": 4096, "gpu_memory_utilization": 0.9}, "/hf", port=8000)
    eng.start(art)
    results = {}
    try:
        for c in LEVELS:
            n = max(16, 3 * c)
            cmd = ["vllm", "bench", "serve", "--backend", "openai", "--base-url", "http://127.0.0.1:8000",
                   "--endpoint", "/v1/completions", "--model", "model", "--tokenizer", MODEL,
                   "--dataset-name", "random", "--random-input-len", "512", "--random-output-len", "256",
                   "--num-prompts", str(n), "--max-concurrency", str(c), "--ignore-eos", "--seed", "1234",
                   "--percentile-metrics", "ttft,tpot,itl,e2el", "--metric-percentiles", "50,95",
                   "--save-result", "--result-dir", str(out), "--result-filename", f"vllm-bench-c{c}.json"]  # fmt: skip
            p = subprocess.run(cmd, capture_output=True, text=True, check=False)
            (out / f"vllm-bench-c{c}.log").write_text(p.stdout + p.stderr)
            if p.returncode:
                help_text = subprocess.run(["vllm", "bench", "serve", "--help"], capture_output=True, text=True)
                (out / "vllm-bench-help.txt").write_text(help_text.stdout + help_text.stderr)
                results[c] = {"error": (p.stdout + p.stderr)[-1500:]}
                continue
            results[c] = json.loads((out / f"vllm-bench-c{c}.json").read_text())
    finally:
        (out / "engine.log").write_text(eng.logs())
        eng.stop()
    (out / "summary.json").write_text(json.dumps({"git": git, "levels": list(LEVELS), "results": results}, indent=2))
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(out, arcname="crosscheck-vllm-bench")
    return buf.getvalue()


@app.local_entrypoint()
def main(yes: bool = False, approved_gate: float = 60.0) -> None:
    from cortex_serve.spend import guarded

    worst = L4_USD_PER_HOUR * MAX_MINUTES / 60
    print(f"[cost] crosscheck: L4, hard cap {MAX_MINUTES} min -> worst case ${worst:.2f}", flush=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    with guarded("crosscheck vllm bench", worst, approved_gate, yes):
        data = crosscheck.remote({"sha": sha})
    dest = Path("serve/results/m2-crosscheck") / time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    dest.mkdir(parents=True, exist_ok=True)
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    print(f"[done] crosscheck -> {dest}")
