"""Run a servebench config on a Modal GPU, engine and load client in the same container (G-05).

    uv run --extra modal modal run serve/deploy/modal/bench_app.py --config serve/configs/gpu/smoke_l4.yaml
    uv run --extra modal modal run serve/deploy/modal/bench_app.py --config ... --yes      # actually spend

Cost safety:
* every config must declare ``budget: {gpu, usd_per_hour, max_minutes}``; the worst case is printed and
  nothing runs without ``--yes``;
* the Modal function timeout is ``max_minutes``, so spend is hard-capped even if something hangs;
* before launching, the metered spend (``modal billing summary``) + this run's worst case must stay under the
  approved stage gate and the $245 credit ceiling (D-030); runs are launched one at a time so that holds;
* model weights are cached in the ``cortex-hf-cache`` volume (downloaded once).
"""

from __future__ import annotations

import io
import json
import os
import subprocess
import tarfile
from pathlib import Path

import modal
import yaml

# Pinned engine images (docs/research/M2_*). Configs must use exactly these refs.
ENGINE_IMAGES = {
    "vllm": "vllm/vllm-openai:v0.32.0@sha256:befccf0e7e4089aa23926bb6f887a85c0d46c2a99e6de5c5ffdf5cb7957c5b34",
    "sglang": "lmsysorg/sglang:v0.5.22@sha256:e3a4f06001a29542fc4a8638536d9ffc5eb9c9942b08603c09b19a56eae4a1ae",
    "llamacpp": "ghcr.io/ggml-org/llama.cpp:server-cuda-b12110@sha256:a2963b4817f03eea7e0d6065ea576252322792441ef97aaae178b924520547d6",  # noqa: E501
}
PY_DEPS = [
    "pydantic>=2.10",
    "typer>=0.15",
    "rich>=13",
    "pyyaml>=6",
    "httpx>=0.28",
    "tokenizers>=0.21",
    "polars>=1.20",
    "matplotlib>=3.9",
]

app = modal.App("cortex-servebench")
hf_cache = modal.Volume.from_name("cortex-hf-cache", create_if_missing=True)
VOLUMES: dict[str | os.PathLike[str], modal.Volume] = {"/hf": hf_cache}


def _image(kind: str) -> modal.Image:
    ref = ENGINE_IMAGES[kind]
    image = modal.Image.from_registry(ref, add_python="3.12") if kind == "llamacpp" else modal.Image.from_registry(ref)
    return (
        image.entrypoint([])
        .uv_pip_install(*PY_DEPS)
        .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
        .add_local_python_source("cortex_serve")
    )


def _bench(config_yaml: str, only: list[str] | None, git_sha: str, git_dirty: bool) -> bytes:
    """Runs inside the GPU container: start engine, run the config, return the run dir as tar.gz."""
    os.environ["SERVEBENCH_GIT_SHA"] = git_sha
    os.environ["SERVEBENCH_GIT_DIRTY"] = "1" if git_dirty else "0"
    from cortex_serve.servebench.config import RunConfig
    from cortex_serve.servebench.runner import run

    cfg = RunConfig.model_validate(yaml.safe_load(config_yaml))
    out = Path("/tmp/results")
    try:
        run_dir = run(cfg, out, models_dir=Path("/hf/gguf"), repo_dir=Path("/"), only=only)
    finally:
        hf_cache.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        tar.add(run_dir, arcname=str(run_dir.relative_to(out)))
    return buf.getvalue()


# One function per engine image; GPU type and timeout are set per call with .with_options().
@app.function(image=_image("vllm"), gpu="L4", volumes=VOLUMES, timeout=600)
def bench_vllm(config_yaml: str, only: list[str] | None, git_sha: str, git_dirty: bool) -> bytes:
    return _bench(config_yaml, only, git_sha, git_dirty)


@app.function(image=_image("sglang"), gpu="L4", volumes=VOLUMES, timeout=600)
def bench_sglang(config_yaml: str, only: list[str] | None, git_sha: str, git_dirty: bool) -> bytes:
    return _bench(config_yaml, only, git_sha, git_dirty)


@app.function(image=_image("llamacpp"), gpu="L4", volumes=VOLUMES, timeout=600)
def bench_llamacpp(config_yaml: str, only: list[str] | None, git_sha: str, git_dirty: bool) -> bytes:
    return _bench(config_yaml, only, git_sha, git_dirty)


FUNCTIONS = {"vllm": bench_vllm, "sglang": bench_sglang, "llamacpp": bench_llamacpp}


@app.local_entrypoint()
def main(config: str, yes: bool = False, only: str = "", approved_gate: float = 60.0) -> None:
    from cortex_serve.servebench.config import RunConfig, budget_check

    text = Path(config).read_text()
    cfg = RunConfig.model_validate(yaml.safe_load(text))
    if cfg.budget is None:
        raise SystemExit("config has no `budget` section: refusing to run on a paid GPU")
    if cfg.engine.launch != "process" or cfg.engine.image != ENGINE_IMAGES[cfg.engine.kind]:
        raise SystemExit(f"engine must be launch: process with image {ENGINE_IMAGES[cfg.engine.kind]}")
    b = cfg.budget
    print(
        f"[cost] {cfg.name}: {b.gpu} at ${b.usd_per_hour:.2f}/h, hard cap {b.max_minutes:.0f} min "
        f"-> worst case ${b.max_usd:.2f}"
    )
    summary = subprocess.run(["modal", "billing", "summary", "--json"], capture_output=True, text=True, check=True)
    spent = float(json.loads(summary.stdout)["metered_cost"])
    print(f"[cost] metered spend this month so far: ${spent:.2f} (approved gate ${approved_gate:.0f})")
    if reason := budget_check(spent, b.max_usd, approved_gate):
        raise SystemExit(f"[cost] refusing: {reason}")
    if not yes:
        raise SystemExit("[cost] dry run only. Re-run with --yes to start the GPU job.")
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=False
        ).stdout.strip()
    )
    fn = FUNCTIONS[cfg.engine.kind].with_options(gpu=b.gpu, timeout=int(b.max_minutes * 60))
    data = fn.remote(text, only.split(",") if only else None, sha, dirty)
    dest = Path("serve/results")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
    print(f"[done] results extracted under {dest}/{cfg.name}/")
