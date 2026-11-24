"""Run a servebench config on a Modal GPU, engine and load client in the same container (G-05).

    uv run --extra modal modal run serve/deploy/modal/bench_app.py --config serve/configs/gpu/smoke_l4.yaml
    uv run --extra modal modal run serve/deploy/modal/bench_app.py --config ... --yes      # actually spend

Cost safety:
* every config must declare ``budget: {gpu, usd_per_hour, max_minutes}``; the worst case is printed and
  nothing runs without ``--yes``;
* the Modal function timeout is ``max_minutes``, so spend is hard-capped even if something hangs;
* before launching, the metered spend (``modal billing summary``) + this run's worst case must stay under the
  approved stage gate and the $245 credit ceiling (D-030), and nothing may be billed to the card
  (``billed_cost`` must be $0); runs are launched one at a time so that holds;
* model weights are cached in the ``cortex-hf-cache`` volume (downloaded once).
"""

from __future__ import annotations

import io
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
    "huggingface_hub>=0.26",
]

app = modal.App("cortex-servebench")
hf_cache = modal.Volume.from_name("cortex-hf-cache", create_if_missing=True)
VOLUMES: dict[str | os.PathLike[str], modal.Volume] = {"/hf": hf_cache}


def _image(kind: str) -> modal.Image:
    ref = ENGINE_IMAGES[kind]
    image = modal.Image.from_registry(ref, add_python="3.12") if kind == "llamacpp" else modal.Image.from_registry(ref)
    return (
        image.entrypoint([])
        # the SGLang image ships only `python3`; Modal's uv_pip_install resolves `python`
        .run_commands('command -v python || ln -s "$(command -v python3)" /usr/local/bin/python')
        .uv_pip_install(*PY_DEPS)
        .env({"HF_HOME": "/hf", "HF_HUB_ENABLE_HF_TRANSFER": "0"})
        .add_local_python_source("cortex_serve")
    )


def _fetch_gguf(cfg: object) -> None:
    """GGUF engines read ``/hf/gguf/<org>_<repo>/<file>``: download missing files once into the volume."""
    from cortex_serve.servebench.config import RunConfig
    from cortex_serve.servebench.engines.base import artifact_path
    from huggingface_hub import hf_hub_download

    assert isinstance(cfg, RunConfig)
    for art in cfg.artifacts:
        if art.file is None:
            continue
        dest = Path("/hf/gguf") / artifact_path(art)
        if not dest.exists():
            print(f"[fetch] {art.repo}@{art.revision[:8]}/{art.file}")
            hf_hub_download(art.repo, art.file, revision=art.revision, local_dir=dest.parent)
            hf_cache.commit()


def _bench(config_yamls: list[str], only: list[str] | None, git_sha: str, git_dirty: bool) -> bytes:
    """Runs inside the GPU container: run each config in turn (one cold start), return all run dirs as tar.gz.

    A failing config is logged and skipped so the others still return their data.
    """
    os.environ["SERVEBENCH_GIT_SHA"] = git_sha
    os.environ["SERVEBENCH_GIT_DIRTY"] = "1" if git_dirty else "0"
    from cortex_serve.servebench.config import RunConfig
    from cortex_serve.servebench.runner import run

    out = Path("/tmp/results")
    run_dirs: list[Path] = []
    for text in config_yamls:
        cfg = RunConfig.model_validate(yaml.safe_load(text))
        try:
            _fetch_gguf(cfg)
            run_dirs.append(run(cfg, out, models_dir=Path("/hf/gguf"), repo_dir=Path("/"), only=only))
        except Exception as exc:
            print(f"[fail] {cfg.name}: {exc!r}")
        finally:
            hf_cache.commit()
    buf = io.BytesIO()
    with tarfile.open(fileobj=buf, mode="w:gz") as tar:
        for run_dir in run_dirs:
            tar.add(run_dir, arcname=str(run_dir.relative_to(out)))
    return buf.getvalue()


# One function per engine image; GPU type and timeout are set per call with .with_options().
@app.function(image=_image("vllm"), gpu="L4", volumes=VOLUMES, timeout=600)
def bench_vllm(config_yamls: list[str], only: list[str] | None, git_sha: str, git_dirty: bool) -> bytes:
    return _bench(config_yamls, only, git_sha, git_dirty)


@app.function(image=_image("sglang"), gpu="L4", volumes=VOLUMES, timeout=600)
def bench_sglang(config_yamls: list[str], only: list[str] | None, git_sha: str, git_dirty: bool) -> bytes:
    return _bench(config_yamls, only, git_sha, git_dirty)


@app.function(image=_image("llamacpp"), gpu="L4", volumes=VOLUMES, timeout=600)
def bench_llamacpp(config_yamls: list[str], only: list[str] | None, git_sha: str, git_dirty: bool) -> bytes:
    return _bench(config_yamls, only, git_sha, git_dirty)


FUNCTIONS = {"vllm": bench_vllm, "sglang": bench_sglang, "llamacpp": bench_llamacpp}


@app.local_entrypoint()
def main(config: str, yes: bool = False, only: str = "", approved_gate: float = 60.0) -> None:
    """``--config`` takes one path or several comma-separated paths for the same engine image and GPU; they run
    back to back in one container. The worst case is the sum of their budgets."""
    from cortex_serve.servebench.config import RunConfig
    from cortex_serve.spend import guarded

    paths = [Path(c) for c in config.split(",")]
    texts = [p.read_text() for p in paths]
    cfgs = [RunConfig.model_validate(yaml.safe_load(t)) for t in texts]
    for cfg in cfgs:
        if cfg.budget is None:
            raise SystemExit(f"{cfg.name}: no `budget` section: refusing to run on a paid GPU")
        if cfg.engine.launch != "process" or cfg.engine.image != ENGINE_IMAGES[cfg.engine.kind]:
            raise SystemExit(f"{cfg.name}: engine must be launch: process with image {ENGINE_IMAGES[cfg.engine.kind]}")
    budgets = [c.budget for c in cfgs if c.budget is not None]
    if len({c.engine.kind for c in cfgs}) != 1 or len({(b.gpu, b.usd_per_hour) for b in budgets}) != 1:
        raise SystemExit("all configs in one call must share the engine kind, GPU and rate")
    gpu, rate = budgets[0].gpu, budgets[0].usd_per_hour
    minutes = sum(b.max_minutes for b in budgets)
    worst = rate * minutes / 60
    names = ", ".join(c.name for c in cfgs)
    print(f"[cost] {names}: {gpu} at ${rate:.2f}/h, hard cap {minutes:.0f} min -> worst case ${worst:.2f}", flush=True)
    sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True, text=True, check=False).stdout.strip()
    dirty = bool(
        subprocess.run(
            ["git", "status", "--porcelain", "--untracked-files=no"], capture_output=True, text=True, check=False
        ).stdout.strip()
    )
    fn = FUNCTIONS[cfgs[0].engine.kind].with_options(gpu=gpu, timeout=int(minutes * 60))
    with guarded(names, worst, approved_gate, yes):
        data = fn.remote(texts, only.split(",") if only else None, sha, dirty)
    dest = Path("serve/results")
    with tarfile.open(fileobj=io.BytesIO(data), mode="r:gz") as tar:
        tar.extractall(dest, filter="data")
        got = sorted({m.name.split("/")[0] for m in tar.getmembers()})
    print(f"[done] results extracted under {dest}/: {', '.join(got) or 'NOTHING'}")
    if missing := [c.name for c in cfgs if c.name not in got]:
        print(f"[fail] no results for: {', '.join(missing)}")
