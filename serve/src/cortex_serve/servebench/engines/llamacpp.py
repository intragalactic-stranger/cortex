"""llama.cpp ``llama-server`` launcher.

Flags verified against ``llama-server --help`` in the pinned image (see docs/research/M1_*.md).
"""

from __future__ import annotations

import re
from typing import Any

from cortex_serve.servebench.config import ArtifactSpec
from cortex_serve.servebench.engines.base import Engine, artifact_path, docker

_FLAG_MAP = {
    "threads": "-t",
    "parallel": "-np",
    "ctx_per_slot": "--kv-unified-per-slot",
    "ctx_size": "-c",
    "batch_size": "-b",
    "ubatch_size": "-ub",
    "cache_type_k": "-ctk",
    "cache_type_v": "-ctv",
    "flash_attn": "-fa",
    "draft_model": "-md",
    "spec_type": "--spec-type",
    "spec_draft_n_max": "--spec-draft-n-max",
    "cache_reuse": "--cache-reuse",
}
_BOOL_FLAGS = {
    "ignore_eos": "--ignore-eos",
    "no_mmap": "--no-mmap",
    "mlock": "--mlock",
    "swa_full": "--swa-full",
}


def server_args(args: dict[str, Any], model_path: str) -> list[str]:
    if "draft_model" in args and "spec_type" not in args:
        # Current llama-server defaults --spec-type to "none": -md alone loads the draft but never drafts.
        raise ValueError("draft_model requires spec_type (e.g. 'draft-simple'); llama.cpp's default is 'none'")
    out = ["-m", f"/models/{model_path}", "--host", "0.0.0.0", "--port", "8080", "--metrics"]
    for key, value in args.items():
        if key in _FLAG_MAP:
            if key == "draft_model":
                value = f"/models/{value}"
            out += [_FLAG_MAP[key], str(value)]
        elif key in _BOOL_FLAGS:
            if value:
                out.append(_BOOL_FLAGS[key])
        elif key == "extra":
            out += [str(v) for v in value]
        else:
            raise ValueError(f"unknown llama.cpp arg {key!r}")
    return out


class LlamaCppEngine(Engine):
    protocol = "openai"
    container_port = 8080
    health_path = "/health"

    def run_args(self, artifact: ArtifactSpec) -> list[str]:
        return [
            "-v",
            f"{self.models_dir}:/models:ro",
            self.image,
            *server_args(self.args, artifact_path(artifact)),
        ]

    def version(self) -> str | None:
        try:
            out = docker("run", "--rm", self.image, "--version", timeout_s=60)
        except RuntimeError as exc:
            out = str(exc)
        m = re.search(r"version: (\S+) \(build (\d+), commit (\w+)\)", out)
        return f"{m.group(1)} build {m.group(2)} commit {m.group(3)}" if m else None


def parse_memory(log: str) -> dict[str, float]:
    """Extract model and KV sizes from a verbose (``-v``) llama-server log.

    llama-server loads twice when fitting parameters (a probe pass, then the real load), so only the
    text after the last ``print_info: file size`` line is used. Models with sliding-window attention
    print one KV-cache line per cache (full + SWA); they are summed.
    """
    out: dict[str, float] = {}
    marker = log.rfind("print_info: file size")
    if marker < 0:
        return out
    tail = log[marker:]
    if m := re.search(r"file size\s*=\s*([\d.]+) (MiB|GiB) \(([\d.]+) BPW\)", tail):
        out["model_file_mib"] = float(m.group(1)) * (1024 if m.group(2) == "GiB" else 1)
        out["model_bpw"] = float(m.group(3))
    kv = re.findall(r"llama_kv_cache\w*: size =\s*([\d.]+) MiB \(\s*(\d+) cells", tail)
    if kv:
        out["kv_mib"] = sum(float(size) for size, _ in kv)
        out["kv_caches"] = float(len(kv))
    if m := re.search(r"CPU compute buffer size =\s*([\d.]+) MiB", tail):
        out["compute_buffer_mib"] = float(m.group(1))
    return out
