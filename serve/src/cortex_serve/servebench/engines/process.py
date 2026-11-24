"""Engines launched as local processes (inside a Modal GPU container built from the engine's image).

Commands are built by pure functions so they can be unit-tested without the engines installed.
Flags must be re-verified against each pinned image's ``--help`` (docs/research/M2_*.md).
"""

from __future__ import annotations

import re
import subprocess
import time
from pathlib import Path
from typing import Any

import httpx

from cortex_serve.servebench.config import ArtifactSpec
from cortex_serve.servebench.engines.base import Engine


def _flags(args: dict[str, Any], bool_flags: set[str]) -> list[str]:
    out: list[str] = []
    for key, value in args.items():
        if key == "extra":
            out += [str(v) for v in value]
            continue
        flag = "--" + key.replace("_", "-")
        if key in bool_flags:
            if value:
                out.append(flag)
        else:
            out += [flag, str(value)]
    return out


VLLM_BOOL = {"enable_prefix_caching", "no_enable_prefix_caching", "enable_chunked_prefill", "enforce_eager"}
SGLANG_BOOL = {"disable_radix_cache", "enable_mixed_chunk", "disable_cuda_graph"}


def vllm_cmd(artifact: ArtifactSpec, args: dict[str, Any], port: int) -> list[str]:
    model = artifact.repo if artifact.file is None else f"/models/{artifact.file}"
    cmd = ["vllm", "serve", model, "--port", str(port), "--host", "127.0.0.1"]
    if not model.startswith("/"):  # local checkpoint dirs (M3 artifacts) have no hub revision
        cmd[3:3] = ["--revision", artifact.revision]
    cmd += ["--served-model-name", "model"]  # request logging is off by default since vLLM 0.1x
    return cmd + _flags(args, VLLM_BOOL)


def sglang_cmd(artifact: ArtifactSpec, args: dict[str, Any], port: int) -> list[str]:
    cmd = ["python3", "-m", "sglang.launch_server", "--model-path", artifact.repo]
    if not artifact.repo.startswith("/"):
        cmd += ["--revision", artifact.revision]
    cmd += ["--port", str(port), "--host", "127.0.0.1", "--served-model-name", "model", "--enable-metrics"]
    return cmd + _flags(args, SGLANG_BOOL)


class ProcessEngine(Engine):
    protocol = "openai"
    health_path = "/health"
    container_port = 0

    def __init__(self, kind: str, image: str, args: dict[str, Any], models_dir: str, port: int = 8000) -> None:
        super().__init__(image, args, models_dir, host_port=port)
        self.kind = kind
        self.proc: subprocess.Popen[bytes] | None = None
        self.log_path = Path(f"/tmp/servebench-{kind}.log")

    def run_args(self, artifact: ArtifactSpec) -> list[str]:
        if self.kind == "vllm":
            return vllm_cmd(artifact, self.args, self.host_port)
        if self.kind == "sglang":
            return sglang_cmd(artifact, self.args, self.host_port)
        if self.kind == "llamacpp":
            from cortex_serve.servebench.engines.base import artifact_path
            from cortex_serve.servebench.engines.llamacpp import server_args

            argv = server_args(self.args, artifact_path(artifact))
            argv[argv.index("--port") + 1] = str(self.host_port)
            argv[argv.index("--host") + 1] = "127.0.0.1"
            # server_args uses the Docker mount point /models; in process mode the files live under models_dir
            argv = [a.replace("/models/", f"{self.models_dir.rstrip('/')}/", 1) if a.startswith("/models/") else a
                    for a in argv]  # fmt: skip
            return ["/app/llama-server", *argv]
        raise ValueError(f"unsupported process engine {self.kind!r}")

    def start(self, artifact: ArtifactSpec, startup_timeout_s: float = 1800.0) -> str:
        cmd = self.run_args(artifact)
        log = self.log_path.open("wb")
        self.proc = subprocess.Popen(cmd, stdout=log, stderr=subprocess.STDOUT)
        deadline = time.monotonic() + startup_timeout_s
        while time.monotonic() < deadline:
            if self.proc.poll() is not None:
                raise RuntimeError(f"{self.kind} exited during startup:\n{self.logs()[-3000:]}")
            try:
                if httpx.get(self.base_url + self.health_path, timeout=2).status_code == 200:
                    return str(self.proc.pid)
            except httpx.HTTPError:
                pass
            time.sleep(2)
        self.stop()
        raise TimeoutError(f"{self.kind} not healthy after {startup_timeout_s}s")

    def logs(self) -> str:
        return self.log_path.read_text(errors="replace") if self.log_path.exists() else ""

    def version(self) -> str | None:
        probes = {
            "vllm": ["python3", "-c", "import vllm; print(vllm.__version__)"],
            "sglang": ["python3", "-c", "import sglang; print(sglang.__version__)"],
            "llamacpp": ["/app/llama-server", "--version"],
        }
        try:
            out = subprocess.run(probes[self.kind], capture_output=True, text=True, timeout=120, check=False)
        except (OSError, subprocess.TimeoutExpired, KeyError):
            return None
        return (out.stdout or out.stderr).strip().splitlines()[-1] if (out.stdout or out.stderr) else None

    def stop(self) -> None:
        if self.proc and self.proc.poll() is None:
            self.proc.terminate()
            try:
                self.proc.wait(timeout=60)
            except subprocess.TimeoutExpired:
                self.proc.kill()
        self.proc = None


def parse_vllm_memory(log: str) -> dict[str, float]:
    """KV capacity vLLM reports at startup (format verified per pinned version in docs/research/M2_*)."""
    out: dict[str, float] = {}
    if m := re.search(r"Available KV cache memory:\s*([\d.]+)\s*GiB", log):
        out["kv_available_gib"] = float(m.group(1))
    if m := re.search(r"GPU KV cache size:\s*([\d,]+)\s*tokens", log):
        out["kv_capacity_tokens"] = float(m.group(1).replace(",", ""))
    if m := re.search(r"Maximum concurrency for ([\d,]+) tokens per request:\s*([\d.]+)x", log):
        out["max_concurrency_ctx"] = float(m.group(1).replace(",", ""))
        out["max_concurrency"] = float(m.group(2))
    if m := re.search(r"Model loading took\s*([\d.]+)\s*GiB", log):
        out["model_load_gib"] = float(m.group(1))
    return out


def parse_sglang_memory(log: str) -> dict[str, float]:
    out: dict[str, float] = {}
    if m := re.search(r"max_total_num_tokens=(\d+)", log):
        out["kv_capacity_tokens"] = float(m.group(1))
    if m := re.search(r"KV Cache is allocated\. #tokens: (\d+), K size: ([\d.]+) GB, V size: ([\d.]+) GB", log):
        out["kv_capacity_tokens"] = float(m.group(1))
        out["kv_gb"] = float(m.group(2)) + float(m.group(3))
    return out


def parse_memory(kind: str, log: str) -> dict[str, float]:
    if kind == "vllm":
        return parse_vllm_memory(log)
    if kind == "sglang":
        return parse_sglang_memory(log)
    from cortex_serve.servebench.engines.llamacpp import parse_memory as llamacpp_memory

    return llamacpp_memory(log)
