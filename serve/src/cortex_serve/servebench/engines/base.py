"""Shared Docker lifecycle for engine launchers."""

from __future__ import annotations

import subprocess
import time
import uuid
from abc import ABC, abstractmethod
from typing import Any

import httpx

from cortex_serve.servebench.client import Protocol
from cortex_serve.servebench.config import ArtifactSpec


def artifact_path(artifact: ArtifactSpec) -> str:
    """Path of an artifact inside the models mount: ``<org>_<repo>/<file>``."""
    if artifact.file is None:
        raise ValueError(f"{artifact.name}: GGUF engines need `file`")
    return f"{artifact.repo.replace('/', '_')}/{artifact.file}"


def docker(*args: str, timeout_s: float = 120.0) -> str:
    out = subprocess.run(["docker", *args], capture_output=True, text=True, timeout=timeout_s, check=False)
    if out.returncode != 0:
        raise RuntimeError(f"docker {' '.join(args[:3])} failed: {out.stderr.strip()[:500]}")
    return out.stdout.strip()


class Engine(ABC):
    protocol: Protocol
    container_port: int
    health_path: str

    def __init__(self, image: str, args: dict[str, Any], models_dir: str, host_port: int = 18080) -> None:
        self.image = image
        self.args = args
        self.models_dir = models_dir
        self.host_port = host_port
        self.container: str | None = None
        self.model_name = "model"

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.host_port}"

    @abstractmethod
    def run_args(self, artifact: ArtifactSpec) -> list[str]:
        """Arguments after ``docker run -d`` (env, mounts, image, command)."""

    def after_start(self, artifact: ArtifactSpec) -> None:  # noqa: B027 - optional hook
        """Hook for engines that need setup after the container is healthy (e.g. Ollama import)."""

    def version(self) -> str | None:
        return None

    def start(self, artifact: ArtifactSpec, startup_timeout_s: float = 300.0) -> str:
        name = f"servebench-{uuid.uuid4().hex[:8]}"
        cid = docker(
            "run",
            "-d",
            "--name",
            name,
            "-p",
            f"127.0.0.1:{self.host_port}:{self.container_port}",
            *self.run_args(artifact),
        )
        self.container = cid
        deadline = time.monotonic() + startup_timeout_s
        while time.monotonic() < deadline:
            try:
                if httpx.get(self.base_url + self.health_path, timeout=2).status_code == 200:
                    self.after_start(artifact)
                    return cid
            except httpx.HTTPError:
                pass
            if docker("inspect", "-f", "{{.State.Running}}", cid) != "true":
                raise RuntimeError(f"engine exited during startup:\n{self.logs()[-2000:]}")
            time.sleep(1)
        self.stop()
        raise TimeoutError(f"engine not healthy after {startup_timeout_s}s")

    def logs(self) -> str:
        if not self.container:
            return ""
        out = subprocess.run(
            ["docker", "logs", self.container], capture_output=True, text=True, timeout=60, check=False
        )
        return out.stdout + out.stderr

    def stop(self) -> None:
        if self.container:
            subprocess.run(["docker", "rm", "-f", self.container], capture_output=True, timeout=60, check=False)
            self.container = None
