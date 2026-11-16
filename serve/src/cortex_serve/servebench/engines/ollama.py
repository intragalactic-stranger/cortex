"""Ollama launcher: imports the same GGUF file via a Modelfile, then serves it on ``/api/chat``.

Ollama cannot ignore EOS, so output lengths may be shorter than requested; per-token metrics (TTFT,
TPOT, tok/s) remain comparable and output lengths are recorded per request.
"""

from __future__ import annotations

from typing import Any

from cortex_serve.servebench.config import ArtifactSpec
from cortex_serve.servebench.engines.base import Engine, artifact_path, docker


class OllamaEngine(Engine):
    protocol = "ollama"
    container_port = 11434
    health_path = "/api/version"

    def __init__(self, image: str, args: dict[str, Any], models_dir: str, host_port: int = 18080) -> None:
        super().__init__(image, args, models_dir, host_port)
        self.model_name = "servebench"

    def run_args(self, artifact: ArtifactSpec) -> list[str]:
        env = {
            "OLLAMA_NUM_PARALLEL": str(self.args.get("parallel", 1)),
            "OLLAMA_CONTEXT_LENGTH": str(self.args.get("ctx_per_slot", 2048)),
            "OLLAMA_KEEP_ALIVE": "-1",
            "OLLAMA_MAX_LOADED_MODELS": "1",
        }
        flags = [x for k, v in env.items() for x in ("-e", f"{k}={v}")]
        # Ollama copies the GGUF into its own store on `create`. A disk volume (not tmpfs) keeps that copy
        # out of the container's RAM accounting, so memory comparisons with llama.cpp stay fair.
        self.volume = f"servebench-ollama-{id(self):x}"
        return [
            *flags,
            "-v",
            f"{self.models_dir}:/models:ro",
            "-v",
            f"{self.volume}:/root/.ollama",
            self.image,
        ]

    def modelfile(self, artifact: ArtifactSpec) -> str:
        lines = [f"FROM /models/{artifact_path(artifact)}"]
        if "threads" in self.args:
            lines.append(f"PARAMETER num_thread {self.args['threads']}")
        return "\n".join(lines) + "\n"

    def after_start(self, artifact: ArtifactSpec) -> None:
        assert self.container
        script = (
            f"printf '%s' '{self.modelfile(artifact)}' > /tmp/Modelfile "
            f"&& ollama create {self.model_name} -f /tmp/Modelfile"
        )
        docker("exec", self.container, "sh", "-c", script, timeout_s=900)

    def stop(self) -> None:
        super().stop()
        if getattr(self, "volume", None):
            docker("volume", "rm", "-f", self.volume)

    def version(self) -> str | None:
        try:
            return docker("run", "--rm", "--entrypoint", "ollama", self.image, "--version", timeout_s=60).split()[-1]
        except RuntimeError:
            return None
