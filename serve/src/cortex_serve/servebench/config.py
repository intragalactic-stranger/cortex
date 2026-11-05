"""Benchmark run configuration (YAML -> pydantic)."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Literal

import yaml
from pydantic import BaseModel, ConfigDict, Field


class Strict(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)


class ArtifactSpec(Strict):
    """A model artifact served by the engine (for llama.cpp/Ollama: one GGUF file)."""

    name: str = Field(description="Short label used in reports, e.g. 'qwen3-1.7b-q4_k_m'")
    repo: str
    revision: str
    file: str | None = None
    family: str = Field(description="memcalc fixture / HF repo of the base model, e.g. Qwen/Qwen3-1.7B")
    quant: str


class TokenizerSpec(Strict):
    repo: str
    revision: str


class EngineSpec(Strict):
    kind: Literal["llamacpp", "ollama", "vllm", "sglang"]
    image: str = Field(description="Pinned image ref (name:tag@sha256:...). Recorded in the manifest.")
    launch: Literal["docker", "process"] = Field(
        "docker", description="docker: local `docker run`; process: engine binary inside this container (Modal)"
    )
    args: dict[str, Any] = Field(default_factory=dict)


class BudgetSpec(Strict):
    """Hard cost cap for remote GPU runs: the Modal function timeout is set to ``max_minutes``."""

    gpu: str
    usd_per_hour: float = Field(description="Verified on the provider's pricing page on the day of the run")
    max_minutes: float

    @property
    def max_usd(self) -> float:
        return self.usd_per_hour * self.max_minutes / 60


class WorkloadSpec(Strict):
    name: str
    input_tokens: int
    output_tokens: int
    input_jitter: float = 0.0  # +/- fraction of input length, uniform
    shared_prefix_tokens: int = 0
    instruction: str = "Continue the notes above in the same style. Keep writing without stopping."


class LoadSpec(Strict):
    mode: Literal["concurrency", "poisson"] = "concurrency"
    levels: list[float] = Field(description="Concurrency levels, or request rates (req/s) for poisson")
    requests_per_level: int = Field(description="Measured requests per level (per repeat)")
    warmup_requests: int = 2
    timeout_s: float = 600.0


class SLO(Strict):
    ttft_s: float
    tpot_s: float = Field(description="Per-request mean time per output token after the first")


class RunConfig(Strict):
    name: str
    description: str = ""
    experiment: str
    engine: EngineSpec
    artifacts: list[ArtifactSpec]
    tokenizers: dict[str, TokenizerSpec] = Field(description="family -> tokenizer source")
    workloads: list[WorkloadSpec]
    load: LoadSpec
    slo: SLO
    repeats: int = 3
    seed: int = 1234
    disable_thinking: bool = True
    hardware: dict[str, Any] = Field(default_factory=dict)
    budget: BudgetSpec | None = None


def load_config(path: Path) -> RunConfig:
    return RunConfig.model_validate(yaml.safe_load(path.read_text()))
