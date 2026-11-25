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


# Modal credits: $250, expire 2026-11-30 (D-030). Spend is gated in stages so a bad run can't strand the plan.
CREDIT_CEILING_USD = 245.0  # hard stop: never let worst case cross this ($5 margin before the card is charged)
STAGE_GATES_USD = (60.0, 120.0, 180.0)  # pause for a review with the owner before crossing each gate


def budget_check(spent_usd: float, worst_case_usd: float, approved_gate_usd: float) -> str | None:
    """Return a refusal reason, or None if the run may start.

    ``spent_usd`` is the provider-metered spend so far, ``worst_case_usd`` the run's hard-capped cost, and
    ``approved_gate_usd`` the highest stage gate the owner has approved (``STAGE_GATES_USD`` or the ceiling).
    """
    after = spent_usd + worst_case_usd
    if after > CREDIT_CEILING_USD:
        return (
            f"spent ${spent_usd:.2f} + worst case ${worst_case_usd:.2f} "
            f"would exceed the ${CREDIT_CEILING_USD:.0f} ceiling"
        )
    if after > approved_gate_usd:
        return (
            f"spent ${spent_usd:.2f} + worst case ${worst_case_usd:.2f} crosses the approved stage gate "
            f"${approved_gate_usd:.0f}: review spend with the owner, then pass --approved-gate"
        )
    return None


class WorkloadSpec(Strict):
    name: str
    input_tokens: int
    output_tokens: int
    input_jitter: float = 0.0  # +/- fraction of input length, uniform
    shared_prefix_tokens: int = 0
    instruction: str = "Continue the notes above in the same style. Keep writing without stopping."
    source: Literal["synthetic", "ultrachat"] = Field(
        default="synthetic",
        description="synthetic: exact-length word-list prompts; ultrachat: real first-turn user prompts "
        "(HuggingFaceH4/ultrachat_200k test_sft, pinned), natural length",
    )
    ignore_eos: bool = Field(default=True, description="False lets the model stop naturally (real prompts)")
    json_schema: dict[str, Any] | None = Field(
        default=None, description="Constrain output with response_format json_schema; validity is recorded"
    )
    check_schema: dict[str, Any] | None = Field(
        default=None, description="Only *check* outputs against this schema (no constraint), e.g. prompted JSON"
    )


class LoadSpec(Strict):
    mode: Literal["concurrency", "poisson"] = "concurrency"
    levels: list[float] = Field(description="Concurrency levels, or request rates (req/s) for poisson")
    requests_per_level: int = Field(description="Measured requests per level (per repeat)")
    requests_per_slot: int | None = Field(
        default=None,
        description="Concurrency mode: measure at least this many requests per concurrent slot, so high levels "
        "stay saturated (n = max(requests_per_level, requests_per_slot * level))",
    )
    warmup_requests: int = 2
    timeout_s: float = 600.0

    def requests_for(self, level: float) -> int:
        if self.requests_per_slot is None or self.mode != "concurrency":
            return self.requests_per_level
        return max(self.requests_per_level, self.requests_per_slot * int(level))


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
