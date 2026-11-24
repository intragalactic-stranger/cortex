"""llm-compressor 0.15.0 recipes (import paths verified in docs/research M3 §1.2). Built lazily so the package
imports without llm-compressor installed (it lives in its own Modal image: it pins compressed-tensors 0.20.1,
vLLM 0.32.0 pins 0.17.0)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class Recipe:
    name: str
    description: str
    needs_calibration: bool
    target_engine: str = "vllm"


RECIPES = {
    r.name: r
    for r in (
        Recipe("fp8-dynamic", "FP8 E4M3 weights per-channel + dynamic per-token activations (data-free)", False),
        Recipe("w8a8-int8", "SmoothQuant 0.8 + GPTQ int8 W per-channel / A per-token dynamic", True),
        Recipe("w4a16-gptq-g128", "GPTQ int4 weights, group 128, symmetric, static actorder", True),
        Recipe("w4a16-awq-asym", "AWQ scales + RTN int4 weights, group 128, asymmetric", True),
        Recipe("w4a16-gptq-g32-sym", "GPTQ int4 group 32 symmetric (matches Google's Gemma 4 QAT scheme)", True),
    )
}


def build(name: str, ignore: list[str] | None = None) -> list[Any]:
    """Return the llm-compressor modifier list for ``name``."""
    from llmcompressor.modifiers.gptq import GPTQModifier
    from llmcompressor.modifiers.quantization import QuantizationModifier

    ign = ignore or ["lm_head"]
    if name == "fp8-dynamic":
        return [QuantizationModifier(targets="Linear", scheme="FP8_DYNAMIC", ignore=ign)]
    if name == "w8a8-int8":
        from llmcompressor.modifiers.transform.smoothquant import SmoothQuantModifier

        return [SmoothQuantModifier(smoothing_strength=0.8), GPTQModifier(targets="Linear", scheme="W8A8", ignore=ign)]
    if name == "w4a16-gptq-g128":
        return [GPTQModifier(targets="Linear", scheme="W4A16", ignore=ign)]
    if name == "w4a16-awq-asym":
        from llmcompressor.modifiers.transform.awq import AWQModifier

        return [
            AWQModifier(duo_scaling="both"),
            QuantizationModifier(targets=["Linear"], scheme="W4A16_ASYM", ignore=ign),
        ]
    if name == "w4a16-gptq-g32-sym":
        from compressed_tensors.quantization import QuantizationArgs, QuantizationScheme

        weights = QuantizationArgs(
            num_bits=4, type="int", strategy="group", group_size=32, symmetric=True, dynamic=False
        )
        scheme = QuantizationScheme(targets=["Linear"], weights=weights)
        return [GPTQModifier(config_groups={"group_0": scheme}, ignore=ign)]
    raise KeyError(f"unknown recipe {name!r}; known: {sorted(RECIPES)}")
