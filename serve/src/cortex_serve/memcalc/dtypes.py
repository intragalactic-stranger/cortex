"""Bytes per element for weight and KV-cache formats.

Weight formats include their per-group scale overhead. GGUF k-quant values are average
bits-per-weight; "_M" mixes keep some tensors at higher precision, so they are approximate
and will be checked against real file sizes in M3.
"""

from __future__ import annotations

# bytes per parameter
WEIGHT_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "fp8": 1.0,  # E4M3, per-tensor/per-channel scales are negligible
    "int8": 1.0,  # W8A8 per-channel scales negligible
    "w4a16-g128": 0.5 + 2 / 128,  # int4 + one fp16 scale per 128 weights (GPTQ)
    "awq-g128": 0.5 + 2 / 128 + 0.5 / 128,  # + int4 zero-point per group
    "mxfp4": 4.25 / 8,  # 4-bit + one 8-bit shared exponent per 32 values
    # GGUF (llama.cpp) average bits per weight / 8
    "q8_0": 8.5 / 8,
    "q6_k": 6.5625 / 8,
    "q5_k_m": 5.69 / 8,  # approximate mix
    "q4_k_m": 4.89 / 8,  # approximate mix
    "q4_0": 4.5 / 8,
}

# bytes per KV-cache element
KV_BYTES: dict[str, float] = {
    "fp32": 4.0,
    "bf16": 2.0,
    "fp16": 2.0,
    "auto": 2.0,  # engines default to the model dtype, which is 16-bit for every model in our lineup
    "fp8": 1.0,
    "q8_0": 8.5 / 8,  # llama.cpp --cache-type-k q8_0
    "q4_0": 4.5 / 8,
}

APPROXIMATE_WEIGHT_FORMATS = frozenset({"q5_k_m", "q4_k_m"})

STATE_DTYPE_BYTES: dict[str, float] = {"float32": 4.0, "bfloat16": 2.0, "float16": 2.0}


def weight_bytes(fmt: str) -> float:
    try:
        return WEIGHT_BYTES[fmt.lower()]
    except KeyError:
        raise ValueError(f"unknown weight format {fmt!r}; choose from {sorted(WEIGHT_BYTES)}") from None


def kv_bytes(fmt: str) -> float:
    try:
        return KV_BYTES[fmt.lower()]
    except KeyError:
        raise ValueError(f"unknown KV-cache format {fmt!r}; choose from {sorted(KV_BYTES)}") from None
