"""The arithmetic: weights, KV cache, concurrency, decode ceiling, prefill FLOPs.

All functions are pure and work on a ``ModelSpec``, so they are easy to test against hand
calculations and easy to port to TypeScript for the web site (golden vectors keep them in sync).
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel

from cortex_serve.memcalc.devices import Device
from cortex_serve.memcalc.dtypes import APPROXIMATE_WEIGHT_FORMATS, kv_bytes, weight_bytes
from cortex_serve.memcalc.families import producer_of
from cortex_serve.memcalc.gguf import GGUFHeader
from cortex_serve.memcalc.model import LayerKind, LayerSpec, ModelSpec

Components = Literal["text", "all"]

GIB = 2**30
MIB = 2**20
DEFAULT_RESERVE_GIB = {"gpu": 1.5, "cpu": 0.5, "unified": 0.5}

# Formats produced by llm-compressor-style recipes quantize the decoder's linear layers only;
# embeddings, lm_head, norms and non-text towers stay in 16-bit. GGUF quantizes embeddings too.
_LINEAR_ONLY_FORMATS = frozenset({"fp8", "int8", "w4a16-g128", "awq-g128", "mxfp4"})


# ---------------------------------------------------------------- weights


def loaded_param_groups(
    spec: ModelSpec, components: Components = "text", ple_offload: bool = False, with_mtp: bool = False
) -> tuple[int, int]:
    """Return (params in quantizable decoder linears, params kept in 16-bit for llm-compressor formats)."""
    p = spec.params
    decoder = p.decoder_layers + p.experts + p.per_layer_compute + p.unused_shared_kv
    kept = p.embeddings + p.lm_head + p.other + (0 if ple_offload else p.per_layer_embeddings)
    if with_mtp:
        decoder += p.mtp
    if components == "all":
        kept += p.vision + p.audio
    return decoder, kept


def weights_bytes(
    spec: ModelSpec,
    fmt: str,
    components: Components = "text",
    ple_offload: bool = False,
    with_mtp: bool = False,
) -> int:
    decoder, kept = loaded_param_groups(spec, components, ple_offload, with_mtp)
    b = weight_bytes(fmt)
    kept_b = 2.0 if fmt.lower() in _LINEAR_ONLY_FORMATS else b
    return math.ceil(decoder * b + kept * kept_b)


def active_weight_bytes(spec: ModelSpec, fmt: str) -> int:
    """Bytes read from memory to decode one token of one sequence (MoE: routed experts only)."""
    b = weight_bytes(fmt)
    lm_b = 2.0 if fmt.lower() in _LINEAR_ONLY_FORMATS else b
    return math.ceil((spec.active_params - spec.lm_head_params) * b + spec.lm_head_params * lm_b)


def gguf_weight_bytes(gguf: GGUFHeader, ple_offload: bool = False) -> int:
    """Exact weight bytes of a GGUF (as llama.cpp loads it), optionally without the PLE lookup table."""
    return gguf.weight_bytes - (gguf.bytes_matching("per_layer_token_embd") if ple_offload else 0)


def gguf_active_bytes(gguf: GGUFHeader, spec: ModelSpec) -> int:
    """Bytes read per decoded token: everything except lookup-only tables (PLE; token_embd when a separate
    output.weight exists) and, for MoE, the non-routed share of the expert tensors."""
    names = {t.name for t in gguf.tensors}
    total = 0.0
    for t in gguf.tensors:
        if "per_layer_token_embd" in t.name:
            continue
        if t.name == "token_embd.weight" and "output.weight" in names:
            continue
        if "_exps" in t.name and spec.moe:
            total += t.nbytes * spec.moe.active_fraction
            continue
        total += t.nbytes
    return math.ceil(total)


# ---------------------------------------------------------------- KV cache and state


def layer_seq_bytes(layer: LayerSpec, ctx: int, kv_b: float) -> float:
    if layer.kind == LayerKind.FULL:
        return 2 * layer.kv_heads * layer.head_dim * kv_b * ctx
    if layer.kind == LayerKind.SLIDING:
        assert layer.window is not None
        return 2 * layer.kv_heads * layer.head_dim * kv_b * min(ctx, layer.window)
    if layer.kind == LayerKind.LINEAR:
        return layer.state_elems * layer.state_dtype_bytes + layer.conv_elems * layer.conv_dtype_bytes
    return 0.0  # SHARED: reads another layer's KV


def seq_bytes(spec: ModelSpec, ctx: int, kv_fmt: str = "bf16") -> int:
    """KV cache + recurrent state for one sequence of ``ctx`` tokens."""
    kv_b = kv_bytes(kv_fmt)
    return math.ceil(sum(layer_seq_bytes(layer, ctx, kv_b) for layer in spec.layers))


def fixed_state_bytes(spec: ModelSpec) -> int:
    return seq_bytes(spec, 0)  # only LINEAR layers contribute at ctx=0


def kv_bytes_per_token(spec: ModelSpec, kv_fmt: str = "bf16") -> int:
    """Marginal KV growth per token once past every sliding window (FULL layers only)."""
    kv_b = kv_bytes(kv_fmt)
    return math.ceil(sum(2 * x.kv_heads * x.head_dim * kv_b for x in spec.layers if x.kind == LayerKind.FULL))


# ---------------------------------------------------------------- FLOPs


def attention_flops_per_token(spec: ModelSpec, ctx: int) -> int:
    """QK^T and AV for one new token attending over its context (shared layers still compute attention)."""
    total = 0
    for layer in spec.layers:
        src = producer_of(spec.layers, layer)
        if src.kind == LayerKind.LINEAR:
            continue  # O(1) per token; folded into the linear term
        span = min(ctx, src.window) if src.kind == LayerKind.SLIDING and src.window else ctx
        total += 4 * layer.q_heads * layer.head_dim * span
    return total


# ---------------------------------------------------------------- estimate


class Estimate(BaseModel):
    model: str
    revision: str | None
    device: str
    weight_format: str
    kv_format: str
    context: int
    util: float
    reserve_gib: float
    components: str
    layer_kinds: dict[str, int]
    params_stored: int
    params_loaded: int
    params_active: int
    param_source: str
    weights_gib: float
    kv_bytes_per_token: int
    fixed_state_mib_per_seq: float
    seq_mib: float
    budget_gib: float
    free_for_kv_gib: float
    max_concurrent_seqs: int
    decode_ceiling_tok_s: float | None
    decode_ceiling_at_ctx_tok_s: float | None
    roofline_crossover_batch: float | None
    prefill_tflop_per_1k_tokens: float
    attention_gflop_per_token_at_ctx: float
    notes: list[str]


def estimate(
    spec: ModelSpec,
    device: Device,
    weight_fmt: str = "bf16",
    kv_fmt: str = "auto",
    ctx: int = 8192,
    util: float = 0.9,
    reserve_gib: float | None = None,
    components: Components = "text",
    ple_offload: bool = False,
    with_mtp: bool = False,
    gguf: GGUFHeader | None = None,
) -> Estimate:
    """``gguf``: when given, weight bytes and bytes-read-per-token come from the GGUF header (exact),
    and ``weight_fmt`` is only a label."""
    if ctx <= 0:
        raise ValueError("ctx must be positive")
    if not 0 < util <= 1:
        raise ValueError("util must be in (0, 1]")
    notes: list[str] = []
    reserve = DEFAULT_RESERVE_GIB[device.kind] if reserve_gib is None else reserve_gib
    if gguf is not None:
        w_bytes = gguf_weight_bytes(gguf, ple_offload)
        notes.append(f"weights from GGUF header: exact ({gguf.bits_per_weight:.2f} bits/weight average)")
    else:
        w_bytes = weights_bytes(spec, weight_fmt, components, ple_offload, with_mtp)
    decoder, kept = loaded_param_groups(spec, components, ple_offload, with_mtp)
    per_seq = seq_bytes(spec, ctx, kv_fmt)
    budget = device.memory_bytes * util
    free = budget - w_bytes - reserve * GIB
    max_seqs = max(0, math.floor(free / per_seq)) if per_seq else 0
    if free <= 0:
        notes.append("weights + reserve exceed the memory budget: the model does not fit")
    elif max_seqs == 0:
        notes.append("weights fit, but not even one sequence of this context fits in the KV budget")

    active = gguf_active_bytes(gguf, spec) if gguf is not None else active_weight_bytes(spec, weight_fmt)
    ceiling = ceiling_ctx = crossover = None
    if device.bandwidth_gbs:
        bw = device.bandwidth_gbs * 1e9
        ceiling = bw / active
        ceiling_ctx = bw / (active + per_seq)
        tflops = device.bf16_tflops
        if weight_fmt.lower() == "fp8":
            if device.fp8_tflops:
                tflops = device.fp8_tflops
            elif device.kind == "gpu":
                notes.append(
                    f"{device.name} has no native FP8 compute: engines fall back to weight-only FP8 kernels "
                    "that dequantize to 16-bit (memory savings remain, compute does not speed up)"
                )
        if tflops:
            # decode FLOPs/step = 2*P*B ; bytes/step = P*b  ->  compute-bound once B > (F/BW) * b / 2
            crossover = tflops * 1e12 / bw * weight_bytes(weight_fmt) / 2
        elif device.kind == "gpu":
            notes.append(f"no 16-bit tensor-core peak listed for {device.name}: compute crossover omitted")
    else:
        notes.append(f"no bandwidth figure for {device.name}: decode ceilings omitted")

    if gguf is None and weight_fmt.lower() in APPROXIMATE_WEIGHT_FORMATS:
        notes.append(f"{weight_fmt} is a mixed-precision GGUF format; bytes/param is an average")
    if spec.params.per_layer_embeddings and not ple_offload:
        notes.append("per-layer embedding tables counted as device memory (use --ple-offload if the engine maps them)")
    if spec.count(LayerKind.SHARED):
        notes.append(f"{spec.count(LayerKind.SHARED)} layers reuse another layer's KV and own none")
    if spec.count(LayerKind.LINEAR):
        notes.append("linear-attention layers hold a fixed per-sequence state instead of a growing KV cache")
    if spec.params.tied_duplicate:
        notes.append(
            f"checkpoint stores a redundant lm_head ({spec.params.tied_duplicate / 1e6:.0f}M params) "
            "although the config ties it to the embeddings; it is not loaded"
        )
    if spec.params.source == "analytic":
        notes.append("param counts are analytic (no safetensors header available)")

    kinds = {k.value: spec.count(k) for k in LayerKind if spec.count(k)}
    return Estimate(
        model=spec.name,
        revision=spec.revision,
        device=device.name,
        weight_format=weight_fmt,
        kv_format=kv_fmt,
        context=ctx,
        util=util,
        reserve_gib=reserve,
        components=components,
        layer_kinds=kinds,
        params_stored=spec.params.stored,
        params_loaded=decoder + kept,
        params_active=spec.active_params,
        param_source=spec.params.source,
        weights_gib=w_bytes / GIB,
        kv_bytes_per_token=kv_bytes_per_token(spec, kv_fmt),
        fixed_state_mib_per_seq=fixed_state_bytes(spec) / MIB,
        seq_mib=per_seq / MIB,
        budget_gib=budget / GIB,
        free_for_kv_gib=free / GIB,
        max_concurrent_seqs=max_seqs,
        decode_ceiling_tok_s=ceiling,
        decode_ceiling_at_ctx_tok_s=ceiling_ctx,
        roofline_crossover_batch=crossover,
        prefill_tflop_per_1k_tokens=2 * spec.active_params * 1000 / 1e12,
        attention_gflop_per_token_at_ctx=attention_flops_per_token(spec, ctx) / 1e9,
        notes=notes,
    )
