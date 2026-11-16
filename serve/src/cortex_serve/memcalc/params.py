"""Parameter counting, two ways.

1. ``classify_tensors``: exact counts from the checkpoint's safetensors header (tensor names + shapes,
   fetched without downloading weights). This is the source of truth.
2. ``analytic_params``: the textbook formula from ``config.json`` for llama-style dense and MoE
   transformers. It is the learning artifact, and tests require it to match (1) exactly.
"""

from __future__ import annotations

import math
import re
from collections.abc import Mapping, Sequence

from cortex_serve.memcalc.families import Config, text_config, tied_embeddings
from cortex_serve.memcalc.model import LayerKind, LayerSpec, ParamBreakdown

# name -> (dtype, shape)
TensorTable = Mapping[str, tuple[str, Sequence[int]]]

_VISION = re.compile(r"(^|\.)(visual|vision_tower|vision_model|embed_vision|multi_modal_projector)\.")
_AUDIO = re.compile(r"(^|\.)(audio_tower|embed_audio)\.")
_MTP = re.compile(r"(^|\.)mtp\.")
_PLE_TABLE = re.compile(r"embed_tokens_per_layer\.")
_PLE_COMPUTE = re.compile(r"per_layer_model_projection\.|per_layer_projection_norm\.")
_EMBED = re.compile(r"(^|\.)embed_tokens\.weight$")
_LM_HEAD = re.compile(r"(^|\.)lm_head\.")
_LAYER = re.compile(r"(^|\.)layers\.(\d+)\.")
_EXPERT = re.compile(r"\.experts\.")
_KV_WEIGHTS = re.compile(r"\.self_attn\.(k_proj|v_proj|k_norm|v_norm)\.")

_SIMPLE_RULES: tuple[tuple[re.Pattern[str], str], ...] = (
    (_VISION, "vision"),
    (_AUDIO, "audio"),
    (_MTP, "mtp"),
    (_PLE_TABLE, "per_layer_embeddings"),
    (_PLE_COMPUTE, "per_layer_compute"),
    (_EMBED, "embeddings"),
    (_LM_HEAD, "lm_head"),
)


def classify_tensor(name: str, layers: Sequence[LayerSpec], tied: bool = False) -> str:
    """Return the ParamBreakdown field a tensor belongs to."""
    for pattern, bucket in _SIMPLE_RULES:
        if pattern.search(name):
            if bucket == "lm_head" and tied:
                return "tied_duplicate"  # e.g. Qwen3-0.6B/1.7B store lm_head although tie_word_embeddings=true
            return bucket
    m = _LAYER.search(name)
    if m:
        if _EXPERT.search(name):
            return "experts"
        idx = int(m.group(2))
        if idx < len(layers) and layers[idx].kind == LayerKind.SHARED and _KV_WEIGHTS.search(name):
            return "unused_shared_kv"
        return "decoder_layers"
    return "other"


def classify_tensors(tensors: TensorTable, layers: Sequence[LayerSpec], tied: bool = False) -> ParamBreakdown:
    buckets: dict[str, int] = {}
    for name, (_dtype, shape) in tensors.items():
        bucket = classify_tensor(name, layers, tied)
        buckets[bucket] = buckets.get(bucket, 0) + math.prod(shape)
    return ParamBreakdown(source="safetensors-header", decoder_layers=buckets.pop("decoder_layers", 0), **buckets)


_LLAMA_STYLE = {"llama", "mistral", "qwen2", "qwen3", "qwen3_moe", "smollm3"}
_QK_NORM = {"qwen3", "qwen3_moe"}


def analytic_params(cfg: Config) -> ParamBreakdown | None:
    """Closed-form count for llama-style decoders; ``None`` for families it doesn't cover."""
    tc = text_config(cfg)
    mt = tc.get("model_type")
    if mt not in _LLAMA_STYLE:
        return None
    h = int(tc["hidden_size"])
    v = int(tc["vocab_size"])
    n_layers = int(tc["num_hidden_layers"])
    nh = int(tc["num_attention_heads"])
    kv = int(tc.get("num_key_value_heads") or nh)
    hd = int(tc.get("head_dim") or h // nh)

    attn = h * nh * hd + 2 * h * kv * hd + nh * hd * h  # q, k, v, o
    if tc.get("attention_bias"):
        attn += nh * hd + 2 * kv * hd
    if mt in _QK_NORM:
        attn += 2 * hd  # q_norm, k_norm
    norms = 2 * h  # input + post-attention RMSNorm

    experts = 0
    num_experts = tc.get("num_experts")
    if num_experts:
        moe_i = int(tc["moe_intermediate_size"])
        experts = n_layers * int(num_experts) * 3 * h * moe_i
        mlp = h * int(num_experts)  # router
    else:
        mlp = 3 * h * int(tc["intermediate_size"])  # gate, up, down (SwiGLU)

    tied = tied_embeddings(cfg)
    return ParamBreakdown(
        source="analytic",
        decoder_layers=n_layers * (attn + norms + mlp),
        experts=experts,
        embeddings=v * h,
        lm_head=0 if tied else v * h,
        other=h,  # final norm
    )


def check_shapes(tensors: TensorTable, layers: Sequence[LayerSpec]) -> list[str]:
    """Cross-check the config interpretation against real tensor shapes. Returns mismatch messages."""
    problems: list[str] = []
    for name, (_dtype, shape) in tensors.items():
        if _VISION.search(name) or _AUDIO.search(name) or _MTP.search(name):
            continue
        m = _LAYER.search(name)
        if not m:
            continue
        idx = int(m.group(2))
        if idx >= len(layers):
            problems.append(f"{name}: layer index beyond config")
            continue
        layer = layers[idx]
        if name.endswith("self_attn.k_proj.weight") and layer.kind != LayerKind.SHARED:
            expected = layer.kv_heads * layer.head_dim
            if shape[0] != expected:
                problems.append(f"{name}: rows {shape[0]} != kv_heads*head_dim {expected}")
        if name.endswith("linear_attn.conv1d.weight"):
            channels, kernel = shape[0], shape[-1]
            if channels * (kernel - 1) != layer.conv_elems:
                problems.append(f"{name}: conv state {channels}x{kernel - 1} != {layer.conv_elems}")
    return problems
