"""Turn a Hugging Face ``config.json`` into per-layer memory specs.

Supported memory designs (each verified against the model's real tensor shapes in tests):

* dense / GQA / MoE transformers (Qwen3, Llama, Mistral, Phi-3, SmolLM3, ...): every layer FULL
* Gemma 4: SLIDING + FULL layers (FULL uses ``global_head_dim``), and the last
  ``num_kv_shared_layers`` layers reuse the KV of the last non-shared layer of the same type
  (see transformers ``Gemma4TextConfig.kv_sharing_modes``)
* Qwen3.5 hybrids: LINEAR (Gated DeltaNet) + FULL layers; the DeltaNet recurrent state is
  stored in ``mamba_ssm_dtype`` (float32 in the released configs)
"""

from __future__ import annotations

from typing import Any

from cortex_serve.memcalc.dtypes import STATE_DTYPE_BYTES
from cortex_serve.memcalc.model import LayerKind, LayerSpec, MoESpec

Config = dict[str, Any]


class UnsupportedArchitectureError(ValueError):
    pass


_KIND_BY_LAYER_TYPE = {
    "full_attention": LayerKind.FULL,
    "attention": LayerKind.FULL,
    "sliding_attention": LayerKind.SLIDING,
    "linear_attention": LayerKind.LINEAR,
}


def text_config(cfg: Config) -> Config:
    """Multimodal checkpoints nest the language model under ``text_config``."""
    tc = cfg.get("text_config")
    return tc if isinstance(tc, dict) else cfg


def tied_embeddings(cfg: Config) -> bool:
    tc = text_config(cfg)
    value = cfg.get("tie_word_embeddings", tc.get("tie_word_embeddings"))
    return bool(value) if value is not None else False


def model_dtype_bytes(cfg: Config) -> float:
    tc = text_config(cfg)
    name = (
        tc.get("dtype") or tc.get("torch_dtype") or cfg.get("dtype") or cfg.get("torch_dtype") or "bfloat16"
    )
    return STATE_DTYPE_BYTES.get(str(name), 2.0)


def _layer_kinds(tc: Config) -> list[LayerKind]:
    n = int(tc["num_hidden_layers"])
    types = tc.get("layer_types")
    if types:
        if len(types) != n:
            raise UnsupportedArchitectureError(f"layer_types has {len(types)} entries but {n} layers")
        unknown = sorted({t for t in types if t not in _KIND_BY_LAYER_TYPE})
        if unknown:
            raise UnsupportedArchitectureError(f"unsupported layer types {unknown} (e.g. Mamba layers)")
        return [_KIND_BY_LAYER_TYPE[t] for t in types]
    if tc.get("use_sliding_window") and tc.get("sliding_window"):
        return [LayerKind.SLIDING] * n
    return [LayerKind.FULL] * n


def build_layers(cfg: Config) -> tuple[LayerSpec, ...]:
    tc = text_config(cfg)
    hidden = int(tc["hidden_size"])
    q_heads = int(tc["num_attention_heads"])
    kv_heads = int(tc.get("num_key_value_heads") or q_heads)
    head_dim = int(tc.get("head_dim") or hidden // q_heads)
    window = tc.get("sliding_window")
    # Gemma 4 global (full-attention) layers use a larger head_dim and may use their own KV-head count
    global_head_dim = int(tc.get("global_head_dim") or head_dim)
    global_kv_heads = int(tc.get("num_global_key_value_heads") or kv_heads)
    act_bytes = model_dtype_bytes(cfg)

    kinds = _layer_kinds(tc)
    layers: list[LayerSpec] = []
    for i, kind in enumerate(kinds):
        if kind == LayerKind.FULL:
            layers.append(
                LayerSpec(
                    index=i, kind=kind, q_heads=q_heads, kv_heads=global_kv_heads, head_dim=global_head_dim
                )
            )
        elif kind == LayerKind.SLIDING:
            layers.append(
                LayerSpec(
                    index=i,
                    kind=kind,
                    q_heads=q_heads,
                    kv_heads=kv_heads,
                    head_dim=head_dim,
                    window=int(window) if window else None,
                )
            )
        else:  # LINEAR (Gated DeltaNet)
            k_heads = int(tc["linear_num_key_heads"])
            v_heads = int(tc["linear_num_value_heads"])
            k_dim = int(tc["linear_key_head_dim"])
            v_dim = int(tc["linear_value_head_dim"])
            kernel = int(tc["linear_conv_kernel_dim"])
            state_dtype = str(tc.get("mamba_ssm_dtype") or "float32")
            layers.append(
                LayerSpec(
                    index=i,
                    kind=kind,
                    state_elems=v_heads * k_dim * v_dim,
                    state_dtype_bytes=STATE_DTYPE_BYTES[state_dtype],
                    conv_elems=(2 * k_heads * k_dim + v_heads * v_dim) * (kernel - 1),
                    conv_dtype_bytes=act_bytes,
                )
            )
    return _apply_kv_sharing(layers, int(tc.get("num_kv_shared_layers") or 0))


def _apply_kv_sharing(layers: list[LayerSpec], num_shared: int) -> tuple[LayerSpec, ...]:
    """Mirror transformers' Gemma4 ``kv_sharing_modes``: the last ``num_shared`` layers reuse the KV of
    the last non-shared layer of the same kind. Shared layers keep their attention shape (they still
    compute attention) but own no KV."""
    if num_shared == 0:
        return tuple(layers)
    first_shared = len(layers) - num_shared
    producers: dict[LayerKind, LayerSpec] = {}
    for layer in layers[:first_shared]:
        producers[layer.kind] = layer
    out = list(layers[:first_shared])
    for layer in layers[first_shared:]:
        producer = producers.get(layer.kind)
        if producer is None:
            raise UnsupportedArchitectureError(f"layer {layer.index} ({layer.kind}) has no KV producer")
        out.append(layer.model_copy(update={"kind": LayerKind.SHARED, "shares_from": producer.index}))
    return tuple(out)


def producer_of(layers: tuple[LayerSpec, ...], layer: LayerSpec) -> LayerSpec:
    return layers[layer.shares_from] if layer.shares_from is not None else layer


def build_moe(cfg: Config) -> MoESpec | None:
    tc = text_config(cfg)
    experts = tc.get("num_experts") or tc.get("num_local_experts")
    per_token = tc.get("num_experts_per_tok") or tc.get("top_k_experts")
    if experts and per_token:
        return MoESpec(num_experts=int(experts), experts_per_token=int(per_token))
    return None
