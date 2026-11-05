"""Normalised, architecture-agnostic description of a model's memory-relevant structure.

The key idea: every architecture becomes a list of per-layer specs, so that dense GQA,
sliding-window, cross-layer KV sharing, and linear-attention (recurrent state) layers are
all handled by the same arithmetic in ``calc.py``.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class LayerKind(StrEnum):
    FULL = "full"  # attention over the whole context; KV grows with every token
    SLIDING = "sliding"  # attention over the last `window` tokens; KV capped at the window
    SHARED = "shared"  # reuses another layer's KV (Gemma 4 KV sharing); owns no KV
    LINEAR = "linear"  # linear attention / Gated DeltaNet; fixed-size recurrent + conv state


class LayerSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    index: int
    kind: LayerKind
    q_heads: int = 0
    kv_heads: int = 0
    head_dim: int = 0
    window: int | None = None
    shares_from: int | None = None
    # Linear-attention state (per sequence), in elements; dtype sizes are given separately.
    state_elems: int = 0
    state_dtype_bytes: float = 0.0
    conv_elems: int = 0
    conv_dtype_bytes: float = 0.0

    @property
    def owns_kv(self) -> bool:
        return self.kind in (LayerKind.FULL, LayerKind.SLIDING)


class MoESpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    num_experts: int
    experts_per_token: int

    @property
    def active_fraction(self) -> float:
        return self.experts_per_token / self.num_experts


class ParamBreakdown(BaseModel):
    """Parameter counts by component. All values are element counts (not bytes)."""

    model_config = ConfigDict(frozen=True)

    source: str = Field(description="'safetensors-header' (exact) or 'analytic' (from config.json)")
    decoder_layers: int = Field(description="Transformer blocks, excluding MoE expert weights")
    experts: int = 0
    embeddings: int = 0
    lm_head: int = 0  # 0 when tied to the embeddings
    per_layer_embeddings: int = 0  # Gemma PLE lookup tables (looked up, not multiplied)
    per_layer_compute: int = 0  # Gemma PLE projections that *are* matmuls
    unused_shared_kv: int = 0  # k/v weights stored in KV-shared layers but never used
    tied_duplicate: int = 0  # lm_head stored even though config ties it to the embeddings (never loaded)
    vision: int = 0
    audio: int = 0
    mtp: int = 0  # multi-token-prediction heads (only loaded for speculative decoding)
    other: int = 0  # final norms and anything unclassified

    @property
    def stored(self) -> int:
        return (
            self.decoder_layers
            + self.experts
            + self.embeddings
            + self.lm_head
            + self.per_layer_embeddings
            + self.per_layer_compute
            + self.unused_shared_kv
            + self.tied_duplicate
            + self.vision
            + self.audio
            + self.mtp
            + self.other
        )


class ModelSpec(BaseModel):
    model_config = ConfigDict(frozen=True)

    name: str
    revision: str | None = None
    model_type: str
    hidden_size: int
    vocab_size: int
    tied_embeddings: bool
    max_context: int | None = None
    layers: tuple[LayerSpec, ...]
    moe: MoESpec | None = None
    params: ParamBreakdown

    @property
    def num_layers(self) -> int:
        return len(self.layers)

    def count(self, kind: LayerKind) -> int:
        return sum(1 for layer in self.layers if layer.kind == kind)

    @property
    def lm_head_params(self) -> int:
        """The output projection is a real matmul even when its weights are tied to the embeddings."""
        return self.params.lm_head if self.params.lm_head else self.vocab_size * self.hidden_size

    @property
    def active_params(self) -> int:
        """Params whose weights are read for one decode step of one sequence.

        Excludes lookup-only tables (input embeddings, PLE), unused weights, and non-text towers;
        includes the lm_head matmul and, for MoE, only the routed share of the experts.
        """
        experts = round(self.params.experts * self.moe.active_fraction) if self.moe else self.params.experts
        return self.params.decoder_layers + experts + self.params.per_layer_compute + self.lm_head_params
