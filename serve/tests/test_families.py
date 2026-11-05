"""Config -> per-layer memory specs."""

from __future__ import annotations

import pytest
from cortex_serve.memcalc.families import UnsupportedArchitectureError, build_layers
from cortex_serve.memcalc.model import LayerKind

from .conftest import spec


@pytest.mark.parametrize(
    ("name", "counts"),
    [
        ("Qwen__Qwen3-4B-Instruct-2507", {LayerKind.FULL: 36}),
        ("Qwen__Qwen3-30B-A3B", {LayerKind.FULL: 48}),
        ("Qwen__Qwen3.5-4B", {LayerKind.FULL: 8, LayerKind.LINEAR: 24}),
        ("google__gemma-4-E4B-it", {LayerKind.FULL: 4, LayerKind.SLIDING: 20, LayerKind.SHARED: 18}),
        ("google__gemma-4-E2B-it", {LayerKind.FULL: 3, LayerKind.SLIDING: 12, LayerKind.SHARED: 20}),
    ],
)
def test_layer_kind_counts(name: str, counts: dict[LayerKind, int]) -> None:
    s = spec(name)
    assert {k: s.count(k) for k in LayerKind if s.count(k)} == counts


def test_gemma4_kv_sharing_producers() -> None:
    layers = spec("google__gemma-4-E4B-it").layers
    # first shared layer is 42 - 18 = 24; producers are the last non-shared layer of each type
    assert layers[23].kind == LayerKind.FULL and layers[22].kind == LayerKind.SLIDING
    for layer in layers[24:]:
        assert layer.kind == LayerKind.SHARED
        assert layer.shares_from in (22, 23)
    assert layers[29].shares_from == 23  # a global layer reuses the last global producer
    assert layers[24].shares_from == 22


def test_gemma4_global_layers_use_global_head_dim() -> None:
    layers = spec("google__gemma-4-E4B-it").layers
    assert layers[5].kind == LayerKind.FULL and layers[5].head_dim == 512
    assert layers[0].kind == LayerKind.SLIDING and layers[0].head_dim == 256 and layers[0].window == 512


def test_qwen35_state_is_fp32() -> None:
    linear = [x for x in spec("Qwen__Qwen3.5-4B").layers if x.kind == LayerKind.LINEAR]
    assert all(x.state_dtype_bytes == 4.0 for x in linear)
    assert linear[0].state_elems == 32 * 128 * 128


def test_unknown_layer_type_rejected() -> None:
    cfg = {
        "hidden_size": 64,
        "num_attention_heads": 4,
        "num_hidden_layers": 2,
        "layer_types": ["mamba", "attention"],
    }
    with pytest.raises(UnsupportedArchitectureError):
        build_layers(cfg)
