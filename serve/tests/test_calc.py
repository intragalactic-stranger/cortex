"""Properties of the arithmetic that must hold for any model."""

from __future__ import annotations

import pytest
from cortex_serve.memcalc.calc import estimate, seq_bytes, weights_bytes
from cortex_serve.memcalc.devices import get_device

from .conftest import REAL_MODELS, spec


@pytest.mark.parametrize("name", REAL_MODELS)
def test_memory_monotonic_in_context(name: str) -> None:
    s = spec(name)
    sizes = [seq_bytes(s, c) for c in (1, 256, 512, 1024, 8192, 32768)]
    assert sizes == sorted(sizes)


@pytest.mark.parametrize("name", REAL_MODELS)
def test_fp8_kv_never_larger(name: str) -> None:
    s = spec(name)
    assert seq_bytes(s, 8192, "fp8") <= seq_bytes(s, 8192, "bf16")


def test_sliding_layers_stop_growing_past_window() -> None:
    s = spec("google__gemma-4-E4B-it")
    growth_after_window = seq_bytes(s, 2048) - seq_bytes(s, 1024)
    full_only = 4 * 2 * 2 * 512 * 2 * 1024  # 4 global layers x 1024 extra tokens
    assert growth_after_window == full_only


def test_quantized_weights_keep_embeddings_16bit() -> None:
    s = spec("Qwen__Qwen3-4B-Instruct-2507")
    w4 = weights_bytes(s, "w4a16-g128")
    gguf = weights_bytes(s, "q4_0")
    assert w4 > gguf  # llm-compressor keeps the 389M-param embedding in bf16; GGUF quantizes it
    assert weights_bytes(s, "bf16") > weights_bytes(s, "fp8") > w4


def test_ple_offload_and_components() -> None:
    s = spec("google__gemma-4-E4B-it")
    base = weights_bytes(s, "bf16")
    assert weights_bytes(s, "bf16", ple_offload=True) == base - 2 * s.params.per_layer_embeddings
    assert weights_bytes(s, "bf16", components="all") > base


def test_model_that_does_not_fit() -> None:
    est = estimate(spec("Qwen__Qwen3-30B-A3B"), get_device("L4"), "bf16")
    assert est.max_concurrent_seqs == 0
    assert any("does not fit" in n for n in est.notes)


def test_fp8_on_gpu_without_fp8_compute_is_flagged() -> None:
    est = estimate(spec("Qwen__Qwen3-0.6B"), get_device("A10G"), "fp8")
    assert any("no native FP8" in n for n in est.notes)
    assert est.roofline_crossover_batch is not None  # falls back to the 16-bit peak


def test_lower_bit_weights_turn_compute_bound_sooner() -> None:
    s = spec("Qwen__Qwen3-4B-Instruct-2507")
    l4 = get_device("L4")
    bf16 = estimate(s, l4, "bf16").roofline_crossover_batch
    int4 = estimate(s, l4, "w4a16-g128").roofline_crossover_batch
    assert bf16 is not None and int4 is not None
    assert bf16 == pytest.approx(403.3, rel=0.01)  # 121 TFLOPS / 300 GB/s
    assert int4 < bf16 / 3


def test_cpu_device_uses_measured_bandwidth() -> None:
    est = estimate(spec("Qwen__Qwen3-0.6B"), get_device("local-epyc-7571"), "q4_k_m")
    assert est.decode_ceiling_tok_s is not None and 40 < est.decode_ceiling_tok_s < 80  # 20.3 GB/s / ~0.36 GB
    assert est.roofline_crossover_batch is None  # no FLOPS figure for this CPU
    assert est.max_concurrent_seqs > 0


@pytest.mark.parametrize("bad", [{"ctx": 0}, {"util": 0.0}, {"util": 1.5}])
def test_invalid_inputs(bad: dict[str, float]) -> None:
    with pytest.raises(ValueError):
        estimate(spec("Qwen__Qwen3-0.6B"), get_device("L4"), **bad)  # type: ignore[arg-type]
