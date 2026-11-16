"""GGUF header parsing against a synthetic file (real model files are too big for CI)."""

from __future__ import annotations

import struct
from pathlib import Path

import pytest
from cortex_serve.memcalc.calc import estimate, gguf_active_bytes, gguf_weight_bytes
from cortex_serve.memcalc.devices import get_device
from cortex_serve.memcalc.gguf import read_header

from .conftest import spec


def _s(text: str) -> bytes:
    raw = text.encode()
    return struct.pack("<Q", len(raw)) + raw


def write_gguf(path: Path, tensors: list[tuple[str, tuple[int, ...], int]]) -> None:
    """Minimal GGUF v3: 2 metadata keys (a string and a u32 array) + tensor infos (no tensor data needed)."""
    out = b"GGUF" + struct.pack("<IQQ", 3, len(tensors), 2)
    out += _s("general.architecture") + struct.pack("<I", 8) + _s("qwen3")
    out += _s("tokenizer.ggml.token_type") + struct.pack("<IIQ", 9, 4, 3) + struct.pack("<3I", 1, 1, 3)
    for offset, (name, shape, type_id) in enumerate(tensors):
        out += _s(name) + struct.pack("<I", len(shape)) + b"".join(struct.pack("<Q", d) for d in shape)
        out += struct.pack("<IQ", type_id, offset)
    path.write_bytes(out)


def test_exact_bytes_by_ggml_type(tmp_path: Path) -> None:
    f = tmp_path / "m.gguf"
    write_gguf(
        f,
        [
            ("token_embd.weight", (1024, 256), 14),  # Q6_K: 262144 / 256 * 210 = 215040 B
            ("blk.0.attn_q.weight", (1024, 1024), 12),  # Q4_K: 1048576 / 256 * 144 = 589824 B
            ("blk.0.ffn_down.weight", (64, 32), 2),  # Q4_0: 2048 / 32 * 18 = 1152 B
            ("blk.0.attn_norm.weight", (1024,), 0),  # F32: 4096 B
        ],
    )
    h = read_header(f)
    assert h.version == 3 and h.metadata["general.architecture"] == "qwen3"
    assert h.metadata["tokenizer.ggml.token_type"] == "<array of 3>"
    assert h.bytes_by_type() == {"Q6_K": 215040, "Q4_K": 589824, "Q4_0": 1152, "F32": 4096}
    assert h.weight_bytes == 810112
    assert h.n_params == 262144 + 1048576 + 2048 + 1024
    assert read_header(f, keep_arrays=True).metadata["tokenizer.ggml.token_type"] == [1, 1, 3]


def test_ple_offload_and_active_bytes(tmp_path: Path) -> None:
    f = tmp_path / "g.gguf"
    write_gguf(
        f,
        [
            ("per_layer_token_embd.weight", (256, 1024), 14),  # lookup-only table
            ("token_embd.weight", (256, 512), 8),
            ("blk.0.attn_q.weight", (256, 256), 8),
        ],
    )
    h = read_header(f)
    ple = 256 * 1024 // 256 * 210
    assert gguf_weight_bytes(h, ple_offload=True) == h.weight_bytes - ple
    # tied model (no output.weight): token_embd is read for the output projection, PLE is not
    assert gguf_active_bytes(h, spec("Qwen__Qwen3-0.6B")) == h.weight_bytes - ple


def test_estimate_uses_gguf(tmp_path: Path) -> None:
    f = tmp_path / "q.gguf"
    write_gguf(f, [("token_embd.weight", (1024, 1024), 8), ("blk.0.attn_q.weight", (1024, 1024), 8)])
    est = estimate(spec("Qwen__Qwen3-0.6B"), get_device("local-epyc-7571"), "q8_0", gguf=read_header(f))
    assert est.weights_gib * 2**30 == pytest.approx(2 * 1024 * 1024 // 32 * 34, abs=1)
    assert any("GGUF header" in n for n in est.notes)


def test_rejects_non_gguf(tmp_path: Path) -> None:
    f = tmp_path / "x.gguf"
    f.write_bytes(b"NOPE")
    with pytest.raises(ValueError):
        read_header(f)
