"""Read a GGUF file's header: metadata + every tensor's ggml type and shape, giving exact weight bytes.

GGUF k-quant "mixes" (Q4_K_M etc.) keep some tensors at higher precision, so an average bits-per-weight
is only an approximation (e.g. Qwen3-0.6B Q4_K_M is 5.24 BPW, Gemma 4 E2B QAT Q4_0 is 5.76 BPW).
Reading the header (a few MB at the start of the file) makes memcalc exact for any GGUF.

Format reference: ggml/docs/gguf.md (v3). Block sizes from ggml.c ``type_traits``.
"""

from __future__ import annotations

import struct
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import Any, BinaryIO

# ggml_type id -> (name, elements per block, bytes per block)
GGML_TYPES: dict[int, tuple[str, int, int]] = {
    0: ("F32", 1, 4),
    1: ("F16", 1, 2),
    2: ("Q4_0", 32, 18),
    3: ("Q4_1", 32, 20),
    6: ("Q5_0", 32, 22),
    7: ("Q5_1", 32, 24),
    8: ("Q8_0", 32, 34),
    9: ("Q8_1", 32, 36),
    10: ("Q2_K", 256, 84),
    11: ("Q3_K", 256, 110),
    12: ("Q4_K", 256, 144),
    13: ("Q5_K", 256, 176),
    14: ("Q6_K", 256, 210),
    15: ("Q8_K", 256, 292),
    16: ("IQ2_XXS", 256, 66),
    17: ("IQ2_XS", 256, 74),
    18: ("IQ3_XXS", 256, 98),
    19: ("IQ1_S", 256, 50),
    20: ("IQ4_NL", 32, 18),
    21: ("IQ3_S", 256, 110),
    22: ("IQ2_S", 256, 82),
    23: ("IQ4_XS", 256, 136),
    24: ("I8", 1, 1),
    25: ("I16", 1, 2),
    26: ("I32", 1, 4),
    27: ("I64", 1, 8),
    28: ("F64", 1, 8),
    29: ("IQ1_M", 256, 56),
    30: ("BF16", 1, 2),
    34: ("TQ1_0", 256, 54),
    35: ("TQ2_0", 256, 66),
    39: ("MXFP4", 32, 17),
}

# metadata value types
_SCALARS = {0: "<B", 1: "<b", 2: "<H", 3: "<h", 4: "<I", 5: "<i", 6: "<f", 7: "<?", 10: "<Q", 11: "<q", 12: "<d"}
_STRING, _ARRAY = 8, 9


@dataclass(frozen=True)
class GGUFTensor:
    name: str
    ggml_type: str
    shape: tuple[int, ...]
    nbytes: int

    @property
    def n_elements(self) -> int:
        n = 1
        for d in self.shape:
            n *= d
        return n


@dataclass(frozen=True)
class GGUFHeader:
    version: int
    metadata: dict[str, Any]
    tensors: tuple[GGUFTensor, ...]

    @property
    def weight_bytes(self) -> int:
        return sum(t.nbytes for t in self.tensors)

    @property
    def n_params(self) -> int:
        return sum(t.n_elements for t in self.tensors)

    @property
    def bits_per_weight(self) -> float:
        return 8 * self.weight_bytes / self.n_params

    def bytes_by_type(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for t in self.tensors:
            out[t.ggml_type] = out.get(t.ggml_type, 0) + t.nbytes
        return out

    def bytes_matching(self, *substrings: str) -> int:
        return sum(t.nbytes for t in self.tensors if any(s in t.name for s in substrings))


def _read(f: BinaryIO, fmt: str) -> Any:
    size = struct.calcsize(fmt)
    data = f.read(size)
    if len(data) != size:
        raise ValueError("truncated GGUF header")
    return struct.unpack(fmt, data)[0]


def _string(f: BinaryIO) -> str:
    n = _read(f, "<Q")
    return str(f.read(n).decode("utf-8", errors="replace"))


def _value(f: BinaryIO, vtype: int, keep_arrays: bool) -> Any:
    if vtype in _SCALARS:
        return _read(f, _SCALARS[vtype])
    if vtype == _STRING:
        return _string(f)
    if vtype == _ARRAY:
        item_type = _read(f, "<I")
        count = _read(f, "<Q")
        if not keep_arrays and item_type in _SCALARS:  # skip big arrays (e.g. token scores) cheaply
            f.seek(count * struct.calcsize(_SCALARS[item_type]), 1)
            return f"<array of {count}>"
        items = [_value(f, item_type, keep_arrays) for _ in range(count)]
        return items if keep_arrays else f"<array of {count}>"
    raise ValueError(f"unknown GGUF metadata type {vtype}")


def _tensor_infos(f: BinaryIO, count: int) -> Iterator[GGUFTensor]:
    for _ in range(count):
        name = _string(f)
        n_dims = _read(f, "<I")
        shape = tuple(_read(f, "<Q") for _ in range(n_dims))
        type_id = _read(f, "<I")
        _offset = _read(f, "<Q")
        if type_id not in GGML_TYPES:
            raise ValueError(f"{name}: unknown ggml type id {type_id}")
        type_name, block, block_bytes = GGML_TYPES[type_id]
        n = 1
        for d in shape:
            n *= d
        if n % block:
            raise ValueError(f"{name}: {n} elements not divisible by block size {block}")
        yield GGUFTensor(name, type_name, shape, n // block * block_bytes)


def read_header(path: Path, keep_arrays: bool = False) -> GGUFHeader:
    with path.open("rb") as f:
        if f.read(4) != b"GGUF":
            raise ValueError(f"{path} is not a GGUF file")
        version = _read(f, "<I")
        n_tensors = _read(f, "<Q")
        n_kv = _read(f, "<Q")
        metadata = {}
        for _ in range(n_kv):
            key = _string(f)
            metadata[key] = _value(f, _read(f, "<I"), keep_arrays)
        tensors = tuple(_tensor_infos(f, n_tensors))
    return GGUFHeader(version, metadata, tensors)
