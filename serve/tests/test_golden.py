"""Golden vectors: hand calculations that memcalc (and later the TS port) must reproduce."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
from cortex_serve.memcalc.calc import (
    estimate,
    fixed_state_bytes,
    kv_bytes_per_token,
    seq_bytes,
    weights_bytes,
)
from cortex_serve.memcalc.devices import get_device

from .conftest import spec

GOLDEN: dict[str, Any] = json.loads((Path(__file__).parent / "fixtures" / "golden.json").read_text())


@pytest.mark.parametrize("case", GOLDEN["cases"], ids=lambda c: c["id"])
def test_memory_cases(case: dict[str, Any]) -> None:
    s = spec(case["model"])
    exp = case["expect"]
    assert seq_bytes(s, case["ctx"], case["kv_dtype"]) == exp["seq_bytes"]
    if "kv_bytes_per_token" in exp:
        assert kv_bytes_per_token(s, case["kv_dtype"]) == exp["kv_bytes_per_token"]
    if "fixed_state_bytes" in exp:
        assert fixed_state_bytes(s) == exp["fixed_state_bytes"]
    if "params_stored" in exp:
        assert s.params.stored == exp["params_stored"]


@pytest.mark.parametrize("case", GOLDEN["estimates"], ids=lambda c: c["id"])
def test_estimate_cases(case: dict[str, Any]) -> None:
    s = spec(case["model"])
    exp = case["expect"]
    est = estimate(s, get_device(case["device"]), case["dtype"], case["kv_dtype"], case["ctx"], case["util"])
    if "weights_bytes" in exp:
        assert weights_bytes(s, case["dtype"]) == exp["weights_bytes"]
    if "max_concurrent_seqs" in exp:
        assert est.max_concurrent_seqs == exp["max_concurrent_seqs"]
    if "decode_ceiling_tok_s" in exp:
        assert est.decode_ceiling_tok_s == pytest.approx(exp["decode_ceiling_tok_s"], rel=exp["tolerance"])
