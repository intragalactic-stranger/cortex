"""Parameter counting: the analytic formula must equal the exact safetensors-header count."""

from __future__ import annotations

import gzip
import json

import pytest
from cortex_serve.memcalc.families import build_layers
from cortex_serve.memcalc.loader import _read_json
from cortex_serve.memcalc.params import analytic_params, check_shapes

from .conftest import MODELS, REAL_MODELS, spec

DENSE_OR_MOE = ["Qwen__Qwen3-0.6B", "Qwen__Qwen3-1.7B", "Qwen__Qwen3-4B-Instruct-2507", "Qwen__Qwen3-30B-A3B"]


@pytest.mark.parametrize("name", DENSE_OR_MOE)
def test_analytic_matches_header_exactly(name: str) -> None:
    analytic = analytic_params(_read_json(MODELS / name / "config.json"))
    assert analytic is not None
    header = spec(name).params
    assert analytic.stored == header.stored - header.tied_duplicate
    assert analytic.experts == header.experts
    assert analytic.embeddings == header.embeddings
    assert analytic.lm_head == header.lm_head


def test_redundant_tied_lm_head_detected() -> None:
    # Qwen3-0.6B: config says tie_word_embeddings=true, but the checkpoint also stores lm_head.weight
    p = spec("Qwen__Qwen3-0.6B").params
    assert p.tied_duplicate == 151_936 * 1_024
    assert p.stored - p.tied_duplicate == 596_049_920  # the real unique parameter count
    assert spec("Qwen__Qwen3-4B-Instruct-2507").params.tied_duplicate == 0  # 4B does not duplicate it


def test_llama_8b_analytic_total() -> None:
    assert spec("synthetic__llama-3.1-8b").params.stored == 8_030_261_248


@pytest.mark.parametrize("name", ["Qwen__Qwen3.5-4B", "google__gemma-4-E4B-it"])
def test_no_analytic_formula_for_hybrids(name: str) -> None:
    assert analytic_params(_read_json(MODELS / name / "config.json")) is None


@pytest.mark.parametrize("name", REAL_MODELS)
def test_config_interpretation_matches_tensor_shapes(name: str) -> None:
    cfg = _read_json(MODELS / name / "config.json")
    tensors = {
        k: (v[0], v[1])
        for k, v in json.loads(gzip.decompress((MODELS / name / "tensors.json.gz").read_bytes())).items()
    }
    assert check_shapes(tensors, build_layers(cfg)) == []


def test_gemma4_e4b_breakdown() -> None:
    p = spec("google__gemma-4-E4B-it").params
    assert p.per_layer_embeddings == 262_144 * 10_752  # 42 layers x 256 dims, one row per token id
    assert p.embeddings == 262_144 * 2_560
    assert p.lm_head == 0  # tied
    assert p.vision > 0 and p.audio > 0
    assert p.unused_shared_kv > 0  # k/v weights stored in the 18 KV-shared layers but never used
    assert p.other < 10_000


def test_qwen35_breakdown() -> None:
    p = spec("Qwen__Qwen3.5-4B").params
    assert p.vision > 300_000_000
    assert p.mtp > 100_000_000  # multi-token-prediction head, only needed for speculative decoding


def test_moe_active_params() -> None:
    s = spec("Qwen__Qwen3-30B-A3B")
    assert s.moe is not None and s.moe.num_experts == 128 and s.moe.experts_per_token == 8
    assert 3.0e9 < s.active_params < 3.6e9  # "A3B": ~3.3B active
