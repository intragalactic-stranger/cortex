"""M3 pieces that run without llm-compressor or a GPU: calibration split, recipe registry, spend reservations."""

from __future__ import annotations

import json
import os
from pathlib import Path

import pytest
from cortex_serve.quantize.calib import ids_digest, split_bfsi
from cortex_serve.quantize.recipes import RECIPES, build
from cortex_serve.spend import inflight_usd


def _row(title: str, part: str, sec: int, text: str = "x" * 300) -> dict[str, str]:
    return {
        "title_number": title,
        "part_number": part,
        "rag_id": f"ecfr:{title}:{part}:{sec}",
        "section_subject": f"§ {part}.{sec}",
        "section_text": text,
    }


def test_bfsi_split_is_part_disjoint_and_deterministic() -> None:
    rows = [_row(t, str(p), s) for t in ("12", "17") for p in range(40) for s in range(5)]
    rows += [_row("26", "1", 1), _row("12", "99", 1, text="short")]  # other title, too short: excluded
    a = split_bfsi(rows, n_calib=100, n_heldout=30)
    b = split_bfsi(list(reversed(rows)), n_calib=100, n_heldout=30)
    assert a == b and len(a.calib) == 100 and len(a.heldout) == 30

    def parts(ids: list[str]) -> set[str]:
        return {":".join(i.split(":")[1:3]) for i in ids}

    assert not parts(a.calib_ids) & parts(a.heldout_ids)
    assert all(not i.startswith("ecfr:26") for i in a.calib_ids + a.heldout_ids)
    assert ids_digest(a.calib_ids) == ids_digest(b.calib_ids)


def test_recipe_registry() -> None:
    assert {"fp8-dynamic", "w8a8-int8", "w4a16-gptq-g128", "w4a16-awq-asym", "w4a16-gptq-g32-sym"} <= set(RECIPES)
    assert not RECIPES["fp8-dynamic"].needs_calibration and RECIPES["w4a16-gptq-g128"].needs_calibration
    with pytest.raises((KeyError, ImportError)):
        build("nope")


def test_inflight_counts_live_reservations_only(tmp_path: Path) -> None:
    (tmp_path / "live.json").write_text(json.dumps({"pid": os.getpid(), "worst_usd": 1.5}))
    (tmp_path / "dead.json").write_text(json.dumps({"pid": 2**22 + 12345, "worst_usd": 9.0}))
    assert inflight_usd(tmp_path) == 1.5
    assert not (tmp_path / "dead.json").exists()
