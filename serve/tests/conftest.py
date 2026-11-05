from __future__ import annotations

from pathlib import Path

import pytest
from cortex_serve.memcalc.loader import load_local
from cortex_serve.memcalc.model import ModelSpec

MODELS = Path(__file__).parent / "fixtures" / "models"
REAL_MODELS = sorted(p.name for p in MODELS.iterdir() if not p.name.startswith("synthetic"))


def spec(name: str) -> ModelSpec:
    return load_local(MODELS / name)


@pytest.fixture(scope="session")
def models_dir() -> Path:
    return MODELS
