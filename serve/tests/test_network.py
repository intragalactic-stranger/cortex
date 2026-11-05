"""Opt-in: compare live Hub data with the vendored fixtures (detects upstream config drift).

Run with: uv run pytest -m network
"""

from __future__ import annotations

import pytest
from cortex_serve.memcalc.loader import load_hub

from .conftest import spec


@pytest.mark.network
def test_live_hub_matches_fixture() -> None:
    live = load_hub("Qwen/Qwen3-0.6B", revision="c1899de289a04d12100db370d81485cdf75e47ca")
    fixture = spec("Qwen__Qwen3-0.6B")
    assert live.params == fixture.params
    assert live.layers == fixture.layers
