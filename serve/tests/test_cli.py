from __future__ import annotations

import json
from pathlib import Path

from cortex_serve.memcalc.cli import app
from typer.testing import CliRunner

runner = CliRunner()


def test_estimate_json(models_dir: Path) -> None:
    result = runner.invoke(app, ["estimate", "-m", str(models_dir / "Qwen__Qwen3-4B-Instruct-2507"), "--json"])
    assert result.exit_code == 0, result.output
    data = json.loads(result.output)
    assert data["max_concurrent_seqs"] == 9
    assert data["param_source"] == "safetensors-header"


def test_estimate_table(models_dir: Path) -> None:
    result = runner.invoke(app, ["estimate", "-m", str(models_dir / "google__gemma-4-E2B-it"), "-d", "T4"])
    assert result.exit_code == 0, result.output
    assert "max concurrent sequences" in result.output


def test_compare(models_dir: Path) -> None:
    args = ["compare", "--ctx", "8192", "--ctx", "32768", "--json"]
    for name in ("Qwen__Qwen3-4B-Instruct-2507", "Qwen__Qwen3.5-4B", "google__gemma-4-E4B-it"):
        args += ["-m", str(models_dir / name)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert len(json.loads(result.output)) == 6


def test_devices() -> None:
    result = runner.invoke(app, ["devices"])
    assert result.exit_code == 0
    assert "H100-80GB" in result.output


def test_unknown_device(models_dir: Path) -> None:
    result = runner.invoke(app, ["estimate", "-m", str(models_dir / "Qwen__Qwen3-0.6B"), "-d", "TPU"])
    assert result.exit_code != 0
