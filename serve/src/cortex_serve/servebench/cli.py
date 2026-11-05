"""servebench CLI."""

from __future__ import annotations

from pathlib import Path
from typing import Annotated

import typer

from cortex_serve.servebench.config import load_config
from cortex_serve.servebench.quality.run import run_quality
from cortex_serve.servebench.report import write_report
from cortex_serve.servebench.runner import run

app = typer.Typer(no_args_is_help=True, add_completion=False, help="Engine-agnostic LLM serving benchmark.")

REPO = Path(__file__).resolve().parents[4]


@app.command("run")
def run_cmd(
    config: Annotated[Path, typer.Argument(help="Benchmark config YAML")],
    models_dir: Annotated[Path, typer.Option("--models-dir", help="Directory with downloaded model artifacts")],
    out: Annotated[Path, typer.Option("--out", help="Results root")] = REPO / "serve" / "results",
    artifact: Annotated[list[str] | None, typer.Option("--artifact", "-a", help="Only these artifacts")] = None,
) -> None:
    """Run a benchmark config; writes manifest.json, points.parquet, requests.parquet and engine logs."""
    run_dir = run(load_config(config), out, models_dir.resolve(), REPO, artifact)
    typer.echo(str(run_dir))


@app.command("report")
def report_cmd(
    run_dirs: Annotated[list[Path], typer.Argument(help="One or more run directories")],
    out: Annotated[Path, typer.Option("--out", help="Report directory")],
    title: Annotated[str, typer.Option("--title")] = "servebench report",
) -> None:
    """Aggregate runs (median over repeats) into report.md, summary.csv and plots."""
    typer.echo(str(write_report(run_dirs, out, title)))


TOOLS_IMAGE = "ghcr.io/ggml-org/llama.cpp:full@sha256:93573d882ea21d1b77712a33162934daaabcdf3a076dfff91a9ed3c2aed8cd41"


@app.command("quality")
def quality_cmd(
    config: Annotated[Path, typer.Argument(help="Benchmark config YAML (its artifacts are evaluated)")],
    models_dir: Annotated[Path, typer.Option("--models-dir")],
    out: Annotated[Path, typer.Option("--out")] = REPO / "serve" / "results",
    arc_n: Annotated[int, typer.Option("--arc-n", help="ARC-Challenge questions")] = 100,
    ppl_chunks: Annotated[int, typer.Option("--ppl-chunks", help="512-token WikiText-2 chunks")] = 20,
    tools_image: Annotated[str, typer.Option("--tools-image")] = TOOLS_IMAGE,
    artifact: Annotated[list[str] | None, typer.Option("--artifact", "-a")] = None,
) -> None:
    """ARC-Challenge accuracy (via the serving API) + WikiText-2 perplexity (llama-perplexity) per artifact."""
    d = run_quality(load_config(config), out, models_dir.resolve(), REPO, tools_image, arc_n, ppl_chunks, artifact)
    typer.echo(str(d))
