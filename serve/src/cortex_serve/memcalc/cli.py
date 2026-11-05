"""memcalc command-line interface."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.table import Table

from cortex_serve.memcalc.calc import Components, Estimate, estimate
from cortex_serve.memcalc.devices import get_device, load_devices
from cortex_serve.memcalc.gguf import read_header
from cortex_serve.memcalc.loader import load_model

app = typer.Typer(
    no_args_is_help=True,
    add_completion=False,
    help="Predict LLM inference memory: weights, KV cache, max concurrency and decode ceilings.",
)
console = Console()

ModelOpt = Annotated[str, typer.Option("--model", "-m", help="HF repo id (org/name) or a local model dir/config")]
DeviceOpt = Annotated[str, typer.Option("--device", "-d", help="Device name; see `memcalc devices`")]
DtypeOpt = Annotated[str, typer.Option("--dtype", help="Weight format: bf16, fp8, w4a16-g128, q4_k_m, ...")]
KvOpt = Annotated[str, typer.Option("--kv-dtype", help="KV cache format: auto, bf16, fp8, q8_0, ...")]
UtilOpt = Annotated[float, typer.Option("--util", help="Fraction of device memory the engine may use")]
ReserveOpt = Annotated[
    float | None, typer.Option("--reserve-gib", help="Runtime reserve (CUDA ctx, graphs, activations)")
]
ComponentsOpt = Annotated[str, typer.Option("--components", help="'text' (language model) or 'all' (+vision/audio)")]
JsonOpt = Annotated[bool, typer.Option("--json", help="Emit JSON instead of a table")]


def _fmt(value: float | None, spec: str = ",.1f") -> str:
    return "n/a" if value is None else format(value, spec)


def _components(value: str) -> Components:
    if value not in ("text", "all"):
        raise typer.BadParameter("components must be 'text' or 'all'")
    return "text" if value == "text" else "all"


def _render(est: Estimate) -> None:
    t = Table(title=f"{est.model} on {est.device}", show_header=False, title_justify="left")
    t.add_column(style="bold")
    t.add_column(justify="right")
    rows = [
        ("weights / KV format", f"{est.weight_format} / {est.kv_format}"),
        ("layers", ", ".join(f"{n} {k}" for k, n in est.layer_kinds.items())),
        (
            "params stored / loaded / active",
            f"{est.params_stored / 1e9:.2f}B / {est.params_loaded / 1e9:.2f}B / "
            f"{est.params_active / 1e9:.2f}B  ({est.param_source})",
        ),
        ("weight memory", f"{est.weights_gib:.2f} GiB"),
        ("KV per token (full-attention layers)", f"{est.kv_bytes_per_token / 1024:,.1f} KiB"),
        ("fixed state per sequence", f"{est.fixed_state_mib_per_seq:,.1f} MiB"),
        (f"memory per sequence @ {est.context:,} tokens", f"{est.seq_mib:,.1f} MiB"),
        (
            f"budget ({est.util:.0%}) - weights - reserve ({est.reserve_gib} GiB)",
            f"{est.free_for_kv_gib:.2f} GiB",
        ),
        ("max concurrent sequences", f"{est.max_concurrent_seqs:,}"),
        ("decode ceiling, 1 stream (weights only)", f"{_fmt(est.decode_ceiling_tok_s)} tok/s"),
        (f"decode ceiling, 1 stream @ {est.context:,} ctx", f"{_fmt(est.decode_ceiling_at_ctx_tok_s)} tok/s"),
        ("batch where decode turns compute-bound", _fmt(est.roofline_crossover_batch, ",.0f")),
        ("prefill compute per 1K tokens (linear part)", f"{est.prefill_tflop_per_1k_tokens:.2f} TFLOP"),
        (f"attention per token @ {est.context:,} ctx", f"{est.attention_gflop_per_token_at_ctx:.3f} GFLOP"),
    ]
    for k, v in rows:
        t.add_row(k, v)
    console.print(t)
    for note in est.notes:
        console.print(f"  [dim]• {note}[/dim]")


@app.command("estimate")
def estimate_cmd(
    model: ModelOpt,
    device: DeviceOpt = "L4",
    dtype: DtypeOpt = "bf16",
    kv_dtype: KvOpt = "auto",
    ctx: Annotated[int, typer.Option("--ctx", help="Context length per sequence")] = 8192,
    util: UtilOpt = 0.9,
    reserve_gib: ReserveOpt = None,
    components: ComponentsOpt = "text",
    ple_offload: Annotated[bool, typer.Option("--ple-offload", help="Gemma PLE tables kept off-device")] = False,
    with_mtp: Annotated[bool, typer.Option("--with-mtp", help="Load multi-token-prediction heads")] = False,
    gguf: Annotated[
        Path | None, typer.Option("--gguf", help="GGUF file: exact weight bytes from its header (llama.cpp tier)")
    ] = None,
    revision: Annotated[str | None, typer.Option("--revision", help="HF branch, tag or commit")] = None,
    as_json: JsonOpt = False,
) -> None:
    """Estimate memory and throughput ceilings for one model on one device."""
    spec = load_model(model, revision)
    est = estimate(
        spec,
        get_device(device),
        dtype,
        kv_dtype,
        ctx,
        util,
        reserve_gib,
        _components(components),
        ple_offload,
        with_mtp,
        read_header(gguf) if gguf else None,
    )
    if as_json:
        typer.echo(est.model_dump_json(indent=2))
    else:
        _render(est)


@app.command("compare")
def compare_cmd(
    model: Annotated[list[str], typer.Option("--model", "-m", help="Repeat for each model")],
    ctx: Annotated[list[int] | None, typer.Option("--ctx", help="Repeat for each context length")] = None,
    device: DeviceOpt = "L4",
    dtype: DtypeOpt = "bf16",
    kv_dtype: KvOpt = "auto",
    util: UtilOpt = 0.9,
    as_json: JsonOpt = False,
) -> None:
    """Compare models side by side (the E15 architecture-study table)."""
    contexts = ctx or [8192]
    dev = get_device(device)
    specs = [load_model(m) for m in model]
    results = [estimate(s, dev, dtype, kv_dtype, c, util) for s in specs for c in contexts]
    if as_json:
        typer.echo(json.dumps([r.model_dump() for r in results], indent=2))
        return
    t = Table(title=f"Memory per sequence and max concurrency on {dev.name} ({dtype} weights, {kv_dtype} KV)")
    for col in ("model", "layers", "weights GiB", "KV/token KiB", "fixed state MiB"):
        t.add_column(col, justify="right" if col != "model" else "left")
    for c in contexts:
        t.add_column(f"MiB/seq @{c // 1024}K", justify="right")
        t.add_column(f"max seqs @{c // 1024}K", justify="right")
    for i, s in enumerate(specs):
        row = results[i * len(contexts) : (i + 1) * len(contexts)]
        first = row[0]
        cells = [
            s.name,
            ", ".join(f"{n} {k}" for k, n in first.layer_kinds.items()),
            f"{first.weights_gib:.2f}",
            f"{first.kv_bytes_per_token / 1024:.1f}",
            f"{first.fixed_state_mib_per_seq:.1f}",
        ]
        for r in row:
            cells += [f"{r.seq_mib:,.0f}", f"{r.max_concurrent_seqs:,}"]
        t.add_row(*cells)
    console.print(t)


@app.command("devices")
def devices_cmd() -> None:
    """List known devices (approximate vendor figures)."""
    t = Table(title="Devices (approximate; see devices.yaml for sources)")
    for col in ("name", "kind", "memory GiB", "bandwidth GB/s", "bf16 TFLOPS", "fp8 TFLOPS"):
        t.add_column(col, justify="left" if col in ("name", "kind") else "right")
    for d in load_devices().values():
        t.add_row(
            d.name,
            d.kind,
            f"{d.memory_gib:g}",
            _fmt(d.bandwidth_gbs, ",.0f"),
            _fmt(d.bf16_tflops, ",.0f"),
            _fmt(d.fp8_tflops, ",.0f"),
        )
    console.print(t)
