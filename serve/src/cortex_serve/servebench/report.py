"""Turn ``points.parquet`` files into a markdown summary and plots.

Repeats are aggregated as the median, with the min-max range shown as the spread.
"""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import polars as pl

KEYS = ["artifact", "workload", "level"]


def load_points(run_dirs: list[Path]) -> pl.DataFrame:
    frames = [pl.read_parquet(d / "points.parquet") for d in run_dirs if (d / "points.parquet").exists()]
    if not frames:
        raise FileNotFoundError("no points.parquet in the given run directories")
    return pl.concat(frames, how="diagonal_relaxed")


def aggregate(points: pl.DataFrame) -> pl.DataFrame:
    def med(col: str) -> pl.Expr:
        return pl.col(col).median().alias(col)

    # CPU runs sample cgroup RAM, GPU runs sample nvidia-smi memory; report whichever the runs have
    mem_cols = [c for c in ("res_peak_memory_mib", "res_gpu_mem_peak_mib") if c in points.columns]
    peak = pl.max_horizontal([pl.col(c) for c in mem_cols]).max() if mem_cols else pl.lit(None, dtype=pl.Float64)

    return (
        points.group_by([*KEYS, "engine", "quant", "family"])
        .agg(
            med("output_throughput_tok_s"),
            pl.col("output_throughput_tok_s").min().alias("thr_min"),
            pl.col("output_throughput_tok_s").max().alias("thr_max"),
            med("ttft_p50_s"),
            med("ttft_p95_s"),
            med("tpot_p50_s"),
            med("tpot_p95_s"),
            med("itl_p95_s"),
            med("e2e_p50_s"),
            med("slo_attainment"),
            med("goodput_tok_s"),
            med("output_tokens_mean"),
            med("input_tokens_mean"),
            peak.alias("peak_mem_mib"),
            pl.col("errors").sum().alias("errors"),
            pl.len().alias("repeats"),
        )
        .sort(["workload", "artifact", "level"])
    )


def _fmt(v: float | None, scale: float = 1.0, digits: int = 2) -> str:
    return "-" if v is None else f"{v * scale:.{digits}f}"


def markdown(agg: pl.DataFrame) -> str:
    lines = [
        "| Workload | Artifact | Engine | Conc. | Out tok/s (median [min-max]) | TTFT p50 / p95 (s) | "
        "TPOT p50 / p95 (ms) | SLO met | Goodput tok/s | Out tok/req | Peak mem MiB | Reps | Err |",
        "|---|---|---|---|---|---|---|---|---|---|---|---|---|",
    ]
    for r in agg.iter_rows(named=True):
        lines.append(
            f"| {r['workload']} | {r['artifact']} | {r['engine']} | {r['level']:g} | "
            f"{_fmt(r['output_throughput_tok_s'], digits=1)} "
            f"[{_fmt(r['thr_min'], digits=1)}-{_fmt(r['thr_max'], digits=1)}] | "
            f"{_fmt(r['ttft_p50_s'])} / {_fmt(r['ttft_p95_s'])} | "
            f"{_fmt(r['tpot_p50_s'], 1000, 0)} / {_fmt(r['tpot_p95_s'], 1000, 0)} | "
            f"{_fmt(r['slo_attainment'], 100, 0)}% | {_fmt(r['goodput_tok_s'], digits=1)} | "
            f"{_fmt(r['output_tokens_mean'], digits=0)} | {_fmt(r['peak_mem_mib'], digits=0)} | "
            f"{r['repeats']} | {r['errors']} |"
        )
    return "\n".join(lines)


def plot(agg: pl.DataFrame, out_dir: Path) -> list[Path]:
    paths = []
    for wl in agg["workload"].unique().sort().to_list():
        sub = agg.filter(pl.col("workload") == wl)
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        for eng, art in sub.select("engine", "artifact").unique().sort(["engine", "artifact"]).iter_rows():
            s = sub.filter((pl.col("artifact") == art) & (pl.col("engine") == eng)).sort("level")
            label = f"{eng} / {art}"
            ax1.plot(s["level"], s["output_throughput_tok_s"], marker="o", label=label)
            ax2.plot(s["output_throughput_tok_s"], s["tpot_p50_s"] * 1000, marker="o", label=label)
        ax1.set_xlabel("concurrent requests")
        ax1.set_ylabel("output tokens/s (all requests)")
        ax1.set_title(f"{wl}: throughput vs concurrency")
        ax2.set_xlabel("output tokens/s")
        ax2.set_ylabel("TPOT p50 (ms)")
        ax2.set_title(f"{wl}: latency vs throughput")
        for ax in (ax1, ax2):
            ax.grid(alpha=0.3)
        ax1.legend(fontsize=8)
        fig.tight_layout()
        path = out_dir / f"{wl}.png"
        fig.savefig(path, dpi=130)
        plt.close(fig)
        paths.append(path)
    return paths


def write_report(run_dirs: list[Path], out_dir: Path, title: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    agg = aggregate(load_points(run_dirs))
    agg.write_csv(out_dir / "summary.csv")
    figures = plot(agg, out_dir)
    body = [f"# {title}", "", f"Sources: {', '.join(str(d) for d in run_dirs)}", "", markdown(agg), ""]
    body += [f"![{p.stem}]({p.name})" for p in figures]
    path = out_dir / "report.md"
    path.write_text("\n".join(body) + "\n")
    return path
