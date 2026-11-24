"""Autoscaling timeline: client outcomes binned over time, aligned with Prometheus series."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

from cortex_serve.gatewaybench.load import Outcome, Stage, concurrency_at
from cortex_serve.servebench.metrics import percentile


@dataclass
class Bin:
    t_s: float  # bin start, seconds since t0
    completed: int
    failed: int
    ttft_p50_s: float | None
    ttft_p95_s: float | None


def bin_outcomes(outcomes: Sequence[Outcome], t0: float, bin_s: float, end_s: float) -> list[Bin]:
    """Bin by completion time. TTFT percentiles use successful requests whose first token fell in the bin."""
    n = max(1, int(-(-end_s // bin_s)))
    ttfts: list[list[float]] = [[] for _ in range(n)]
    done = [0] * n
    failed = [0] * n
    for o in outcomes:
        rel = o.start_unix - t0
        if o.ok and o.ttft_s is not None:
            i = int((rel + o.ttft_s) // bin_s)
            if 0 <= i < n:
                ttfts[i].append(o.ttft_s)
        j = int((rel + (o.e2e_s or 0.0)) // bin_s)
        if 0 <= j < n:
            done[j] += 1
            if not o.ok:
                failed[j] += 1
    return [
        Bin(
            t_s=i * bin_s,
            completed=done[i],
            failed=failed[i],
            ttft_p50_s=percentile(ttfts[i], 50),
            ttft_p95_s=percentile(ttfts[i], 95),
        )
        for i in range(n)
    ]


def parse_events(text: str) -> list[tuple[float, str]]:
    """``events.log`` lines: ``<unix seconds> <label...>``; other lines are ignored."""
    out = []
    for line in text.splitlines():
        head, _, label = line.strip().partition(" ")
        try:
            out.append((float(head), label.strip()))
        except ValueError:
            continue
    return out


def relative(points: Sequence[tuple[float, float]], t0: float) -> tuple[list[float], list[float]]:
    return [t - t0 for t, _ in points], [v for _, v in points]


def plot_timeline(
    out_png: Path,
    title: str,
    profile: Sequence[Stage],
    bins: Sequence[Bin],
    series: Mapping[str, Sequence[tuple[float, float]]],
    t0: float,
    slo_ttft_s: float,
    events: Sequence[tuple[float, str]] = (),
) -> None:
    """Three aligned panels: load and queue, replicas, TTFT.

    ``series`` keys used: ``demand`` (running + queued in the pool), ``queued`` (requests_deferred),
    ``replicas_desired`` (HPA), ``replicas_ready`` (Deployment available replicas).
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    end = max((b.t_s for b in bins), default=0.0) + (bins[1].t_s - bins[0].t_s if len(bins) > 1 else 0)
    fig, axes = plt.subplots(3, 1, figsize=(10, 8.5), sharex=True, constrained_layout=True)
    ax1, ax2, ax3 = axes

    xs = [i * 1.0 for i in range(int(end) + 1)]
    ax1.step(
        xs,
        [concurrency_at(profile, x) for x in xs],
        where="post",
        color="#555555",
        lw=1.5,
        label="offered load (client concurrency)",
    )
    for key, color, label in (
        ("demand", "#1f77b4", "pool demand: running + queued"),
        ("queued", "#d62728", "queue depth: requests_deferred"),
    ):
        if series.get(key):
            x, y = relative(series[key], t0)
            ax1.plot(x, y, color=color, lw=1.5, label=label)
    ax1.set_ylabel("requests")
    ax1.legend(loc="upper left", fontsize=8, frameon=False)
    ax1.grid(alpha=0.3)

    for key, color, style, label in (
        ("replicas_desired", "#9467bd", "--", "desired (KEDA/HPA)"),
        ("replicas_ready", "#2ca02c", "-", "ready replicas"),
    ):
        if series.get(key):
            x, y = relative(series[key], t0)
            ax2.step(x, y, where="post", color=color, ls=style, lw=2, label=label)
    ax2.set_ylabel("replicas (primary pool)")
    ax2.set_ylim(bottom=0)
    ax2.yaxis.get_major_locator().set_params(integer=True)
    ax2.legend(loc="upper left", fontsize=8, frameon=False)
    ax2.grid(alpha=0.3)

    bx = [b.t_s for b in bins]
    ax3.plot(bx, [b.ttft_p50_s for b in bins], marker=".", color="#1f77b4", lw=1, label="TTFT p50 (client)")
    ax3.plot(bx, [b.ttft_p95_s for b in bins], marker=".", color="#ff7f0e", lw=1, label="TTFT p95 (client)")
    ax3.axhline(slo_ttft_s, color="#d62728", ls=":", lw=1.2, label=f"SLO {slo_ttft_s:g} s")
    fails = [b for b in bins if b.failed]
    if fails:
        ax3.scatter([b.t_s for b in fails], [slo_ttft_s] * len(fails), marker="x", color="k", label="failed requests")
    ax3.set_ylabel("TTFT (s)")
    ax3.set_xlabel("seconds since start of load")
    ax3.set_ylim(bottom=0)
    ax3.legend(loc="upper left", fontsize=8, frameon=False)
    ax3.grid(alpha=0.3)

    for i, (t, label) in enumerate(events):
        for ax in axes:
            ax.axvline(t - t0, color="#7f7f7f", ls="--", lw=0.9)
        ax1.annotate(
            label,
            (t - t0, 1.0),
            xycoords=("data", "axes fraction"),
            rotation=90,
            fontsize=7,
            va="top",
            ha="right",
            color="#555555",
            xytext=(-2, -2 - 0 * i),
            textcoords="offset points",
        )

    fig.suptitle(title, fontsize=11)
    out_png.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_png, dpi=130)
    plt.close(fig)
