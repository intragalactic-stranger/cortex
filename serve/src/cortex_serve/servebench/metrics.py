"""Aggregate per-request results into one benchmark point: latency percentiles, throughput, goodput.

Goodput (DistServe-style): a request "meets the SLO" when its own TTFT <= slo.ttft_s AND its mean
time per output token (TPOT) <= slo.tpot_s. Goodput = SLO-meeting requests (or their output tokens)
per second of measured wall time.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from typing import Any

from cortex_serve.servebench.client import RequestResult
from cortex_serve.servebench.config import SLO


def percentile(values: Sequence[float], q: float) -> float | None:
    """Linear-interpolation percentile (same definition as numpy's default)."""
    if not values:
        return None
    xs = sorted(values)
    pos = (len(xs) - 1) * q / 100
    lo, hi = math.floor(pos), math.ceil(pos)
    return xs[lo] + (xs[hi] - xs[lo]) * (pos - lo)


def meets_slo(r: RequestResult, slo: SLO) -> bool:
    tpot = r.tpot_s
    return r.ok and r.ttft_s is not None and r.ttft_s <= slo.ttft_s and tpot is not None and tpot <= slo.tpot_s


def summarize(results: Sequence[RequestResult], wall_s: float, slo: SLO) -> dict[str, Any]:
    ok = [r for r in results if r.ok]
    ttft = [r.ttft_s for r in ok if r.ttft_s is not None]
    tpot = [t for r in ok if (t := r.tpot_s) is not None]
    e2e = [r.e2e_s for r in ok if r.e2e_s is not None]
    itl = [g for r in ok for g in r.itls_s]
    out_tokens = sum(r.output_tokens or 0 for r in ok)
    in_tokens = sum(r.input_tokens for r in ok)
    good = [r for r in ok if meets_slo(r, slo)]
    good_tokens = sum(r.output_tokens or 0 for r in good)
    wall = max(wall_s, 1e-9)
    row: dict[str, Any] = {
        "requests": len(results),
        "errors": len(results) - len(ok),
        "wall_s": wall_s,
        "input_tokens_mean": in_tokens / len(ok) if ok else None,
        "output_tokens_mean": out_tokens / len(ok) if ok else None,
        "output_throughput_tok_s": out_tokens / wall,
        "total_throughput_tok_s": (out_tokens + in_tokens) / wall,
        "request_throughput_rps": len(ok) / wall,
        "slo_attainment": len(good) / len(results) if results else None,
        "goodput_rps": len(good) / wall,
        "goodput_tok_s": good_tokens / wall,
    }
    for name, values in (("ttft", ttft), ("tpot", tpot), ("itl", itl), ("e2e", e2e)):
        for q in (50, 90, 95, 99):
            row[f"{name}_p{q}_s"] = percentile(values, q)
        row[f"{name}_mean_s"] = sum(values) / len(values) if values else None
    return row
