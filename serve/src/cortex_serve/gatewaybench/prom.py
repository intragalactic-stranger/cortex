"""Minimal Prometheus HTTP API client (query_range) with timeouts and retries."""

from __future__ import annotations

import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import Any

import httpx


@dataclass
class Series:
    labels: dict[str, str]
    points: list[tuple[float, float]]  # (unix seconds, value)


def parse_matrix(payload: Mapping[str, Any]) -> list[Series]:
    """Parse a ``/api/v1/query_range`` JSON body (resultType matrix)."""
    if payload.get("status") != "success":
        raise RuntimeError(f"prometheus error: {payload.get('error')}")
    data = payload["data"]
    if data.get("resultType") != "matrix":
        raise RuntimeError(f"expected a matrix, got {data.get('resultType')}")
    out = []
    for item in data["result"]:
        points = [(float(t), float(v)) for t, v in item["values"]]
        out.append(Series(labels=dict(item["metric"]), points=points))
    return out


def query_range(
    base_url: str,
    query: str,
    start: float,
    end: float,
    step_s: float = 5.0,
    timeout_s: float = 30.0,
    retries: int = 3,
) -> list[Series]:
    params: dict[str, str | float] = {"query": query, "start": start, "end": end, "step": step_s}
    last: Exception | None = None
    for attempt in range(retries):
        try:
            r = httpx.get(f"{base_url}/api/v1/query_range", params=params, timeout=timeout_s)
            r.raise_for_status()
            return parse_matrix(r.json())
        except (httpx.HTTPError, RuntimeError) as exc:
            last = exc
            time.sleep(2**attempt)
    raise RuntimeError(f"query_range failed after {retries} attempts: {query!r}") from last
