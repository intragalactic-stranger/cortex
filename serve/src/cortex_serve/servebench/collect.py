"""Resource and engine-metric collection during a benchmark.

* ``CgroupSampler`` reads a Docker container's cgroup v2 files at 1 Hz (CPU usage, memory): cheap
  and precise, unlike polling ``docker stats``.
* ``scrape_prometheus`` parses an engine's ``/metrics`` text exposition (llama.cpp ``--metrics``,
  vLLM, SGLang).
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Any

import httpx

CGROUP_ROOT = Path("/sys/fs/cgroup/system.slice")


class CgroupSampler:
    def __init__(self, container_id: str, interval_s: float = 1.0) -> None:
        self.path = CGROUP_ROOT / f"docker-{container_id}.scope"
        self.interval_s = interval_s
        self.samples: list[dict[str, float]] = []
        self._stop = threading.Event()
        self._thread = threading.Thread(target=self._run, daemon=True)

    @property
    def available(self) -> bool:
        return (self.path / "memory.current").exists()

    def _read(self) -> dict[str, float]:
        cpu = dict(line.split() for line in (self.path / "cpu.stat").read_text().splitlines())
        return {
            "t": time.time(),
            "memory_bytes": float((self.path / "memory.current").read_text()),
            "cpu_usage_usec": float(cpu["usage_usec"]),
        }

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                self.samples.append(self._read())
            except (OSError, KeyError, ValueError):
                return
            self._stop.wait(self.interval_s)

    def start(self) -> CgroupSampler:
        if self.available:
            self._thread.start()
        return self

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread.is_alive():
            self._thread.join(timeout=5)
        return summarize_samples(self.samples)


def summarize_samples(samples: list[dict[str, float]]) -> dict[str, Any]:
    if len(samples) < 2:
        return {"samples": len(samples)}
    span = samples[-1]["t"] - samples[0]["t"]
    cpu_s = (samples[-1]["cpu_usage_usec"] - samples[0]["cpu_usage_usec"]) / 1e6
    return {
        "samples": len(samples),
        "peak_memory_mib": max(s["memory_bytes"] for s in samples) / 2**20,
        "mean_memory_mib": sum(s["memory_bytes"] for s in samples) / len(samples) / 2**20,
        "avg_cpu_cores": cpu_s / span if span > 0 else None,
    }


def parse_prometheus(text: str) -> dict[str, float]:
    out: dict[str, float] = {}
    for line in text.splitlines():
        if not line or line.startswith("#"):
            continue
        parts = line.rsplit(" ", 1)
        if len(parts) != 2:
            continue
        try:
            out[parts[0]] = float(parts[1])
        except ValueError:
            continue
    return out


def scrape_prometheus(base_url: str, timeout_s: float = 5.0) -> dict[str, float]:
    try:
        resp = httpx.get(base_url + "/metrics", timeout=timeout_s)
        resp.raise_for_status()
    except httpx.HTTPError:
        return {}
    return parse_prometheus(resp.text)
