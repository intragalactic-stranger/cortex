"""Resource and engine-metric collection during a benchmark.

* ``CgroupSampler`` reads a Docker container's cgroup v2 files at 1 Hz (CPU usage, memory): cheap
  and precise, unlike polling ``docker stats``.
* ``scrape_prometheus`` parses an engine's ``/metrics`` text exposition (llama.cpp ``--metrics``,
  vLLM, SGLang).
"""

from __future__ import annotations

import shutil
import subprocess
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


class NvidiaSmiSampler:
    """GPU telemetry at 1 Hz via ``nvidia-smi -lms`` (utilisation %, memory used MiB, power W)."""

    QUERY = "utilization.gpu,memory.used,power.draw"

    def __init__(self, interval_ms: int = 1000) -> None:
        self.interval_ms = interval_ms
        self.proc: subprocess.Popen[str] | None = None

    def start(self) -> NvidiaSmiSampler:
        if shutil.which("nvidia-smi"):
            self.proc = subprocess.Popen(
                [
                    "nvidia-smi",
                    f"--query-gpu={self.QUERY}",
                    "--format=csv,noheader,nounits",
                    f"-lms={self.interval_ms}",
                ],
                stdout=subprocess.PIPE,
                text=True,
            )
        return self

    def stop(self) -> dict[str, Any]:
        if not self.proc:
            return {"samples": 0}
        self.proc.terminate()
        out, _ = self.proc.communicate(timeout=10)
        return summarize_gpu(out)


def summarize_gpu(text: str) -> dict[str, Any]:
    rows = []
    for line in text.strip().splitlines():
        parts = [p.strip() for p in line.split(",")]
        try:
            rows.append(tuple(float(p) for p in parts[:3]))
        except ValueError:
            continue
    if not rows:
        return {"samples": 0}
    return {
        "samples": len(rows),
        "gpu_util_mean_pct": sum(r[0] for r in rows) / len(rows),
        "gpu_mem_peak_mib": max(r[1] for r in rows),
        "gpu_power_mean_w": sum(r[2] for r in rows) / len(rows),
    }


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


def prom_delta(before: dict[str, float], after: dict[str, float], prefix: str) -> float | None:
    """Sum of counter increases over every label set of a metric name (None if the metric is absent)."""
    keys = [k for k in after if k.split("{", 1)[0] == prefix]
    if not keys:
        return None
    return sum(after[k] - before.get(k, 0.0) for k in keys)


def json_valid(text: str, schema: dict[str, Any]) -> bool:
    """Parses as JSON and has the schema's required top-level keys with matching basic types."""
    import json

    try:
        obj = json.loads(text)
    except (ValueError, TypeError):
        return False
    if schema.get("type") == "object" and not isinstance(obj, dict):
        return False
    types: dict[str, type | tuple[type, ...]] = {
        "string": str, "integer": int, "number": (int, float), "boolean": bool, "array": list, "object": dict,
    }  # fmt: skip
    props: dict[str, Any] = schema.get("properties", {})
    for key in schema.get("required", []):
        if not isinstance(obj, dict) or key not in obj:
            return False
        want = types.get(str(props.get(key, {}).get("type", "")))
        if want is not None and not isinstance(obj[key], want):
            return False
    return True
