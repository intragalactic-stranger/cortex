"""Run manifest: everything needed to reproduce (or distrust) a result (spec §0.4)."""

from __future__ import annotations

import contextlib
import hashlib
import json
import os
import platform
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from cortex_serve import __version__


def _cmd(args: list[str], cwd: Path | None = None) -> str | None:
    try:
        out = subprocess.run(args, capture_output=True, text=True, timeout=30, cwd=cwd, check=False)
    except (OSError, subprocess.TimeoutExpired):
        return None
    return (out.stdout or out.stderr).strip() or None


def git_state(repo_dir: Path) -> dict[str, Any]:
    sha = _cmd(["git", "rev-parse", "HEAD"], repo_dir)
    dirty = bool(_cmd(["git", "status", "--porcelain", "--untracked-files=no"], repo_dir))
    return {"sha": sha, "dirty": dirty}


def host_info() -> dict[str, Any]:
    cpu = None
    try:
        for line in Path("/proc/cpuinfo").read_text().splitlines():
            if line.startswith("model name"):
                cpu = line.split(":", 1)[1].strip()
                break
    except OSError:
        pass
    mem_kib = None
    with contextlib.suppress(OSError, ValueError, IndexError):
        mem_kib = int(Path("/proc/meminfo").read_text().split()[1])
    return {
        "hostname_hash": hashlib.sha256(platform.node().encode()).hexdigest()[:12],
        "cpu": cpu,
        "logical_cpus": os.cpu_count(),
        "memory_gib": round(mem_kib / 2**20, 2) if mem_kib else None,
        "kernel": platform.release(),
        "python": sys.version.split()[0],
        "docker": _cmd(["docker", "version", "--format", "{{.Server.Version}}"]),
        "gpu": _cmd(["nvidia-smi", "--query-gpu=name,driver_version,memory.total", "--format=csv,noheader"]),
    }


def image_digest(image: str) -> str | None:
    return _cmd(["docker", "image", "inspect", "--format", "{{index .RepoDigests 0}}", image])


def file_sha256(path: Path) -> str:
    """sha256 of a (large) file, cached next to it keyed by size + mtime."""
    stat = path.stat()
    cache = path.with_suffix(path.suffix + ".sha256.json")
    if cache.exists():
        data = json.loads(cache.read_text())
        if data.get("size") == stat.st_size and data.get("mtime") == stat.st_mtime:
            return str(data["sha256"])
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    digest = h.hexdigest()
    with contextlib.suppress(OSError):  # read-only model dir: fine, just slower next time
        cache.write_text(json.dumps({"size": stat.st_size, "mtime": stat.st_mtime, "sha256": digest}))
    return digest


def build_manifest(**sections: Any) -> dict[str, Any]:
    return {
        "schema": 1,
        "created_utc": datetime.now(UTC).isoformat(timespec="seconds"),
        "servebench_version": __version__,
        "host": host_info(),
        **sections,
    }
