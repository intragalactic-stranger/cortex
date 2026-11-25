"""Local spend guard for Modal launches (D-030): credits only, staged gates, $245 ceiling.

Several GPU jobs may run at once, so each launcher *reserves* its worst case in a local file while it runs; a new
launch must fit ``metered + in-flight worst cases + its own worst case`` under the gate. Metered spend already
includes part of the in-flight jobs, so this double-counts on purpose (conservative).
"""

from __future__ import annotations

import json
import os
import subprocess
import time
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path

from cortex_serve.servebench.config import budget_check

INFLIGHT_DIR = Path.home() / ".cache" / "cortex-modal-inflight"


def _alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def inflight_usd(directory: Path = INFLIGHT_DIR) -> float:
    total = 0.0
    for f in directory.glob("*.json") if directory.exists() else []:
        rec = json.loads(f.read_text())
        if _alive(int(rec["pid"])):
            total += float(rec["worst_usd"])
        else:
            f.unlink(missing_ok=True)
    return total


def billing() -> tuple[float, float]:
    out = subprocess.run(
        ["modal", "billing", "summary", "--json"], capture_output=True, text=True, check=True, timeout=60
    )
    data = json.loads(out.stdout)
    return float(data["metered_cost"]), float(data["billed_cost"])


@contextmanager
def guarded(name: str, worst_usd: float, approved_gate: float, yes: bool) -> Iterator[None]:
    """Check credits-only + gates, then hold a reservation for the duration of the remote call."""
    spent, billed = billing()
    others = inflight_usd()
    print(
        f"[cost] metered this month ${spent:.2f}, in-flight worst case ${others:.2f}, billed to card ${billed:.2f} "
        f"(approved gate ${approved_gate:.0f})",
        flush=True,
    )
    if billed > 0.005:
        raise SystemExit(f"[cost] refusing: ${billed:.2f} is already billed to the card; credits must cover everything")
    if reason := budget_check(spent + others, worst_usd, approved_gate):
        raise SystemExit(f"[cost] refusing: {reason}")
    if not yes:
        raise SystemExit("[cost] dry run only. Re-run with --yes to start the GPU job.")
    INFLIGHT_DIR.mkdir(parents=True, exist_ok=True)
    rec = INFLIGHT_DIR / f"{os.getpid()}-{int(time.time())}.json"
    rec.write_text(json.dumps({"pid": os.getpid(), "name": name, "worst_usd": worst_usd, "started": time.time()}))
    try:
        yield
    finally:
        rec.unlink(missing_ok=True)


def skip_config(index: int, elapsed_s: float, need_s: float, deadline_s: float) -> bool:
    """In a multi-config GPU session (deadline = sum of the configs' caps), skip a later config whose own cap no
    longer fits because earlier configs overran theirs. The first config always runs."""
    return bool(deadline_s) and index > 0 and elapsed_s + need_s > deadline_s
