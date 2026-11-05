"""WikiText-2 perplexity with llama.cpp's ``llama-perplexity`` (comparable only within one tokenizer)."""

from __future__ import annotations

import re
import subprocess
from pathlib import Path
from typing import Any

_FINAL = re.compile(r"Final estimate: PPL = ([\d.]+) \+/- ([\d.]+)")


def parse_ppl(log: str) -> tuple[float, float] | None:
    m = _FINAL.search(log)
    return (float(m.group(1)), float(m.group(2))) if m else None


def run_perplexity(
    image: str,
    models_dir: Path,
    model_path: str,
    data_file: Path,
    ctx: int = 512,
    chunks: int = 20,
    threads: int = 4,
    timeout_s: float = 3600,
) -> dict[str, Any]:
    cmd = [
        "docker",
        "run",
        "--rm",
        "-v",
        f"{models_dir}:/models:ro",
        "-v",
        f"{data_file.parent}:/data:ro",
        "--entrypoint",
        "/app/llama-perplexity",
        image,
        "-m",
        f"/models/{model_path}",
        "-f",
        f"/data/{data_file.name}",
        "-c",
        str(ctx),
        "--chunks",
        str(chunks),
        "-t",
        str(threads),
    ]
    out = subprocess.run(cmd, capture_output=True, text=True, timeout=timeout_s, check=False)
    log = out.stdout + out.stderr
    result = parse_ppl(log)
    return {
        "ctx": ctx,
        "chunks": chunks,
        "tokens": ctx * chunks,
        "ppl": result[0] if result else None,
        "ppl_err": result[1] if result else None,
        "returncode": out.returncode,
        "log_tail": log[-1500:],
    }
