"""ARC-Challenge multiple-choice accuracy through the serving API (greedy, letter answer).

A quick, engine-agnostic quality smoke test: it exercises the real chat template and sampling path.
For quantization deltas it's complemented by perplexity (``perplexity.py``).
"""

from __future__ import annotations

import asyncio
import math
import random
import re
from typing import Any

import httpx

from cortex_serve.servebench.client import Protocol, send
from cortex_serve.servebench.quality.datasets import ARC, fetch_parquet
from cortex_serve.servebench.workloads import Request

_LETTER = re.compile(r"\b([A-E1-5])\b")
_FINAL = re.compile(r"answer\s*(?:is)?\s*[:\-]?\s*\**\(?([A-E1-5])\b", re.IGNORECASE)
COT_SUFFIX = "Think step by step, then finish with a final line of the form 'Answer: <label>'."


def build_items(n: int, seed: int, cot: bool = False) -> list[dict[str, Any]]:
    df = fetch_parquet(*ARC)
    rows = df.to_dicts()
    random.Random(seed).shuffle(rows)
    items = []
    for row in rows[:n]:
        labels, texts = row["choices"]["label"], row["choices"]["text"]
        options = "\n".join(f"{lab}. {txt}" for lab, txt in zip(labels, texts, strict=True))
        tail = COT_SUFFIX if cot else f"Answer with only the label ({', '.join(labels)}) of the correct option."
        prompt = f"Question: {row['question']}\n{options}\n\n{tail}"
        items.append({"id": row["id"], "prompt": prompt, "answer": row["answerKey"], "labels": labels})
    return items


def parse_answer(text: str, labels: list[str]) -> str | None:
    """First option label in the first non-empty line; hedges that name several options count as no answer."""
    lines = [ln for ln in text.strip().splitlines() if ln.strip()]
    if not lines:
        return None
    found = [m.group(1) for m in _LETTER.finditer(lines[0]) if m.group(1) in labels]
    return found[0] if len(set(found)) == 1 else None


def parse_final(text: str, labels: list[str]) -> str | None:
    """Label in the last 'Answer: X' statement (reasoning-first answers); None if absent."""
    found = [m.group(1) for m in _FINAL.finditer(text) if m.group(1) in labels]
    return found[-1] if found else None


def wilson(k: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n == 0:
        return (0.0, 0.0)
    p = k / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (centre - half, centre + half)


async def evaluate(
    base_url: str,
    protocol: Protocol,
    model: str,
    n: int,
    seed: int,
    concurrency: int = 2,
    cot: bool = False,
    max_tokens: int = 8,
) -> dict[str, Any]:
    """``cot=True``: the model may reason first; the last 'Answer: X' line is scored (chat-only models)."""
    items = build_items(n, seed, cot)
    sem = asyncio.Semaphore(concurrency)
    async with httpx.AsyncClient(timeout=300) as client:

        async def one(i: int, item: dict[str, Any]) -> dict[str, Any]:
            async with sem:
                res = await send(client, base_url, protocol, model, Request(i, "arc", item["prompt"], 0, max_tokens))
            pred = (parse_final if cot else parse_answer)(res.text, item["labels"])
            return {
                "id": item["id"],
                "pred": pred,
                "answer": item["answer"],
                "correct": pred == item["answer"],
                "error": res.error,
                "raw": res.text[:40],
            }

        rows = await asyncio.gather(*(one(i, it) for i, it in enumerate(items)))
    k = sum(r["correct"] for r in rows)
    lo, hi = wilson(k, len(rows))
    return {
        "n": len(rows),
        "correct": k,
        "accuracy": k / len(rows),
        "ci95_low": lo,
        "ci95_high": hi,
        "unparsed": sum(r["pred"] is None for r in rows),
        "errors": sum(r["error"] is not None for r in rows),
        "rows": rows,
    }
