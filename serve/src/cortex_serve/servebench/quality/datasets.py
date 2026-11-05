"""Pinned Hugging Face dataset files (parquet), cached locally."""

from __future__ import annotations

from pathlib import Path

import polars as pl

from cortex_serve.memcalc import hub

ARC = (
    "allenai/ai2_arc",
    "210d026faf9955653af8916fad021475a3f00453",
    "ARC-Challenge/test-00000-of-00001.parquet",
)
WIKITEXT2 = (
    "Salesforce/wikitext",
    "b08601e04326c79dfdd32d625aee71d232d685c3",
    "wikitext-2-raw-v1/test-00000-of-00001.parquet",
)


def fetch_parquet(repo: str, revision: str, path: str) -> pl.DataFrame:
    cache = hub.CACHE_DIR.parent / "datasets" / f"{repo.replace('/', '__')}@{revision}" / path
    if not cache.exists():
        with hub.client() as client:
            data = hub._get(client, f"{hub.HF}/datasets/{repo}/resolve/{revision}/{path}")
        cache.parent.mkdir(parents=True, exist_ok=True)
        cache.write_bytes(data)
    return pl.read_parquet(cache)


def wikitext2_raw(out: Path) -> Path:
    """Reconstruct wiki.test.raw (rows are lines with their own newlines)."""
    if not out.exists():
        df = fetch_parquet(*WIKITEXT2)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("".join(df["text"].to_list()))
    return out
