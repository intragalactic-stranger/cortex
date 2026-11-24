"""Calibration and held-out text sets. Pure functions; dataset loading happens in the Modal job.

Generic: ``HuggingFaceH4/ultrachat_200k`` (MIT), chat-formatted.
BFSI (D-026): eCFR Title 12 (Banks and Banking) + Title 17 (Commodity and Securities Exchanges) from
``Azzindani/US_Regulation_ECFR_20260101``. Split by CFR *part* so calibration and held-out text never share a part.
"""

from __future__ import annotations

import hashlib
from collections.abc import Iterable, Mapping
from dataclasses import dataclass

GENERIC = ("HuggingFaceH4/ultrachat_200k", "8049631c405ae6576f93f445c6b8166f76f5505a", "train_sft")
BFSI = ("Azzindani/US_Regulation_ECFR_20260101", "04148fee3c9832a2789a61ee4cdcd38063ec08d7", "train")
BFSI_TITLES = frozenset({"12", "17"})
HELDOUT_BUCKETS = 2  # of 10: ~20% of parts are held out


def _h(text: str) -> int:
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big")


@dataclass(frozen=True)
class Split:
    calib: list[str]
    heldout: list[str]
    calib_ids: list[str]
    heldout_ids: list[str]


def split_bfsi(rows: Iterable[Mapping[str, str]], n_calib: int, n_heldout: int, min_chars: int = 200) -> Split:
    """Deterministic part-level split of eCFR sections, then a stable hash order inside each side."""
    calib: list[tuple[int, str, str]] = []
    held: list[tuple[int, str, str]] = []
    for r in rows:
        if r["title_number"] not in BFSI_TITLES or len(r["section_text"]) < min_chars:
            continue
        part = f"{r['title_number']}:{r['part_number']}"
        item = (_h(r["rag_id"]), r["rag_id"], f"{r['section_subject']}\n{r['section_text']}")
        (held if _h(part) % 10 < HELDOUT_BUCKETS else calib).append(item)
    calib.sort()
    held.sort()
    c, h = calib[:n_calib], held[:n_heldout]
    return Split([t for _, _, t in c], [t for _, _, t in h], [i for _, i, _ in c], [i for _, i, _ in h])


def ids_digest(ids: Iterable[str]) -> str:
    """Short fingerprint of the exact sample ids, recorded in each artifact's manifest."""
    return hashlib.sha256("\n".join(ids).encode()).hexdigest()[:16]
