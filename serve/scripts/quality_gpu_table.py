"""Collect M3 E3 results (serve/results/quality-gpu/<name>/) into one table: score, stderr, delta vs bf16.

uv run python serve/scripts/quality_gpu_table.py [--md]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path("serve/results/quality-gpu")
METRICS = {  # task -> (metric key, higher is better)
    "arc_challenge": ("acc_norm,none", True),
    "hellaswag": ("acc_norm,none", True),
    "leaderboard_mmlu_pro": ("acc,none", True),
    "gsm8k": ("exact_match,strict-match", True),
    "ifeval": ("prompt_level_strict_acc,none", True),
    "wikitext": ("word_perplexity,none", False),
}


def load(name: str) -> dict[str, tuple[float, float | None]]:
    out: dict[str, tuple[float, float | None]] = {}
    for f in (ROOT / name).rglob("results_*.json"):
        for task, vals in json.loads(f.read_text())["results"].items():
            if task in METRICS:
                key, _ = METRICS[task]
                se = vals.get(key.replace(",", "_stderr,", 1))
                out[task] = (vals[key], se if isinstance(se, float) else None)
    man = ROOT / name / "eval_manifest.json"
    if man.exists():
        out["bfsi_ppl"] = (json.loads(man.read_text())["domain"]["bfsi_heldout_ppl"], None)
    return out


def main() -> None:
    names = sorted(p.name for p in ROOT.iterdir() if p.is_dir())
    data = {n: load(n) for n in names}
    base = data.get("qwen3-4b-bf16", {})
    cols = [*METRICS, "bfsi_ppl"]
    sep = " | " if "--md" in sys.argv else "  "
    print(sep.join(["model", *cols]))
    for n in names:
        cells = []
        for c in cols:
            if c not in data[n]:
                cells.append("-")
                continue
            v, se = data[n][c]
            pct = c not in ("wikitext", "bfsi_ppl")
            cell = f"{v * 100:.1f}" if pct else f"{v:.3f}"
            if se is not None and pct:
                cell += f"±{1.96 * se * 100:.1f}"
            if c in base and n != "qwen3-4b-bf16":
                d = (v - base[c][0]) * (100 if pct else 1)
                cell += f" ({d:+.1f})" if pct else f" ({(v / base[c][0] - 1) * 100:+.1f}%)"
            cells.append(cell)
        print(sep.join([n, *cells]))


if __name__ == "__main__":
    main()
