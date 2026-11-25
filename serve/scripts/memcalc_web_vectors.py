"""Export memcalc parity vectors for the TypeScript port used by the web site (M6).

    uv run python serve/scripts/memcalc_web_vectors.py     # -> web/test/fixtures/memcalc_vectors.json

golden.json holds the hand-calculated cases; this file adds machine-generated breadth: every fixture model's
per-layer spec and parameter breakdown, and full ``estimate()`` outputs over a grid of devices, weight formats,
KV formats, contexts, utilisations and options. ``web/test/vectors.test.ts`` requires the TS port to reproduce
every field. Re-run after any change to memcalc.
"""

from __future__ import annotations

import json
from itertools import cycle
from pathlib import Path
from typing import Any

from cortex_serve.memcalc.calc import estimate, seq_bytes
from cortex_serve.memcalc.devices import get_device
from cortex_serve.memcalc.loader import load_local

ROOT = Path(__file__).resolve().parents[2]
MODELS = ROOT / "serve" / "tests" / "fixtures" / "models"
OUT = ROOT / "web" / "test" / "fixtures" / "memcalc_vectors.json"

DEVICES = ["L4", "H100-80GB", "A10G", "T4", "Apple-M4-Max-128GB", "local-epyc-7571"]
LOW_BIT = cycle(["w4a16-g128", "awq-g128", "q4_k_m", "q8_0", "mxfp4", "q5_k_m"])
CTXS = [1, 511, 4096, 8192, 32768, 131072]


def main() -> None:
    models: dict[str, Any] = {}
    estimates: list[dict[str, Any]] = []
    for path in sorted(MODELS.iterdir()):
        spec = load_local(path)
        models[path.name] = {
            "name": spec.name,
            "params": spec.params.model_dump(),
            "moe": spec.moe.model_dump() if spec.moe else None,
            "layers": [layer.model_dump(mode="json") for layer in spec.layers],
            "seq_bytes": {kv: [seq_bytes(spec, c, kv) for c in CTXS] for kv in ("bf16", "fp8", "q4_0")},
        }
        has_ple = spec.params.per_layer_embeddings > 0
        for dev in DEVICES:
            device = get_device(dev)
            runs: list[dict[str, Any]] = [
                {"weight_fmt": "bf16", "kv_fmt": "auto", "ctx": 8192, "util": 0.9},
                {
                    "weight_fmt": "fp8",
                    "kv_fmt": "fp8",
                    "ctx": 32768,
                    "util": 0.85,
                    "components": "all",
                    "ple_offload": has_ple,
                },
                {"weight_fmt": next(LOW_BIT), "kv_fmt": "q8_0", "ctx": 1000, "util": 0.5, "reserve_gib": 0.25},
                {"weight_fmt": "int8", "kv_fmt": "bf16", "ctx": 131072, "util": 1.0, "with_mtp": True},
            ]
            for run in runs:
                est = estimate(spec, device, **run)
                estimates.append({"model": path.name, "device": dev, "args": run, "expect": est.model_dump()})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"ctxs": CTXS, "models": models, "estimates": estimates}, separators=(",", ":")) + "\n")
    print(f"{OUT}: {len(models)} models, {len(estimates)} estimates, {OUT.stat().st_size / 1024:.0f} KiB")


if __name__ == "__main__":
    main()
