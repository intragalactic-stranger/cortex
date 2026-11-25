"""Audit M3 artifacts on the Modal volume from their safetensors headers only (no weights read). CPU, cents.

uv run --extra modal modal run serve/deploy/modal/artifact_audit.py --artifacts a,b
"""

from __future__ import annotations

import modal

app = modal.App("cortex-artifact-audit")
hf_cache = modal.Volume.from_name("cortex-hf-cache")
image = modal.Image.debian_slim(python_version="3.12")


def _group(name: str) -> str:
    for key in ("vision", "audio", "embed_tokens_per_layer", "embed_tokens", "lm_head"):
        if key in name:
            return key
    return "text_layers" if ".layers." in name else "other"


@app.function(image=image, volumes={"/hf": hf_cache}, timeout=300)
def audit(artifacts: list[str]) -> dict[str, dict[str, float]]:
    import json
    import struct
    from collections import defaultdict
    from pathlib import Path

    out: dict[str, dict[str, float]] = {}
    for a in artifacts:
        f = Path("/hf/artifacts") / a / "model.safetensors"
        with f.open("rb") as fh:
            n = struct.unpack("<Q", fh.read(8))[0]
            header = json.loads(fh.read(n))
        header.pop("__metadata__", None)
        sizes: dict[str, float] = defaultdict(float)
        for name, t in header.items():
            b = t["data_offsets"][1] - t["data_offsets"][0]
            sizes[f"{_group(name)}:{t['dtype']}"] += b / 2**30
            sizes["total"] += b / 2**30
        out[a] = {k: round(v, 3) for k, v in sorted(sizes.items())}
    return out


@app.local_entrypoint()
def main(artifacts: str) -> None:
    for a, sizes in audit.remote(artifacts.split(",")).items():
        print(a)
        for k, v in sizes.items():
            print(f"  {k:40} {v:8.3f} GiB")
