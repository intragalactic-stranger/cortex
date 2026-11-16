"""Run the quality suite (ARC-Challenge accuracy + WikiText-2 perplexity) over a config's artifacts."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import polars as pl

from cortex_serve.servebench.config import RunConfig
from cortex_serve.servebench.engines import make_engine
from cortex_serve.servebench.engines.base import artifact_path
from cortex_serve.servebench.manifest import build_manifest, file_sha256, git_state, image_digest
from cortex_serve.servebench.quality.arc import evaluate
from cortex_serve.servebench.quality.datasets import ARC, WIKITEXT2, wikitext2_raw
from cortex_serve.servebench.quality.perplexity import run_perplexity


def run_quality(
    cfg: RunConfig,
    out_root: Path,
    models_dir: Path,
    repo_dir: Path,
    tools_image: str,
    arc_n: int = 100,
    ppl_chunks: int = 20,
    only: list[str] | None = None,
) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / f"{cfg.name}-quality" / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    artifacts = [a for a in cfg.artifacts if not only or a.name in only]
    manifest = build_manifest(
        run={"name": f"{cfg.name}-quality", "experiment": cfg.experiment, "stamp": stamp},
        suite={
            "arc": {"dataset": ARC, "n": arc_n, "seed": cfg.seed, "greedy": True},
            "perplexity": {"dataset": WIKITEXT2, "ctx": 512, "chunks": ppl_chunks},
        },
        engine={"kind": cfg.engine.kind, "image": cfg.engine.image, "digest": image_digest(cfg.engine.image)},
        tools_image={"image": tools_image, "digest": image_digest(tools_image)},
        git=git_state(repo_dir),
        artifacts=[{**a.model_dump(), "sha256": file_sha256(models_dir / artifact_path(a))} for a in artifacts],
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    wiki = wikitext2_raw(out_root / ".data" / "wiki.test.raw")
    rows: list[dict[str, Any]] = []
    for a in artifacts:
        print(f"[quality] {a.name}: perplexity", flush=True)
        ppl = run_perplexity(tools_image, models_dir, artifact_path(a), wiki, chunks=ppl_chunks)
        print(f"[quality] {a.name}: PPL {ppl['ppl']}; ARC n={arc_n}", flush=True)
        args = {k: v for k, v in cfg.engine.args.items() if k != "ignore_eos"}
        engine = make_engine(cfg.engine.kind, cfg.engine.image, args, str(models_dir))
        engine.start(a)
        try:
            arc = asyncio.run(evaluate(engine.base_url, engine.protocol, engine.model_name, arc_n, cfg.seed))
        finally:
            engine.stop()
        (run_dir / f"arc-{a.name}.json").write_text(json.dumps(arc["rows"], indent=1))
        row = {
            "artifact": a.name,
            "family": a.family,
            "quant": a.quant,
            "ppl": ppl["ppl"],
            "ppl_err": ppl["ppl_err"],
            **{f"arc_{k}": v for k, v in arc.items() if k != "rows"},
        }
        rows.append(row)
        print(
            f"[quality] {a.name}: ARC {arc['accuracy']:.3f} [{arc['ci95_low']:.3f}, {arc['ci95_high']:.3f}]",
            flush=True,
        )
        pl.DataFrame(rows).write_parquet(run_dir / "quality.parquet")
        if ppl["ppl"] is None:
            (run_dir / f"ppl-{a.name}.log").write_text(ppl["log_tail"])
    return run_dir
