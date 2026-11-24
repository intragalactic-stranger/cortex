"""Orchestrates a benchmark run: engine lifecycle, workloads x repeats x load levels, collection, output."""

from __future__ import annotations

import asyncio
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import httpx
import polars as pl

from cortex_serve.servebench.client import RequestResult, send
from cortex_serve.servebench.collect import CgroupSampler, NvidiaSmiSampler, scrape_prometheus
from cortex_serve.servebench.config import ArtifactSpec, RunConfig
from cortex_serve.servebench.engines import Engine, make_engine
from cortex_serve.servebench.engines.base import artifact_path
from cortex_serve.servebench.engines.llamacpp import LlamaCppEngine
from cortex_serve.servebench.engines.llamacpp import parse_memory as parse_llamacpp_memory
from cortex_serve.servebench.engines.process import ProcessEngine
from cortex_serve.servebench.engines.process import parse_memory as parse_process_memory
from cortex_serve.servebench.load import LevelRun, run_concurrency, run_poisson
from cortex_serve.servebench.manifest import build_manifest, file_sha256, git_state, image_digest
from cortex_serve.servebench.metrics import percentile, summarize
from cortex_serve.servebench.tokenizer import HubTokenizer, Tokenizer
from cortex_serve.servebench.workloads import Request, generate


def _log(msg: str) -> None:
    print(f"[{datetime.now().strftime('%H:%M:%S')}] {msg}", flush=True)


def probe_llamacpp_memory(engine: LlamaCppEngine, artifact: ArtifactSpec) -> dict[str, float]:
    """Start once with -v to read model/KV buffer sizes from the log, then stop."""
    probe = LlamaCppEngine(
        engine.image, {**engine.args, "extra": [*engine.args.get("extra", []), "-v"]}, engine.models_dir
    )
    try:
        probe.start(artifact)
        return parse_llamacpp_memory(probe.logs())
    finally:
        probe.stop()


async def _run_level(
    engine: Engine, cfg: RunConfig, warmup: list[Request], measured: list[Request], level: float
) -> LevelRun:
    limits = httpx.Limits(max_connections=int(max(level, 1)) + 8)
    # vLLM and SGLang take ignore_eos per request (llama.cpp uses the server flag --ignore-eos)
    extra = {"ignore_eos": True} if cfg.engine.kind in ("vllm", "sglang") else None
    async with httpx.AsyncClient(timeout=cfg.load.timeout_s, limits=limits) as client:

        async def do(req: Request) -> RequestResult:
            return await send(
                client, engine.base_url, engine.protocol, engine.model_name, req, cfg.disable_thinking, extra=extra
            )

        if cfg.load.mode == "concurrency":
            return await run_concurrency(do, warmup, measured, int(level))
        return await run_poisson(do, warmup, measured, level, cfg.seed)


def _server_medians(results: list[RequestResult]) -> dict[str, float | None]:
    keys = ("prompt_per_second", "predicted_per_second", "eval_duration_ms")
    out: dict[str, float | None] = {}
    for key in keys:
        vals = [r.server_timings[key] for r in results if key in r.server_timings]
        if vals:
            out[f"server_{key}_p50"] = percentile(vals, 50)
    return out


def _artifact_record(a: ArtifactSpec, models_dir: Path) -> dict[str, Any]:
    rec: dict[str, Any] = a.model_dump()
    if a.file is not None and (models_dir / artifact_path(a)).exists():
        path = models_dir / artifact_path(a)
        rec.update(sha256=file_sha256(path), bytes=path.stat().st_size)
    qm = Path(a.repo) / "cortex_manifest.json"
    if a.repo.startswith("/") and qm.exists():  # M3 local checkpoint: carry its quantization manifest
        rec["quantize_manifest"] = json.loads(qm.read_text())
    return rec  # HF-repo artifacts (vLLM/SGLang) are identified by repo + pinned revision


def draft_stats(results: list[RequestResult]) -> dict[str, float | None]:
    """Speculative-decoding totals for one load point from llama-server's per-request ``timings``."""
    drafted = sum(r.server_timings.get("draft_n", 0.0) for r in results)
    accepted = sum(r.server_timings.get("draft_n_accepted", 0.0) for r in results)
    return {
        "draft_tokens": drafted,
        "draft_accepted": accepted,
        "draft_acceptance": accepted / drafted if drafted else None,
    }


def run(cfg: RunConfig, out_root: Path, models_dir: Path, repo_dir: Path, only: list[str] | None = None) -> Path:
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_dir = out_root / cfg.name / stamp
    run_dir.mkdir(parents=True, exist_ok=True)
    engine = make_engine(cfg.engine.kind, cfg.engine.image, cfg.engine.args, str(models_dir), cfg.engine.launch)
    artifacts = [a for a in cfg.artifacts if not only or a.name in only]

    manifest = build_manifest(
        run={"name": cfg.name, "experiment": cfg.experiment, "stamp": stamp},
        config=cfg.model_dump(),
        engine={
            "kind": cfg.engine.kind,
            "image": cfg.engine.image,
            "digest": image_digest(cfg.engine.image),
            "version": engine.version(),
        },
        git=git_state(repo_dir),
        artifacts=[_artifact_record(a, models_dir) for a in artifacts],
        budget=cfg.budget.model_dump() if cfg.budget else None,
    )
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))
    _log(f"run dir {run_dir}")

    request_rows: list[dict[str, Any]] = []
    point_rows: list[dict[str, Any]] = []
    tokenizers: dict[str, Tokenizer] = {}
    for artifact in artifacts:
        tok_spec = cfg.tokenizers[artifact.family]
        tok = tokenizers.setdefault(artifact.family, HubTokenizer(tok_spec.repo, tok_spec.revision))
        memory: dict[str, float] = {}
        if isinstance(engine, LlamaCppEngine):
            _log(f"{artifact.name}: memory probe")
            memory = probe_llamacpp_memory(engine, artifact)
        _log(f"{artifact.name}: starting {cfg.engine.kind}")
        engine.start(artifact)
        if isinstance(engine, ProcessEngine):
            memory = parse_process_memory(engine.kind, engine.logs())
        try:
            for wl in cfg.workloads:
                n_points = cfg.repeats * len(cfg.load.levels)
                warm_n = cfg.load.warmup_requests
                measured_total = cfg.repeats * sum(cfg.load.requests_for(lv) for lv in cfg.load.levels)
                pool = generate(wl, tok, warm_n * n_points + measured_total, cfg.seed)
                cursor = 0
                for rep in range(cfg.repeats):
                    for level in cfg.load.levels:
                        per_point = cfg.load.requests_for(level)
                        warmup = pool[cursor : cursor + warm_n]
                        cursor += warm_n
                        measured = pool[cursor : cursor + per_point]
                        cursor += per_point
                        sampler: CgroupSampler | NvidiaSmiSampler = (
                            NvidiaSmiSampler().start()
                            if isinstance(engine, ProcessEngine)
                            else CgroupSampler(engine.container or "").start()
                        )
                        before = scrape_prometheus(engine.base_url)
                        lr = asyncio.run(_run_level(engine, cfg, warmup, measured, level))
                        resources = sampler.stop()
                        after = scrape_prometheus(engine.base_url)
                        for r in lr.results:
                            r.output_tokens = len(tok.encode(r.text)) if r.text else 0
                        drafts = draft_stats(lr.results)
                        if "draft_model" in cfg.engine.args and not drafts["draft_tokens"]:
                            # Guard against a silent no-op: a loaded draft model that never drafted (E6, 2026-11-16).
                            raise RuntimeError(f"{artifact.name}: draft model configured but 0 tokens drafted")
                        row = {
                            "run": cfg.name,
                            "stamp": stamp,
                            "experiment": cfg.experiment,
                            "engine": cfg.engine.kind,
                            "engine_version": manifest["engine"]["version"],
                            "artifact": artifact.name,
                            "family": artifact.family,
                            "quant": artifact.quant,
                            "workload": wl.name,
                            "target_input_tokens": wl.input_tokens,
                            "target_output_tokens": wl.output_tokens,
                            "load_mode": cfg.load.mode,
                            "level": level,
                            "repeat": rep,
                            "slo_ttft_s": cfg.slo.ttft_s,
                            "slo_tpot_s": cfg.slo.tpot_s,
                            **summarize(lr.results, lr.wall_s, cfg.slo),
                            **{f"res_{k}": v for k, v in resources.items()},
                            **{f"mem_{k}": v for k, v in memory.items()},
                            **_server_medians(lr.results),
                            **drafts,
                            "metrics_scraped": bool(before and after),
                        }
                        point_rows.append(row)
                        for r in lr.results:
                            request_rows.append(
                                {
                                    "artifact": artifact.name,
                                    "workload": wl.name,
                                    "level": level,
                                    "repeat": rep,
                                    "request_id": r.request_id,
                                    "input_tokens": r.input_tokens,
                                    "engine_prompt_tokens": r.engine_prompt_tokens,
                                    "output_tokens": r.output_tokens,
                                    "engine_output_tokens": r.engine_output_tokens,
                                    "ttft_s": r.ttft_s,
                                    "e2e_s": r.e2e_s,
                                    "tpot_s": r.tpot_s,
                                    "itl_s": r.itls_s,
                                    "finish_reason": r.finish_reason,
                                    "error": r.error,
                                    "text_chars": len(r.text),
                                }
                            )
                        _log(
                            f"{artifact.name} {wl.name} rep{rep} level={level:g}: "
                            f"{row['output_throughput_tok_s']:.1f} out tok/s, TTFT p50 {row['ttft_p50_s'] or 0:.2f}s, "
                            f"TPOT p50 {(row['tpot_p50_s'] or 0) * 1000:.0f} ms, errors {row['errors']}"
                        )
                        pl.DataFrame(point_rows).write_parquet(run_dir / "points.parquet")
            (run_dir / f"engine-{artifact.name}.log").write_text(engine.logs())
        finally:
            engine.stop()
    pl.DataFrame(request_rows).write_parquet(run_dir / "requests.parquet")
    _log("done")
    return run_dir
