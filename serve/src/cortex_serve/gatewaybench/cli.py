"""gatewaybench CLI: load the M5 gateway, log every request, build the autoscaling timeline, gate canaries."""

from __future__ import annotations

import asyncio
import json
import os
import time
from dataclasses import asdict
from pathlib import Path
from typing import Annotated, Any

import httpx
import typer

from cortex_serve.gatewaybench import canary as canary_mod
from cortex_serve.gatewaybench.load import (
    Outcome,
    Stage,
    outcome_json,
    parse_profile,
    profile_duration,
    read_outcomes,
    run_profile,
    send,
)
from cortex_serve.gatewaybench.manifest import cluster_manifest
from cortex_serve.gatewaybench.prom import query_range
from cortex_serve.gatewaybench.summary import summarize_outcomes
from cortex_serve.gatewaybench.timeline import bin_outcomes, plot_timeline
from cortex_serve.servebench.config import WorkloadSpec
from cortex_serve.servebench.manifest import build_manifest, git_state
from cortex_serve.servebench.tokenizer import HubTokenizer
from cortex_serve.servebench.workloads import Request, generate

app = typer.Typer(no_args_is_help=True, add_completion=False, help="M5 gateway load, timeline and canary tools.")

REPO = Path(__file__).resolve().parents[4]
TOKENIZER = ("Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca")
BASE_URL = "http://127.0.0.1:4000"
PROM_URL = "http://127.0.0.1:9090"


def _api_key() -> str:
    key = os.environ.get("CORTEX_API_KEY")
    if not key:
        raise typer.BadParameter("set CORTEX_API_KEY to a gateway virtual key (serve/scripts/m5_key.sh)")
    return key


def _client(key: str, timeout_s: float) -> httpx.AsyncClient:
    limits = httpx.Limits(max_connections=64, max_keepalive_connections=64)
    return httpx.AsyncClient(
        headers={"Authorization": f"Bearer {key}"},
        timeout=httpx.Timeout(timeout_s, connect=10.0),
        limits=limits,
    )


def _requests(input_tokens: int, output_tokens: int, n: int, seed: int) -> list[Request]:
    spec = WorkloadSpec(
        name=f"chat-{input_tokens}-{output_tokens}", input_tokens=input_tokens, output_tokens=output_tokens
    )
    return generate(spec, HubTokenizer(*TOKENIZER), n, seed)


@app.command("load")
def load_cmd(
    out: Annotated[Path, typer.Option("--out", help="Run directory (created)")],
    profile: Annotated[str, typer.Option("--profile", help="'<seconds>:<concurrency>,...'")],
    alias: Annotated[str, typer.Option("--alias")] = "cortex-default",
    base_url: Annotated[str, typer.Option("--base-url")] = BASE_URL,
    input_tokens: Annotated[int, typer.Option("--input-tokens")] = 128,
    output_tokens: Annotated[int, typer.Option("--output-tokens")] = 64,
    seed: Annotated[int, typer.Option("--seed")] = 1234,
    timeout_s: Annotated[float, typer.Option("--timeout", help="Client-side per-request timeout")] = 300.0,
    namespace: Annotated[str, typer.Option("--namespace")] = "cortex",
    note: Annotated[str, typer.Option("--note")] = "",
) -> None:
    """Run a load profile through the gateway; writes outcomes.jsonl (one line per request) + manifest.json."""
    stages = parse_profile(profile)
    key = _api_key()
    out.mkdir(parents=True, exist_ok=True)
    # Enough distinct prompts that no request repeats within a run (prompt-cache hits would flatter TTFT).
    pool = _requests(input_tokens, output_tokens, 2000, seed)
    manifest: dict[str, Any] = build_manifest(
        kind="gatewaybench-load",
        git=git_state(REPO),
        alias=alias,
        base_url=base_url,
        profile=[asdict(s) for s in stages],
        workload={
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "seed": seed,
            "tokenizer": f"{TOKENIZER[0]}@{TOKENIZER[1]}",
        },
        note=note,
        cluster=cluster_manifest(namespace),
    )
    log_path = out / "outcomes.jsonl"
    with log_path.open("w") as log:

        def on_outcome(o: Outcome) -> None:
            log.write(outcome_json(o) + "\n")
            log.flush()

        async def main() -> list[Outcome]:
            async with _client(key, timeout_s) as client:

                async def send_one(req: Request, worker: int) -> Outcome:
                    return await send(client, base_url, alias, req, worker)

                return await run_profile(send_one, stages, iter(pool), on_outcome)

        manifest["t_start_unix"] = time.time()
        typer.echo(f"load: {profile_duration(stages):.0f} s profile against {alias} -> {log_path}")
        outcomes = asyncio.run(main())
        manifest["t_end_unix"] = time.time()
    (out / "manifest.json").write_text(json.dumps(manifest, indent=2))
    summary = summarize_outcomes(outcomes)
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps({k: summary[k] for k in ("requests", "ok", "failed", "by_deployment")}, indent=2))


def _series(prom: str, query: str, start: float, end: float, step: float) -> list[tuple[float, float]]:
    """Sum all returned series point-wise (queries are already aggregated; this guards against splits)."""
    acc: dict[float, float] = {}
    for s in query_range(prom, query, start, end, step):
        for t, v in s.points:
            acc[t] = acc.get(t, 0.0) + v
    return sorted(acc.items())


@app.command("timeline")
def timeline_cmd(
    run_dir: Annotated[Path, typer.Argument(help="Directory written by 'load'")],
    prom: Annotated[str, typer.Option("--prom")] = PROM_URL,
    model: Annotated[str, typer.Option("--model", help="Pool (cortex.dev/model label)")] = "primary",
    namespace: Annotated[str, typer.Option("--namespace")] = "cortex",
    bin_s: Annotated[float, typer.Option("--bin")] = 10.0,
    slo_ttft_s: Annotated[float, typer.Option("--slo-ttft")] = 3.0,
    title: Annotated[str, typer.Option("--title")] = "",
) -> None:
    """Query Prometheus for the run window and draw replicas vs queue depth vs TTFT."""
    manifest = json.loads((run_dir / "manifest.json").read_text())
    t0, t1 = float(manifest["t_start_unix"]), float(manifest["t_end_unix"])
    start, end, step = t0 - 30, t1 + 90, 5.0
    sel = f'namespace="{namespace}",cortex_model="{model}"'
    deploy = f"cortex-model-{model}"
    queries = {
        "demand": f"sum(llamacpp:requests_processing{{{sel}}}) + sum(llamacpp:requests_deferred{{{sel}}})",
        "queued": f"sum(llamacpp:requests_deferred{{{sel}}})",
        "processing": f"sum(llamacpp:requests_processing{{{sel}}})",
        "replicas_desired": f'max(kube_horizontalpodautoscaler_status_desired_replicas{{namespace="{namespace}",'
        f'horizontalpodautoscaler="keda-hpa-{deploy}"}})',
        "replicas_ready": f'max(kube_deployment_status_replicas_available{{namespace="{namespace}",'
        f'deployment="{deploy}"}})',
        "keda_metric": f'max(keda_scaler_metrics_value{{scaledObject="{deploy}"}})',
    }
    series: dict[str, list[tuple[float, float]]] = {}
    for name, q in queries.items():
        try:
            series[name] = _series(prom, q, start, end, step)
        except RuntimeError as exc:
            typer.echo(f"warning: {name}: {exc}")
            series[name] = []
    (run_dir / "series.json").write_text(json.dumps({"queries": queries, "series": series}, indent=2))
    outcomes = read_outcomes(run_dir / "outcomes.jsonl")
    profile = [Stage(**s) for s in manifest["profile"]]
    bins = bin_outcomes(outcomes, t0, bin_s, (t1 - t0) + bin_s)
    (run_dir / "bins.json").write_text(json.dumps([asdict(b) for b in bins], indent=2))
    plot_timeline(
        run_dir / "timeline.png",
        title or f"Queue-depth autoscaling on k3d (CPU): {manifest['alias']}",
        profile,
        bins,
        series,
        t0,
        slo_ttft_s,
    )
    typer.echo(str(run_dir / "timeline.png"))


@app.command("summary")
def summary_cmd(run_dir: Annotated[Path, typer.Argument()]) -> None:
    """Recompute summary.json from outcomes.jsonl."""
    summary = summarize_outcomes(read_outcomes(run_dir / "outcomes.jsonl"))
    (run_dir / "summary.json").write_text(json.dumps(summary, indent=2))
    typer.echo(json.dumps(summary, indent=2))


@app.command("canary")
def canary_cmd(
    traffic: Annotated[Path, typer.Option("--traffic", help="Load run dir from the 90/10 canary window")],
    out: Annotated[Path, typer.Option("--out")],
    stable_alias: Annotated[str, typer.Option("--stable-alias")] = "cortex-stable",
    candidate_alias: Annotated[str, typer.Option("--candidate-alias")] = "cortex-candidate",
    stable_id: Annotated[str, typer.Option("--stable-id", help="Substring of the stable deployment id")] = "primary-v1",
    candidate_id: Annotated[str, typer.Option("--candidate-id")] = "canary-v2",
    base_url: Annotated[str, typer.Option("--base-url")] = BASE_URL,
    min_requests: Annotated[int, typer.Option("--min-requests")] = 30,
    max_ttft_ratio: Annotated[float, typer.Option("--max-ttft-ratio")] = 1.25,
) -> None:
    """Score the probe set on both versions, combine with traffic stats, write decision.json."""
    key = _api_key()
    outcomes = read_outcomes(traffic / "outcomes.jsonl")
    stable = canary_mod.traffic_stats(outcomes, stable_id)
    cand = canary_mod.traffic_stats(outcomes, candidate_id)

    async def score(alias: str) -> list[dict[str, Any]]:
        rows = []
        async with _client(key, 300.0) as client:
            for i, (q, expected) in enumerate(canary_mod.PROBES):
                req = Request(id=i, workload="canary-probe", prompt=q, input_tokens=0, max_tokens=12)
                o = await send(client, base_url, alias, req)
                rows.append(
                    {
                        "q": q,
                        "expected": expected,
                        "answer": o.text,
                        "ok": o.ok,
                        "deployment": o.deployment,
                        "correct": o.ok and canary_mod.is_correct(o.text, expected),
                    }
                )
        return rows

    s_rows, c_rows = asyncio.run(score(stable_alias)), asyncio.run(score(candidate_alias))
    s_acc = sum(r["correct"] for r in s_rows) / len(s_rows)
    c_acc = sum(r["correct"] for r in c_rows) / len(c_rows)
    policy = canary_mod.Policy(min_canary_requests=min_requests, max_ttft_ratio=max_ttft_ratio)
    d = canary_mod.decide(stable, cand, s_acc, c_acc, policy)
    out.mkdir(parents=True, exist_ok=True)
    result = {
        "policy": asdict(policy),
        "traffic_run": str(traffic),
        "stable": {**asdict(stable), "error_rate": stable.error_rate, "eval_accuracy": s_acc},
        "candidate": {**asdict(cand), "error_rate": cand.error_rate, "eval_accuracy": c_acc},
        "decision": asdict(d),
        "probes": {"stable": s_rows, "candidate": c_rows},
        "manifest": build_manifest(kind="gatewaybench-canary", git=git_state(REPO)),
    }
    (out / "decision.json").write_text(json.dumps(result, indent=2))
    typer.echo(
        json.dumps({"stable": result["stable"], "candidate": result["candidate"], "decision": asdict(d)}, indent=2)
    )


POD_QUERIES = {
    "demand": "llamacpp:requests_processing{{{sel}}} + on(pod) llamacpp:requests_deferred{{{sel}}}",
    "deferred": "llamacpp:requests_deferred{{{sel}}}",
    "cpu_cores": 'sum by (pod) (rate(container_cpu_usage_seconds_total{{namespace="{ns}",container="llama-server",'
    'pod=~"cortex-model-{model}-.*"}}[30s]))',
    "cpu_throttled_ratio": 'sum by (pod) (rate(container_cpu_cfs_throttled_periods_total{{namespace="{ns}",'
    'pod=~"cortex-model-{model}-.*"}}[30s])) / sum by (pod) (rate(container_cpu_cfs_periods_total{{namespace="{ns}",'
    'pod=~"cortex-model-{model}-.*"}}[30s]))',
    "decode_tok_s": "sum by (pod) (rate(llamacpp:tokens_predicted_total{{{sel}}}[30s]))",
}


@app.command("balance")
def balance_cmd(
    run_dir: Annotated[Path, typer.Argument()],
    start_s: Annotated[float, typer.Option("--from", help="Seconds after load start")],
    end_s: Annotated[float, typer.Option("--to")],
    prom: Annotated[str, typer.Option("--prom")] = PROM_URL,
    model: Annotated[str, typer.Option("--model")] = "primary",
    namespace: Annotated[str, typer.Option("--namespace")] = "cortex",
) -> None:
    """Per-pod mean/max of demand, queue, CPU, throttling and decode rate in a window (load balance check)."""
    t0 = float(json.loads((run_dir / "manifest.json").read_text())["t_start_unix"])
    sel = f'namespace="{namespace}",cortex_model="{model}"'
    out: dict[str, Any] = {"window_s": [start_s, end_s], "queries": {}, "per_pod": {}}
    for name, tmpl in POD_QUERIES.items():
        q = tmpl.format(sel=sel, ns=namespace, model=model)
        out["queries"][name] = q
        for s in query_range(prom, q, t0 + start_s, t0 + end_s, 5.0):
            vals = [v for _, v in s.points]
            pod = s.labels.get("pod", "?")
            out["per_pod"].setdefault(pod, {})[name] = {
                "mean": sum(vals) / len(vals) if vals else None,
                "max": max(vals) if vals else None,
            }
    path = run_dir / f"balance_{int(start_s)}-{int(end_s)}s.json"
    path.write_text(json.dumps(out, indent=2))
    for pod, m in sorted(out["per_pod"].items()):
        typer.echo(
            pod + "  " + "  ".join(f"{k}={v['mean']:.2f}/{v['max']:.2f}" for k, v in m.items() if v["mean"] is not None)
        )


@app.command("pods")
def pods_cmd(
    run_dir: Annotated[Path, typer.Argument()],
    prom: Annotated[str, typer.Option("--prom")] = PROM_URL,
    model: Annotated[str, typer.Option("--model")] = "primary",
    namespace: Annotated[str, typer.Option("--namespace")] = "cortex",
    grace_s: Annotated[float, typer.Option("--grace", help="terminationGracePeriodSeconds of the pods")] = 120.0,
) -> None:
    """Pod lifecycle during a run from kube-state-metrics: created, ready, delete requested (s after load start).

    ``metadata.deletionTimestamp`` is the deletion *deadline* (request time + grace period), so the
    request time is that minus ``--grace``.
    """
    m = json.loads((run_dir / "manifest.json").read_text())
    t0, t1 = float(m["t_start_unix"]), float(m["t_end_unix"])
    sel = f'namespace="{namespace}",pod=~"cortex-model-{model}-.*"'
    rows: dict[str, dict[str, float | None]] = {}
    for name, metric in (
        ("created", "kube_pod_created"),
        ("ready", "kube_pod_status_ready_time"),
        ("deletion", "kube_pod_deletion_timestamp"),
    ):
        for s in query_range(prom, f"max by (pod) ({metric}{{{sel}}})", t0 - 600, t1 + 120, 5.0):
            pod = s.labels.get("pod", "?")
            rows.setdefault(pod, {"created": None, "ready": None, "deletion": None})
            rows[pod][name] = round(s.points[-1][1] - t0, 1) if s.points else None
    for r in rows.values():
        c, rd, dl = r["created"], r["ready"], r.pop("deletion")
        r["create_to_ready_s"] = round(rd - c, 1) if c is not None and rd is not None else None
        r["delete_requested"] = round(dl - grace_s, 1) if dl is not None else None
    (run_dir / "pods.json").write_text(json.dumps(rows, indent=2))
    for pod, r in sorted(rows.items(), key=lambda kv: kv[1]["created"] or 0.0):
        typer.echo(f"{pod}  {r}")
