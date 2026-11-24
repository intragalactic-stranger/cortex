"""gatewaybench: load profile, outcome parsing, timeline binning, canary rule, Prometheus parsing."""

from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx
import pytest
from cortex_serve.gatewaybench.canary import Policy, VersionStats, decide, is_correct, traffic_stats
from cortex_serve.gatewaybench.load import (
    Outcome,
    Stage,
    concurrency_at,
    outcome_from,
    outcome_json,
    parse_profile,
    profile_duration,
    read_outcomes,
    run_profile,
)
from cortex_serve.gatewaybench.manifest import pods_summary
from cortex_serve.gatewaybench.prom import parse_matrix
from cortex_serve.gatewaybench.summary import summarize_outcomes
from cortex_serve.gatewaybench.timeline import bin_outcomes
from cortex_serve.servebench.client import RequestResult
from cortex_serve.servebench.workloads import Request


def _o(
    i: int,
    start: float,
    ttft: float | None = 1.0,
    e2e: float = 2.0,
    dep: str | None = "d--primary-v1",
    error: str | None = None,
    status: int = 200,
) -> Outcome:
    return Outcome(
        id=i, worker=0, alias="a", start_unix=start, ttft_s=ttft, e2e_s=e2e, status=status, error=error, deployment=dep
    )


# --- profile -------------------------------------------------------------------------------------


def test_parse_profile_and_duration() -> None:
    p = parse_profile("60:1, 120:4,30:0")
    assert p == [Stage(60, 1), Stage(120, 4), Stage(30, 0)]
    assert profile_duration(p) == 210


@pytest.mark.parametrize("bad", ["", "60", "0:2", "10:-1", "x:1"])
def test_parse_profile_rejects(bad: str) -> None:
    with pytest.raises(ValueError):
        parse_profile(bad)


def test_concurrency_at_boundaries() -> None:
    p = [Stage(10, 1), Stage(20, 4), Stage(5, 0)]
    assert concurrency_at(p, -1) == 0
    assert concurrency_at(p, 0) == 1
    assert concurrency_at(p, 9.99) == 1
    assert concurrency_at(p, 10) == 4  # stage boundaries belong to the next stage
    assert concurrency_at(p, 29.9) == 4
    assert concurrency_at(p, 30) == 0
    assert concurrency_at(p, 100) == 0


def test_run_profile_respects_concurrency() -> None:
    active = 0
    peak = 0

    async def fake_send(req: Request, worker: int) -> Outcome:
        nonlocal active, peak
        active += 1
        peak = max(peak, active)
        await asyncio.sleep(0.01)
        active -= 1
        return _o(req.id, 0.0)

    reqs = iter(Request(id=i, workload="w", prompt="p", input_tokens=1, max_tokens=1) for i in range(10_000))
    seen: list[Outcome] = []
    outs = asyncio.run(run_profile(fake_send, [Stage(0.15, 1), Stage(0.15, 3)], reqs, seen.append, idle_poll_s=0.01))
    assert peak == 3
    assert len(outs) == len(seen) > 10
    assert {o.id for o in outs} == {o.id for o in seen}


# --- outcome parsing -----------------------------------------------------------------------------


def test_outcome_from_headers() -> None:
    res = RequestResult(request_id=3, workload="w", input_tokens=5, max_tokens=4, ttft_s=0.5, e2e_s=1.0, text="hi")
    res.engine_output_tokens = 4
    h = httpx.Headers(
        {
            "x-litellm-model-id": "cortex-default--canary-v2",
            "x-litellm-attempted-retries": "1",
            "x-litellm-attempted-fallbacks": "junk",
        }
    )
    o = outcome_from(res, h, 200, id=3, worker=1, alias="cortex-default", start_unix=100.0)
    assert o.ok and o.deployment == "cortex-default--canary-v2"
    assert o.attempted_retries == 1 and o.attempted_fallbacks is None
    assert o.output_tokens == 4


def test_outcome_not_ok_without_tokens_or_on_error() -> None:
    assert not _o(0, 0, ttft=None).ok
    assert not _o(0, 0, error="HTTP 500").ok
    assert not _o(0, 0, status=429).ok


def test_outcome_roundtrip(tmp_path: Path) -> None:
    p = tmp_path / "o.jsonl"
    o = _o(1, 5.0)
    o.text = "x" * 200
    p.write_text(outcome_json(o) + "\n\n")
    back = read_outcomes(p)
    assert len(back) == 1 and back[0].text == "x" * 80 and back[0].deployment == o.deployment


# --- summary and timeline ------------------------------------------------------------------------


def test_summarize_outcomes() -> None:
    outs = [_o(0, 0.0, ttft=1.0), _o(1, 1.0, ttft=3.0, dep="d--canary-v2"), _o(2, 2.0, error="boom", dep=None)]
    outs[1].attempted_retries = 1
    s = summarize_outcomes(outs)
    assert (s["requests"], s["ok"], s["failed"]) == (3, 2, 1)
    assert s["error_rate"] == pytest.approx(1 / 3)
    assert s["by_deployment"] == {"d--primary-v1": 1, "d--canary-v2": 1, "none": 1}
    assert s["with_retries"] == 1
    assert s["ttft_p50_s"] == pytest.approx(2.0)


def test_bin_outcomes_by_first_token_and_completion() -> None:
    t0 = 1000.0
    outs = [
        _o(0, t0 + 1, ttft=2.0, e2e=4.0),  # first token at 3 -> bin 0; done at 5 -> bin 0
        _o(1, t0 + 8, ttft=4.0, e2e=6.0),  # first token at 12 -> bin 1; done at 14 -> bin 1
        _o(2, t0 + 9, ttft=None, e2e=1.0, error="x"),  # failed, done at 10 -> bin 1
    ]
    bins = bin_outcomes(outs, t0, 10.0, 20.0)
    assert [b.t_s for b in bins] == [0.0, 10.0]
    assert (bins[0].completed, bins[0].failed, bins[0].ttft_p50_s) == (1, 0, 2.0)
    assert (bins[1].completed, bins[1].failed, bins[1].ttft_p95_s) == (2, 1, 4.0)


# --- canary --------------------------------------------------------------------------------------


def test_canary_hold_on_little_evidence() -> None:
    d = decide(VersionStats(300, 0, 2.0), VersionStats(5, 0, 2.0), 0.9, 0.9, Policy(min_canary_requests=30))
    assert d.verdict == "HOLD"


def test_canary_promote_when_all_checks_pass() -> None:
    # hand-computed: ttft limit = 1.25 * 2.0 = 2.5 >= 2.4; errors 0/40 <= 1%; acc 0.85 >= 0.9 - 0.05
    d = decide(VersionStats(400, 0, 2.0), VersionStats(40, 0, 2.4), 0.90, 0.85, Policy())
    assert d.verdict == "PROMOTE" and all(d.checks.values())


@pytest.mark.parametrize(
    ("canary", "acc", "failed"),
    [
        (VersionStats(40, 1, 2.0), 0.9, "errors"),  # 2.5% > max(1%, 0%)
        (VersionStats(40, 0, 2.6), 0.9, "latency"),  # 2.6 > 2.5
        (VersionStats(40, 0, 2.0), 0.8, "quality"),  # 0.80 < 0.85
        (VersionStats(40, 0, None), 0.9, "latency"),
    ],
)
def test_canary_rollback(canary: VersionStats, acc: float, failed: str) -> None:
    d = decide(VersionStats(400, 0, 2.0), canary, 0.9, acc, Policy())
    assert d.verdict == "ROLLBACK" and d.checks[failed] is False and d.reasons


def test_canary_error_budget_follows_worse_stable() -> None:
    # stable itself at 5% errors: the candidate is not blamed for a shared backend problem
    d = decide(VersionStats(400, 20, 2.0), VersionStats(40, 2, 2.0), 0.9, 0.9, Policy())
    assert d.checks["errors"]


def test_traffic_stats_by_deployment() -> None:
    outs = [_o(0, 0, dep="x--primary-v1"), _o(1, 0, dep="x--canary-v2", error="e"), _o(2, 0, dep=None, error="e")]
    s = traffic_stats(outs, "canary-v2")
    assert (s.requests, s.errors, s.error_rate) == (1, 1, 1.0)


@pytest.mark.parametrize(
    ("answer", "expected", "ok"),
    [
        ("12", "12", True),
        ("Paris.", "paris", True),
        ("<think>\nhmm\n</think>\n\nThe answer is 72.", "72", True),
        ("H2O", "h2o", True),
        ("I think it is probably not 63 but let me explain at length", "63", False),
        ("", "7", False),
        ("Lyon", "paris", False),
    ],
)
def test_is_correct(answer: str, expected: str, ok: bool) -> None:
    assert is_correct(answer, expected) is ok


# --- prometheus / manifest -----------------------------------------------------------------------


def test_parse_matrix() -> None:
    body = {
        "status": "success",
        "data": {"resultType": "matrix", "result": [{"metric": {"pod": "a"}, "values": [[1.0, "2"], [6.0, "3.5"]]}]},
    }
    [s] = parse_matrix(body)
    assert s.labels == {"pod": "a"} and s.points == [(1.0, 2.0), (6.0, 3.5)]
    with pytest.raises(RuntimeError):
        parse_matrix({"status": "error", "error": "bad"})


def test_pods_summary() -> None:
    pods = {
        "items": [
            {
                "metadata": {
                    "name": "p",
                    "labels": {"app.kubernetes.io/component": "model", "cortex.dev/model": "primary"},
                    "annotations": {"cortex.dev/weights": "f.gguf"},
                },
                "spec": {"containers": [{"name": "llama-server", "image": "img@sha256:1", "args": ["-m", "x"]}]},
                "status": {"containerStatuses": [{"name": "llama-server", "imageID": "img@sha256:1"}]},
            }
        ]
    }
    [row] = pods_summary(pods)
    assert row["model"] == "primary" and row["image_id"] == "img@sha256:1" and row["weights"] == "f.gguf"
    assert json.dumps(row)
