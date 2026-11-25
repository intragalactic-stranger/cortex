"""Load against the gateway: a time-varying closed-loop profile with per-request outcome logging.

The profile is a list of stages (duration, concurrency). Worker ``i`` is active while ``i`` is below
the stage's concurrency; an active worker sends its next request as soon as the last one finishes.
Every request is logged (start time, TTFT, end-to-end time, HTTP status, error, which backend
deployment served it, and the gateway's retry/fallback counters), so a run can prove "zero failed
requests" rather than only report percentiles.
"""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import Callable, Coroutine, Iterator, Sequence
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import httpx

from cortex_serve.servebench.client import RequestResult, consume_openai, request_body
from cortex_serve.servebench.workloads import Request

Clock = Callable[[], float]


@dataclass(frozen=True)
class Stage:
    duration_s: float
    concurrency: int


def parse_profile(text: str) -> list[Stage]:
    """``"60:1,120:4,60:0"`` -> 60 s at 1 user, 120 s at 4 users, 60 s idle."""
    stages = []
    for part in text.split(","):
        dur, _, conc = part.strip().partition(":")
        if not conc:
            raise ValueError(f"stage {part!r} must be '<seconds>:<concurrency>'")
        stage = Stage(float(dur), int(conc))
        if stage.duration_s <= 0 or stage.concurrency < 0:
            raise ValueError(f"bad stage {part!r}")
        stages.append(stage)
    if not stages:
        raise ValueError("empty profile")
    return stages


def profile_duration(profile: Sequence[Stage]) -> float:
    return sum(s.duration_s for s in profile)


def concurrency_at(profile: Sequence[Stage], t: float) -> int:
    """Target concurrency ``t`` seconds after the start; 0 once the profile is over."""
    if t < 0:
        return 0
    elapsed = 0.0
    for s in profile:
        elapsed += s.duration_s
        if t < elapsed:
            return s.concurrency
    return 0


@dataclass
class Outcome:
    id: int
    worker: int
    alias: str
    start_unix: float  # wall clock, to align with Prometheus series
    ttft_s: float | None = None
    e2e_s: float | None = None
    status: int | None = None
    error: str | None = None
    deployment: str | None = None  # x-litellm-model-id: which backend deployment answered
    api_base: str | None = None
    attempted_retries: int | None = None
    attempted_fallbacks: int | None = None
    output_tokens: int | None = None
    prompt_tokens: int | None = None
    text: str = ""
    extra: dict[str, Any] = field(default_factory=dict)

    @property
    def ok(self) -> bool:
        return self.error is None and self.status == 200 and self.ttft_s is not None


def _int_header(headers: httpx.Headers, name: str) -> int | None:
    v = headers.get(name)
    try:
        return int(v) if v is not None else None
    except ValueError:
        return None


def outcome_from(result: RequestResult, headers: httpx.Headers | None, status: int | None, **kw: Any) -> Outcome:
    """Merge a parsed stream (servebench client) with the gateway's response headers."""
    h = headers or httpx.Headers()
    return Outcome(
        ttft_s=result.ttft_s,
        e2e_s=result.e2e_s,
        status=status,
        error=result.error,
        deployment=h.get("x-litellm-model-id"),
        api_base=h.get("x-litellm-model-api-base"),
        attempted_retries=_int_header(h, "x-litellm-attempted-retries"),
        attempted_fallbacks=_int_header(h, "x-litellm-attempted-fallbacks"),
        output_tokens=result.engine_output_tokens,
        prompt_tokens=result.engine_prompt_tokens,
        text=result.text,
        **kw,
    )


async def send(
    client: httpx.AsyncClient,
    base_url: str,
    alias: str,
    req: Request,
    worker: int = 0,
    clock: Clock = time.perf_counter,
    wall: Clock = time.time,
) -> Outcome:
    """One streamed chat completion through the gateway (OpenAI protocol)."""
    res = RequestResult(
        request_id=req.id, workload=req.workload, input_tokens=req.input_tokens, max_tokens=req.max_tokens
    )
    body = request_body("openai", alias, req, disable_thinking=False)
    start_unix = wall()
    t0 = clock()
    headers: httpx.Headers | None = None
    status: int | None = None
    try:
        async with client.stream("POST", base_url + "/v1/chat/completions", json=body) as resp:
            headers, status = resp.headers, resp.status_code
            if resp.status_code != 200:
                res.error = f"HTTP {resp.status_code}: {(await resp.aread())[:300]!r}"
            else:
                await consume_openai(resp.aiter_lines(), res, t0, clock)
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        res.error = f"{type(exc).__name__}: {exc}"
    res.e2e_s = clock() - t0
    if res.error is None and res.ttft_s is None:
        res.error = "no tokens received"
    return outcome_from(res, headers, status, id=req.id, worker=worker, alias=alias, start_unix=start_unix)


SendFn = Callable[[Request, int], Coroutine[Any, Any, Outcome]]


async def run_profile(
    send_fn: SendFn,
    profile: Sequence[Stage],
    requests: Iterator[Request],
    on_outcome: Callable[[Outcome], None] | None = None,
    clock: Clock = time.monotonic,
    idle_poll_s: float = 0.2,
) -> list[Outcome]:
    """Drive ``send_fn`` following ``profile``; returns every outcome (also streamed to ``on_outcome``)."""
    n_workers = max(s.concurrency for s in profile)
    end = profile_duration(profile)
    t0 = clock()
    outcomes: list[Outcome] = []

    async def worker(i: int) -> None:
        while (t := clock() - t0) < end:
            if i >= concurrency_at(profile, t):
                await asyncio.sleep(idle_poll_s)
                continue
            try:
                req = next(requests)
            except StopIteration:
                return
            out = await send_fn(req, i)
            outcomes.append(out)
            if on_outcome:
                on_outcome(out)

    await asyncio.gather(*(worker(i) for i in range(n_workers)))
    return outcomes


def outcome_json(o: Outcome, keep_text: bool = False) -> str:
    d = asdict(o)
    if not keep_text:
        d["text"] = d["text"][:80]
    return json.dumps(d)


def read_outcomes(path: Path) -> list[Outcome]:
    out = []
    with path.open() as f:
        for line in f:
            if line.strip():
                out.append(Outcome(**json.loads(line)))
    return out
