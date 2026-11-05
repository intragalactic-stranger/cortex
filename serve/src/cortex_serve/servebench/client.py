"""Async streaming client with client-side timing (spec §8 rule 1).

TTFT = time from sending the request to the first chunk carrying generated text (content or
reasoning). Every later text-bearing chunk is timestamped, giving inter-token latencies (ITL).
Parsing is separated from I/O (``consume_*``) and takes an injectable clock so the timing logic is
unit-testable without a server.
"""

from __future__ import annotations

import json
import time
from collections.abc import AsyncIterator, Callable
from dataclasses import dataclass, field
from itertools import pairwise
from typing import Any, Literal

import httpx

from cortex_serve.servebench.workloads import Request

Protocol = Literal["openai", "ollama"]
Clock = Callable[[], float]


@dataclass
class RequestResult:
    request_id: int
    workload: str
    input_tokens: int
    max_tokens: int
    start_s: float = 0.0  # absolute clock time the request was sent
    ttft_s: float | None = None
    e2e_s: float | None = None
    chunk_times_s: list[float] = field(default_factory=list)  # relative to start
    text: str = ""
    output_tokens: int | None = None  # counted with the model tokenizer (filled by the runner)
    engine_prompt_tokens: int | None = None
    engine_output_tokens: int | None = None
    finish_reason: str | None = None
    server_timings: dict[str, float] = field(default_factory=dict)
    error: str | None = None

    @property
    def ok(self) -> bool:
        return self.error is None and self.ttft_s is not None

    @property
    def itls_s(self) -> list[float]:
        t = self.chunk_times_s
        return [b - a for a, b in pairwise(t)]

    @property
    def tpot_s(self) -> float | None:
        n = self.output_tokens or self.engine_output_tokens or len(self.chunk_times_s)
        if self.ttft_s is None or self.e2e_s is None or n < 2:
            return None
        return (self.e2e_s - self.ttft_s) / (n - 1)


def request_body(protocol: Protocol, model: str, req: Request, disable_thinking: bool) -> dict[str, Any]:
    messages = [{"role": "user", "content": req.prompt}]
    if protocol == "openai":
        body: dict[str, Any] = {
            "model": model,
            "messages": messages,
            "max_tokens": req.max_tokens,
            "temperature": 0.0,
            "stream": True,
            "stream_options": {"include_usage": True},
        }
        if disable_thinking:
            body["chat_template_kwargs"] = {"enable_thinking": False}
        return body
    body = {
        "model": model,
        "messages": messages,
        "stream": True,
        "options": {"temperature": 0.0, "num_predict": req.max_tokens},
    }
    if disable_thinking:
        body["think"] = False
    return body


def endpoint(protocol: Protocol) -> str:
    return "/v1/chat/completions" if protocol == "openai" else "/api/chat"


async def consume_openai(lines: AsyncIterator[str], res: RequestResult, t0: float, clock: Clock) -> None:
    async for line in lines:
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            break
        chunk = json.loads(payload)
        if "error" in chunk:
            res.error = str(chunk["error"])
            break
        for choice in chunk.get("choices") or []:
            delta = choice.get("delta") or {}
            piece = (delta.get("content") or "") + (delta.get("reasoning_content") or "")
            if piece:
                now = clock() - t0
                if res.ttft_s is None:
                    res.ttft_s = now
                res.chunk_times_s.append(now)
                res.text += piece
            if choice.get("finish_reason"):
                res.finish_reason = choice["finish_reason"]
        usage = chunk.get("usage")
        if usage:
            res.engine_prompt_tokens = usage.get("prompt_tokens")
            res.engine_output_tokens = usage.get("completion_tokens")
        timings = chunk.get("timings")
        if timings:
            res.server_timings = {k: float(v) for k, v in timings.items() if isinstance(v, int | float)}


async def consume_ollama(lines: AsyncIterator[str], res: RequestResult, t0: float, clock: Clock) -> None:
    async for line in lines:
        if not line.strip():
            continue
        chunk = json.loads(line)
        if "error" in chunk:
            res.error = str(chunk["error"])
            break
        msg = chunk.get("message") or {}
        piece = (msg.get("content") or "") + (msg.get("thinking") or "")
        if piece:
            now = clock() - t0
            if res.ttft_s is None:
                res.ttft_s = now
            res.chunk_times_s.append(now)
            res.text += piece
        if chunk.get("done"):
            res.finish_reason = chunk.get("done_reason")
            res.engine_prompt_tokens = chunk.get("prompt_eval_count")
            res.engine_output_tokens = chunk.get("eval_count")
            for key in ("prompt_eval_duration", "eval_duration", "load_duration", "total_duration"):
                if key in chunk:
                    res.server_timings[key + "_ms"] = chunk[key] / 1e6
            break


async def send(
    client: httpx.AsyncClient,
    base_url: str,
    protocol: Protocol,
    model: str,
    req: Request,
    disable_thinking: bool = True,
    clock: Clock = time.perf_counter,
) -> RequestResult:
    res = RequestResult(
        request_id=req.id, workload=req.workload, input_tokens=req.input_tokens, max_tokens=req.max_tokens
    )
    body = request_body(protocol, model, req, disable_thinking)
    t0 = clock()
    res.start_s = t0
    try:
        async with client.stream("POST", base_url + endpoint(protocol), json=body) as resp:
            if resp.status_code != 200:
                res.error = f"HTTP {resp.status_code}: {(await resp.aread())[:300]!r}"
                return res
            consume = consume_openai if protocol == "openai" else consume_ollama
            await consume(resp.aiter_lines(), res, t0, clock)
    except (httpx.HTTPError, json.JSONDecodeError) as exc:
        res.error = f"{type(exc).__name__}: {exc}"
    res.e2e_s = clock() - t0
    if res.error is None and res.ttft_s is None:
        res.error = "no tokens received"
    return res
