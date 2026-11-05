"""Timing logic of the streaming client, driven by a fake clock (no server needed)."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import pytest
from cortex_serve.servebench.client import RequestResult, consume_ollama, consume_openai, request_body
from cortex_serve.servebench.workloads import Request


class FakeClock:
    def __init__(self, times: list[float]) -> None:
        self.times = iter(times)

    def __call__(self) -> float:
        return next(self.times)


async def _lines(items: list[str]) -> AsyncIterator[str]:
    for item in items:
        yield item


def _sse(delta: dict[str, object], **extra: object) -> str:
    return "data: " + json.dumps(
        {"choices": [{"index": 0, "delta": delta, "finish_reason": extra.pop("finish", None)}], **extra}
    )


def _result() -> RequestResult:
    return RequestResult(request_id=0, workload="w", input_tokens=10, max_tokens=4)


def test_openai_ttft_and_itl() -> None:
    lines = [
        _sse({"role": "assistant", "content": None}),  # role-only chunk: not a token
        _sse({"content": "Hel"}),
        _sse({"content": "lo"}),
        "",
        _sse({"reasoning_content": " th"}),  # reasoning text counts as generated output
        _sse({}, finish="length"),
        "data: "
        + json.dumps(
            {
                "choices": [],
                "usage": {"prompt_tokens": 12, "completion_tokens": 3},
                "timings": {"predicted_per_second": 20.5},
            }
        ),
        "data: [DONE]",
    ]
    res = _result()
    asyncio.run(consume_openai(_lines(lines), res, t0=100.0, clock=FakeClock([100.5, 100.6, 100.8])))
    assert res.ttft_s == 0.5
    assert res.chunk_times_s == pytest.approx([0.5, 0.6, 0.8])
    assert res.itls_s == pytest.approx([0.1, 0.2])
    assert res.text == "Hello th"
    assert res.finish_reason == "length"
    assert (res.engine_prompt_tokens, res.engine_output_tokens) == (12, 3)
    assert res.server_timings == {"predicted_per_second": 20.5}


def test_tpot_uses_tokenizer_count() -> None:
    res = _result()
    res.ttft_s, res.e2e_s, res.output_tokens = 1.0, 3.0, 5
    assert res.tpot_s == 0.5  # (3.0 - 1.0) / (5 - 1)


def test_openai_error_chunk() -> None:
    res = _result()
    asyncio.run(consume_openai(_lines(['data: {"error": {"message": "boom"}}']), res, 0.0, FakeClock([])))
    assert res.error is not None and "boom" in res.error


def test_ollama_stream() -> None:
    lines = [
        json.dumps({"message": {"content": "A"}, "done": False}),
        json.dumps({"message": {"content": "B"}, "done": False}),
        json.dumps(
            {
                "message": {"content": ""},
                "done": True,
                "done_reason": "length",
                "prompt_eval_count": 9,
                "eval_count": 2,
                "eval_duration": 5e8,
            }
        ),
    ]
    res = _result()
    asyncio.run(consume_ollama(_lines(lines), res, t0=0.0, clock=FakeClock([0.3, 0.45])))
    assert res.ttft_s == 0.3 and res.text == "AB"
    assert res.engine_output_tokens == 2 and res.server_timings["eval_duration_ms"] == 500.0


def test_request_bodies() -> None:
    req = Request(id=1, workload="w", prompt="hi", input_tokens=1, max_tokens=7)
    oa = request_body("openai", "m", req, disable_thinking=True)
    assert oa["max_tokens"] == 7 and oa["temperature"] == 0.0 and oa["stream"] is True
    assert oa["chat_template_kwargs"] == {"enable_thinking": False}
    ol = request_body("ollama", "m", req, disable_thinking=True)
    assert ol["options"]["num_predict"] == 7 and ol["think"] is False
