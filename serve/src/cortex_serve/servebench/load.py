"""Load generation: closed-loop fixed concurrency and open-loop Poisson arrivals."""

from __future__ import annotations

import asyncio
import random
import time
from collections.abc import Callable, Coroutine, Sequence
from dataclasses import dataclass
from typing import Any

from cortex_serve.servebench.client import RequestResult
from cortex_serve.servebench.workloads import Request

Send = Callable[[Request], Coroutine[Any, Any, RequestResult]]


@dataclass
class LevelRun:
    results: list[RequestResult]
    wall_s: float  # measured phase only (first send -> last completion)


async def _closed_loop(send: Send, requests: Sequence[Request], concurrency: int) -> list[RequestResult]:
    queue: asyncio.Queue[Request] = asyncio.Queue()
    for r in requests:
        queue.put_nowait(r)
    results: list[RequestResult] = []

    async def worker() -> None:
        while True:
            try:
                req = queue.get_nowait()
            except asyncio.QueueEmpty:
                return
            results.append(await send(req))

    await asyncio.gather(*(worker() for _ in range(concurrency)))
    return results


async def run_concurrency(
    send: Send, warmup: Sequence[Request], measured: Sequence[Request], concurrency: int
) -> LevelRun:
    """Closed loop: ``concurrency`` users, each sends its next request as soon as the last finishes."""
    if warmup:
        await _closed_loop(send, warmup, concurrency)
    t0 = time.perf_counter()
    results = await _closed_loop(send, measured, concurrency)
    return LevelRun(results=results, wall_s=time.perf_counter() - t0)


async def run_poisson(
    send: Send, warmup: Sequence[Request], measured: Sequence[Request], rate: float, seed: int
) -> LevelRun:
    """Open loop: requests arrive as a Poisson process at ``rate`` req/s regardless of completions."""
    if warmup:
        await _closed_loop(send, warmup, 1)
    rng = random.Random(seed)
    t0 = time.perf_counter()
    tasks: list[asyncio.Task[RequestResult]] = []
    for req in measured:
        tasks.append(asyncio.create_task(send(req)))
        await asyncio.sleep(rng.expovariate(rate))
    results = list(await asyncio.gather(*tasks))
    return LevelRun(results=results, wall_s=time.perf_counter() - t0)
