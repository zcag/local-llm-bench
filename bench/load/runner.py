"""Concurrency sweep: fire batches of N requests at concurrency C, aggregate."""
from __future__ import annotations
import asyncio
import statistics
import time
from dataclasses import dataclass, field
from typing import Optional
import httpx

from .client import Sample, stream_chat


@dataclass
class LevelResult:
    concurrency: int
    n: int
    ok: int
    fail: int
    # per-request (over successful samples)
    ttft_p50: float = 0.0
    ttft_p90: float = 0.0
    decode_tps_median: float = 0.0   # single-stream generation rate
    # aggregate (system-level)
    system_tps: float = 0.0          # total completion tokens / wall window
    wall_s: float = 0.0
    errors: list[str] = field(default_factory=list)
    samples: list[dict] = field(default_factory=list)


def _pct(xs: list[float], p: float) -> float:
    if not xs:
        return 0.0
    xs = sorted(xs)
    k = (len(xs) - 1) * p
    lo, hi = int(k), min(int(k) + 1, len(xs) - 1)
    return xs[lo] + (xs[hi] - xs[lo]) * (k - lo)


async def run_level(
    base_url: str, model: str, messages: list[dict], concurrency: int,
    n: int, max_tokens: int, tokenizer=None, extra: Optional[dict] = None,
) -> LevelResult:
    sem = asyncio.Semaphore(concurrency)
    results: list[Sample] = []

    async with httpx.AsyncClient() as client:
        async def one():
            async with sem:
                return await stream_chat(client, base_url, model, messages,
                                         max_tokens=max_tokens, tokenizer=tokenizer, extra=extra)
        t0 = time.perf_counter()
        results = await asyncio.gather(*[one() for _ in range(n)])
        wall = time.perf_counter() - t0

    ok = [s for s in results if s.ok]
    fail = [s for s in results if not s.ok]
    total_ctok = sum(s.completion_tokens for s in ok)
    return LevelResult(
        concurrency=concurrency, n=n, ok=len(ok), fail=len(fail),
        ttft_p50=_pct([s.ttft_s for s in ok], 0.5),
        ttft_p90=_pct([s.ttft_s for s in ok], 0.9),
        decode_tps_median=statistics.median([s.decode_tps for s in ok]) if ok else 0.0,
        system_tps=total_ctok / wall if wall else 0.0,
        wall_s=wall,
        errors=[s.error for s in fail][:5],
        samples=[s.as_row() for s in ok],
    )


async def warmup(base_url: str, model: str, n: int = 2):
    msgs = [{"role": "user", "content": "Reply with the single word: ready."}]
    async with httpx.AsyncClient() as client:
        for _ in range(n):
            await stream_chat(client, base_url, model, msgs, max_tokens=8)
