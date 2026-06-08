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


async def soak(base_url: str, model: str, messages: list[dict], duration_s: float,
               max_tokens: int = 256, tokenizer=None, extra=None) -> dict:
    """Back-to-back single-stream decode for duration_s; report decode tok/s per
    minute-window to expose thermal throttling (sustained vs burst)."""
    import time as _t
    windows: list[float] = []        # decode_tps per request
    stamps: list[float] = []         # request end time (rel)
    t0 = _t.perf_counter()
    async with httpx.AsyncClient() as client:
        while _t.perf_counter() - t0 < duration_s:
            s = await stream_chat(client, base_url, model, messages,
                                  max_tokens=max_tokens, tokenizer=tokenizer, extra=extra)
            if s.ok and s.decode_tps:
                windows.append(s.decode_tps)
                stamps.append(_t.perf_counter() - t0)
    if not windows:
        return {"ok": False, "n": 0}
    # bucket into 60s windows
    buckets: dict[int, list[float]] = {}
    for tps, ts in zip(windows, stamps):
        buckets.setdefault(int(ts // 60), []).append(tps)
    per_min = {m: round(statistics.mean(v), 1) for m, v in sorted(buckets.items())}
    first = per_min[min(per_min)]
    last = per_min[max(per_min)]
    return {
        "ok": True, "n": len(windows), "duration_s": round(_t.perf_counter() - t0, 1),
        "decode_tps_first_min": first, "decode_tps_last_min": last,
        "throttle_pct": round((first - last) / first * 100, 1) if first else 0.0,
        "per_min": per_min,
        "decode_tps_median": round(statistics.median(windows), 1),
    }
