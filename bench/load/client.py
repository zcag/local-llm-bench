"""Async OpenAI-compatible client that records timing for one streamed completion.

One streamed request yields everything we need:
  - TTFT       : t(first content token) - t(request sent)   [prefill+queue latency]
  - decode_tps : (completion_tokens - 1) / (t_last - t_first) [pure generation rate]
  - prefill_tps: prompt_tokens / TTFT                         [approx prompt-processing]
Token counts come from server `usage` when present, else from a local tokenizer
(exact, deterministic) so engines that omit usage are still comparable.
"""
from __future__ import annotations
import json
import time
from dataclasses import dataclass, field, asdict
from typing import Optional
import httpx


@dataclass
class Sample:
    ok: bool
    ttft_s: float = 0.0
    total_s: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    decode_tps: float = 0.0
    prefill_tps: float = 0.0
    finish_reason: str = ""
    error: str = ""
    text: str = ""

    def as_row(self) -> dict:
        d = asdict(self)
        d.pop("text", None)
        return d


async def stream_chat(
    client: httpx.AsyncClient,
    base_url: str,
    model: str,
    messages: list[dict],
    max_tokens: int = 256,
    temperature: float = 0.0,
    seed: int = 7,
    tokenizer=None,
    extra: Optional[dict] = None,
) -> Sample:
    body = {
        "model": model,
        "messages": messages,
        "max_tokens": max_tokens,
        "temperature": temperature,
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    if seed is not None:
        body["seed"] = seed
    if extra:
        body.update(extra)

    t0 = time.perf_counter()
    t_first: Optional[float] = None
    t_last = t0
    chunks = 0
    parts: list[str] = []
    usage = None
    finish = ""
    try:
        async with client.stream("POST", f"{base_url}/chat/completions", json=body, timeout=600) as r:
            if r.status_code != 200:
                err = (await r.aread()).decode("utf-8", "replace")[:300]
                return Sample(ok=False, error=f"HTTP {r.status_code}: {err}")
            async for line in r.aiter_lines():
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    obj = json.loads(data)
                except json.JSONDecodeError:
                    continue
                if obj.get("usage"):
                    usage = obj["usage"]
                choices = obj.get("choices") or []
                if not choices:
                    continue
                delta = choices[0].get("delta", {})
                piece = delta.get("content")
                # reasoning tokens are generated tokens too (counted in usage) — anchor
                # the decode window to the FIRST token of any kind so reasoning models
                # aren't measured over an artificially short window (F10).
                reasoning = delta.get("reasoning_content") or delta.get("reasoning")
                if piece or reasoning:
                    now = time.perf_counter()
                    if t_first is None:
                        t_first = now
                    t_last = now
                if piece:
                    chunks += 1
                    parts.append(piece)
                if choices[0].get("finish_reason"):
                    finish = choices[0]["finish_reason"]
    except Exception as e:  # noqa: BLE001 — minimal handling, surface as failed sample
        return Sample(ok=False, error=f"{type(e).__name__}: {e}")

    if t_first is None:
        return Sample(ok=False, error="no content tokens", total_s=time.perf_counter() - t0)

    text = "".join(parts)
    total_s = t_last - t0
    ttft = t_first - t0
    gen_s = max(t_last - t_first, 1e-9)

    if usage:
        ptok = int(usage.get("prompt_tokens", 0))
        ctok = int(usage.get("completion_tokens", 0)) or chunks
    elif tokenizer is not None:
        ptok = _count(tokenizer, messages)
        ctok = len(tokenizer.encode(text))
    else:
        ptok, ctok = 0, chunks

    return Sample(
        ok=True,
        ttft_s=ttft,
        total_s=total_s,
        prompt_tokens=ptok,
        completion_tokens=ctok,
        # decode rate needs enough generated tokens to be meaningful; a 1-2 token
        # reply (e.g. prefill scenarios) over ~0s would otherwise explode to ~1e9.
        decode_tps=(ctok - 1) / gen_s if ctok >= 8 else 0.0,
        prefill_tps=ptok / ttft if ttft > 0 and ptok else 0.0,
        finish_reason=finish,
        text=text,
    )


async def chat_once(
    client: httpx.AsyncClient, base_url: str, model: str, messages: list[dict],
    tools: Optional[list] = None, tool_choice: str = "auto", max_tokens: int = 1024,
    temperature: float = 0.0, seed: int = 7, response_format: Optional[dict] = None,
) -> dict:
    """Non-streaming completion — returns the raw OpenAI message (content +
    tool_calls) plus latency. Used by evals that need the full structured reply."""
    body = {"model": model, "messages": messages, "max_tokens": max_tokens,
            "temperature": temperature, "stream": False, "seed": seed}
    if tools:
        body["tools"] = tools
        body["tool_choice"] = tool_choice
    if response_format:
        body["response_format"] = response_format
    import time as _t
    t0 = _t.perf_counter()
    try:
        r = await client.post(f"{base_url}/chat/completions", json=body, timeout=600)
        dt = _t.perf_counter() - t0
        if r.status_code != 200:
            return {"ok": False, "error": f"HTTP {r.status_code}: {r.text[:300]}", "latency_s": dt}
        msg = r.json()["choices"][0]["message"]
        return {"ok": True, "message": msg, "latency_s": dt}
    except Exception as e:  # noqa: BLE001
        return {"ok": False, "error": f"{type(e).__name__}: {e}", "latency_s": _t.perf_counter() - t0}


def _count(tokenizer, messages: list[dict]) -> int:
    try:
        ids = tokenizer.apply_chat_template(messages, add_generation_prompt=True)
        return len(ids)
    except Exception:
        return sum(len(tokenizer.encode(m.get("content", ""))) for m in messages)
