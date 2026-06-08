"""Long-context retention — needle in a haystack. A unique fact is buried at a
given depth inside filler of a target token length; the model must retrieve it.
Deterministic exact-match grading. Exposes where each model's recall falls off.
"""
from __future__ import annotations
import re
import httpx
from ..load.client import chat_once

_FILLER_SENT = (
    "The committee reviewed the quarterly logistics report and noted that "
    "shipping volumes remained stable across all regional distribution hubs. "
)
NEEDLE = "The secret access code for the Orion vault is {code}."
QUESTION = "What is the secret access code for the Orion vault? Answer with only the code."

LENGTHS = [4000, 8000, 16000, 32000]
DEPTHS = [0.1, 0.5, 0.9]


def _haystack(approx_tokens: int, depth: float, code: str) -> str:
    reps = max(1, int(approx_tokens / 18))   # ~18 tokens/sentence
    sents = [_FILLER_SENT] * reps
    pos = int(len(sents) * depth)
    sents.insert(pos, NEEDLE.format(code=code))
    return "".join(sents)


async def run(base_url: str, model: str) -> dict:
    cases = []
    async with httpx.AsyncClient() as client:
        for i, length in enumerate(LENGTHS):
            for j, depth in enumerate(DEPTHS):
                code = f"{7000 + i*100 + j*7}-XQ"   # unique per cell
                ctx = _haystack(length, depth, code)
                msgs = [
                    {"role": "system", "content": "You answer questions using only the provided document."},
                    {"role": "user", "content": ctx + "\n\n" + QUESTION},
                ]
                r = await chat_once(client, base_url, model, msgs, max_tokens=32)
                ans = (r.get("message", {}).get("content") or "") if r["ok"] else ""
                passed = code in ans.replace(" ", "")
                cases.append({"len": length, "depth": depth, "passed": passed,
                              "want": code, "got": ans.strip()[:40] if not passed else ""})
    n = len(cases)
    p = sum(1 for c in cases if c["passed"])
    # also report deepest length fully passed
    by_len = {}
    for c in cases:
        by_len.setdefault(c["len"], []).append(c["passed"])
    return {"eval": "longcontext", "passed": p, "total": n, "score": round(p / n, 3),
            "by_length": {str(k): round(sum(v) / len(v), 2) for k, v in sorted(by_len.items())},
            "cases": cases}
