"""Coding correctness via EvalPlus (HumanEval+). We generate one completion per
problem through the OpenAI endpoint, then hand the samples to EvalPlus's own
test harness for grading (reuse the recognized eval, don't re-implement pass@k).

Needs the `evalplus` package in the run env. Reports pass@1 on base HumanEval
and the harder HumanEval+ test set.
"""
from __future__ import annotations
import asyncio
import json
import os
import re
import subprocess
import tempfile
import httpx
from ..load.client import chat_once

PROMPT = ("Complete the following Python function. Return ONLY the complete "
          "function implementation in a single ```python code block, including "
          "the signature. Do not add explanations or tests.\n\n```python\n{sig}\n```")

_FENCE = re.compile(r"```(?:python)?\s*(.*?)```", re.DOTALL)


def _extract(content: str, entry_point: str) -> str:
    blocks = _FENCE.findall(content or "")
    code = blocks[-1] if blocks else (content or "")
    # keep from first import/def so stray prose before code is dropped
    m = re.search(r"^(from |import |def |class |@)", code, re.MULTILINE)
    if m:
        code = code[m.start():]
    return code.strip()


async def _gen(client, base_url, model, problems, concurrency=4):
    sem = asyncio.Semaphore(concurrency)
    samples = {}

    async def one(tid, prob):
        async with sem:
            msgs = [{"role": "user", "content": PROMPT.format(sig=prob["prompt"].strip())}]
            r = await chat_once(client, base_url, model, msgs, max_tokens=1024)
            content = r.get("message", {}).get("content", "") if r["ok"] else ""
            samples[tid] = _extract(content, prob["entry_point"])

    await asyncio.gather(*[one(tid, p) for tid, p in problems.items()])
    return samples


async def run(base_url: str, model: str, limit: int | None = None) -> dict:
    try:
        from evalplus.data import get_human_eval_plus
    except ImportError:
        return {"eval": "humaneval+", "error": "evalplus not installed"}

    problems = get_human_eval_plus()
    if limit:
        problems = dict(list(problems.items())[:limit])

    async with httpx.AsyncClient() as client:
        samples = await _gen(client, base_url, model, problems)

    # write samples in evalplus format and grade with its harness
    d = tempfile.mkdtemp(prefix="evalplus-")
    sfile = os.path.join(d, "samples.jsonl")
    with open(sfile, "w") as f:
        for tid, sol in samples.items():
            f.write(json.dumps({"task_id": tid, "solution": sol}) + "\n")

    proc = subprocess.run(
        ["python", "-m", "evalplus.evaluate", "--dataset", "humaneval", "--samples", sfile],
        capture_output=True, text=True, timeout=1800,
    )
    out = proc.stdout + proc.stderr
    base = _passat1(out, "HumanEval ")
    plus = _passat1(out, "HumanEval+")
    return {"eval": "humaneval+", "pass@1_base": base, "pass@1_plus": plus,
            "score": plus if plus is not None else 0.0, "n": len(samples),
            "raw": out[-400:] if base is None else ""}


def _passat1(text: str, marker: str):
    # evalplus prints lines like 'HumanEval (base tests)\npass@1: 0.7561'
    m = re.search(re.escape(marker) + r".*?pass@1:\s*([0-9.]+)", text, re.DOTALL)
    return round(float(m.group(1)), 4) if m else None
