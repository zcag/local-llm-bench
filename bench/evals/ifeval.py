"""Instruction-following eval (IFEval-style, our transparent implementation):
verifiable instructions graded programmatically — no judge needed. Each case is a
prompt with a machine-checkable constraint (format, length, keywords, casing,
structure). Measures whether the model OBEYS precise instructions, independent of
content quality.
"""
from __future__ import annotations
import json
import re
import httpx
from ..load.client import chat_once


def _words(t):
    return re.findall(r"\b\w+\b", t)


CASES = [
    ("lowercase", "Write one sentence about the ocean, in all lowercase letters. No capital letters at all.",
     lambda t: t.strip() != "" and t == t.lower()),
    ("uppercase", "Reply with the word HELLO in all capital letters and nothing else.",
     lambda t: t.strip().upper() == "HELLO" and t.strip() == t.strip().upper()),
    ("exact_words", "Write a sentence about cats that is exactly 5 words long.",
     lambda t: len(_words(t.strip().rstrip("."))) == 5),
    ("min_words", "Describe the sun in at least 50 words.",
     lambda t: len(_words(t)) >= 50),
    ("three_bullets", "List exactly three benefits of sleep as markdown bullet points (lines starting with '- ').",
     lambda t: sum(1 for l in t.splitlines() if l.strip().startswith("- ")) == 3),
    ("keyword_thrice", "Write about gardens. Use the word 'bloom' exactly three times.",
     lambda t: len(re.findall(r"\bbloom\b", t, re.I)) == 3),
    ("forbidden_word", "Describe a forest without using the letter 'e'.",
     lambda t: "e" not in t.lower()),
    ("json_only", "Return a JSON object with keys 'name' and 'age' for a person named Alice aged 30. JSON only, no prose.",
     lambda t: _is_json_with(t, {"name", "age"})),
    ("end_phrase", "Write two sentences about coffee. End your entire response with the exact phrase: THE END",
     lambda t: t.rstrip().endswith("THE END")),
    ("title_wrap", "Give a title for a sci-fi novel, wrapped in double angle brackets like <<title>>.",
     lambda t: bool(re.search(r"<<[^>]+>>", t))),
    ("no_commas", "Write a sentence about rivers that contains no commas at all.",
     lambda t: t.strip() != "" and "," not in t),
    ("numbered_list", "List four primary colors as a numbered list (1. 2. 3. 4.).",
     lambda t: sum(1 for l in t.splitlines() if re.match(r"\s*[1-4]\.", l)) == 4),
]


def _is_json_with(t, keys):
    m = re.search(r"\{.*\}", t, re.DOTALL)
    if not m:
        return False
    try:
        obj = json.loads(m.group(0))
    except json.JSONDecodeError:
        return False
    return keys.issubset(set(obj.keys()))


async def run(base_url: str, model: str) -> dict:
    results = []
    async with httpx.AsyncClient() as client:
        for cid, prompt, check in CASES:
            r = await chat_once(client, base_url, model, [{"role": "user", "content": prompt}],
                                max_tokens=512)
            txt = (r.get("message", {}).get("content") or "") if r["ok"] else ""
            try:
                passed = bool(check(txt))
            except Exception:  # noqa: BLE001
                passed = False
            results.append({"id": cid, "passed": passed, "got": txt.strip()[:60] if not passed else ""})
    n = len(results)
    p = sum(1 for r in results if r["passed"])
    return {"eval": "ifeval_style", "passed": p, "total": n, "score": round(p / n, 3), "cases": results}
