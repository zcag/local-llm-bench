"""Tool-calling eval, BFCL-category structured (our transparent implementation —
NOT the official bfcl-eval package, which would need a custom handler for a local
endpoint and pulls vllm/sentence-transformers). Covers BFCL's core categories:
  simple            — one obvious call, correct args
  multiple          — pick the right tool among several
  parallel          — multiple independent calls in one turn
  parallel_multiple — multiple calls across different tools
  irrelevance       — NO tool fits → must NOT call (over-calling penalty)
Auto-graded by AST-style arg checks. Reported per-category + overall. Runs against
whatever engine serves the model (mlx for Qwen3; llama.cpp --jinja for gpt-oss/
devstral/qwen2.5 whose tool formats mlx_lm doesn't parse — F5).
"""
from __future__ import annotations
import json
import httpx
from ..load.client import chat_once


def _f(name, desc, props, required):
    return {"type": "function", "function": {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": required}}}

WEATHER = _f("get_weather", "Current weather for a city",
             {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}}, ["city"])
STOCK = _f("get_stock_price", "Latest stock price for a ticker",
           {"ticker": {"type": "string"}}, ["ticker"])
CONVERT = _f("convert_currency", "Convert an amount between currencies",
             {"amount": {"type": "number"}, "from_cur": {"type": "string"}, "to_cur": {"type": "string"}},
             ["amount", "from_cur", "to_cur"])
DISTANCE = _f("calc_distance", "Distance in km between two cities",
              {"a": {"type": "string"}, "b": {"type": "string"}}, ["a", "b"])
TRANSLATE = _f("translate", "Translate text to a target language",
               {"text": {"type": "string"}, "target_lang": {"type": "string"}}, ["text", "target_lang"])


def _calls(tcs):
    out = []
    for tc in tcs or []:
        fn = tc.get("function", {})
        raw = fn.get("arguments", "")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            args = None
        out.append((fn.get("name"), args))
    return out


def _has(cs, name, **argcheck):
    """True if some call matches name with args satisfying argcheck (key->predicate/value)."""
    for n, a in cs:
        if n != name or a is None:
            continue
        ok = True
        for k, want in argcheck.items():
            v = a.get(k)
            if callable(want):
                ok &= bool(want(v))
            else:
                ok &= (str(v).lower() == str(want).lower())
        if ok:
            return True
    return False


# (id, category, tools, user, checker(calls)->bool)
CASES = [
    ("simple_weather", "simple", [WEATHER], "What's the weather in Istanbul?",
     lambda cs: len(cs) == 1 and _has(cs, "get_weather", city=lambda v: "istanbul" in str(v).lower())),
    ("simple_stock", "simple", [STOCK], "What's Apple's stock price? Ticker AAPL.",
     lambda cs: len(cs) == 1 and _has(cs, "get_stock_price", ticker=lambda v: str(v).upper() == "AAPL")),
    ("simple_enum", "simple", [WEATHER], "Weather in Berlin in fahrenheit.",
     lambda cs: _has(cs, "get_weather", unit="fahrenheit")),

    ("multiple_pick_stock", "multiple", [WEATHER, STOCK, DISTANCE], "What is Tesla (TSLA) trading at?",
     lambda cs: len(cs) == 1 and _has(cs, "get_stock_price", ticker=lambda v: str(v).upper() == "TSLA")),
    ("multiple_pick_dist", "multiple", [WEATHER, STOCK, DISTANCE], "How far is Paris from Rome?",
     lambda cs: len(cs) == 1 and _has(cs, "calc_distance")),

    ("parallel_weather", "parallel", [WEATHER], "Compare the weather in Tokyo and Paris.",
     lambda cs: len([c for c in cs if c[0] == "get_weather" and c[1] is not None]) >= 2),
    ("parallel_stocks", "parallel", [STOCK], "Get prices for AAPL, MSFT and GOOG.",
     lambda cs: len([c for c in cs if c[0] == "get_stock_price"]) >= 3),

    ("parmulti_weather_stock", "parallel_multiple", [WEATHER, STOCK],
     "What's the weather in Oslo and what is NVDA stock at?",
     lambda cs: _has(cs, "get_weather") and _has(cs, "get_stock_price")),
    ("parmulti_convert_translate", "parallel_multiple", [CONVERT, TRANSLATE],
     "Convert 100 USD to EUR, and translate 'good morning' to Spanish.",
     lambda cs: _has(cs, "convert_currency") and _has(cs, "translate")),

    ("irrelevance_fact", "irrelevance", [WEATHER, STOCK],
     "Who wrote the novel War and Peace? Answer directly.",
     lambda cs: len(cs) == 0),
    ("irrelevance_chat", "irrelevance", [CONVERT, DISTANCE],
     "Thanks, that's all I needed for today!",
     lambda cs: len(cs) == 0),
]


async def run(base_url: str, model: str) -> dict:
    results = []
    async with httpx.AsyncClient() as client:
        for cid, cat, tools, user, check in CASES:
            r = await chat_once(client, base_url, model, [{"role": "user", "content": user}],
                                tools=tools, max_tokens=512)
            if not r["ok"]:
                results.append({"id": cid, "cat": cat, "passed": False, "reason": r["error"][:80]})
                continue
            cs = _calls(r["message"].get("tool_calls"))
            try:
                passed = bool(check(cs))
                reason = "ok" if passed else f"calls={[(n, a) for n, a in cs]}"[:90]
            except Exception as e:  # noqa: BLE001
                passed, reason = False, f"check-err {e}"
            results.append({"id": cid, "cat": cat, "passed": passed, "reason": reason})
    n = len(results)
    p = sum(1 for r in results if r["passed"])
    bycat = {}
    for r in results:
        d = bycat.setdefault(r["cat"], [0, 0]); d[0] += 1; d[1] += int(r["passed"])
    return {"eval": "bfcl_style", "passed": p, "total": n, "score": round(p / n, 3),
            "by_category": {k: f"{v[1]}/{v[0]}" for k, v in bycat.items()}, "cases": results}
