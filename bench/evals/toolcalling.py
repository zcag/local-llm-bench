"""Tool-calling reliability — automatically graded, deterministic. Tests the
failure modes that break local models inside agents: wrong tool, bad/missing
args, enum violations, missing parallel calls, over-calling when no tool is
needed, and malformed JSON arguments.

Each case: (id, category, tools, messages, check). check() receives the parsed
tool_calls list and returns (passed, reason).
"""
from __future__ import annotations
import json
from dataclasses import dataclass
from typing import Callable
import httpx

from ..load.client import chat_once


def _f(name, desc, params, required):
    return {"type": "function", "function": {
        "name": name, "description": desc,
        "parameters": {"type": "object", "properties": params, "required": required}}}

WEATHER = _f("get_weather", "Get current weather for a city",
             {"city": {"type": "string"}, "unit": {"type": "string", "enum": ["celsius", "fahrenheit"]}},
             ["city"])
SEND_EMAIL = _f("send_email", "Send an email",
                {"to": {"type": "string"}, "subject": {"type": "string"}, "body": {"type": "string"}},
                ["to", "subject", "body"])
SEARCH = _f("search_flights", "Search flights",
            {"origin": {"type": "string"}, "destination": {"type": "string"},
             "passengers": {"type": "integer"}}, ["origin", "destination"])
ADD_NUMS = _f("add_to_cart", "Add items to a shopping cart",
              {"items": {"type": "array", "items": {"type": "string"}},
               "quantities": {"type": "array", "items": {"type": "integer"}}}, ["items"])


def _calls(tcs):
    """Normalize tool_calls -> list of (name, args_dict_or_None_if_bad_json)."""
    out = []
    for tc in tcs or []:
        fn = tc.get("function", {})
        raw = fn.get("arguments", "")
        try:
            args = json.loads(raw) if isinstance(raw, str) else raw
        except (json.JSONDecodeError, TypeError):
            args = None  # malformed JSON args — a real failure mode
        out.append((fn.get("name"), args))
    return out


def _first(cs):
    """(name, args) of the first call, or (None, None) if none."""
    return cs[0] if cs else (None, None)


def chk_simple(tcs):
    cs = _calls(tcs)
    name, args = _first(cs)
    if len(cs) != 1 or name != "get_weather":
        return False, f"calls={[c[0] for c in cs]}"
    if args is None:
        return False, "malformed JSON args"
    return str(args.get("city", "")).lower().startswith("istanbul"), f"city={args.get('city')}"


def chk_enum(tcs):
    name, args = _first(_calls(tcs))
    if name != "get_weather" or args is None:
        return False, f"name={name} args={args}"
    return args.get("unit") == "fahrenheit", f"unit={args.get('unit')}"


def chk_select(tcs):
    name, args = _first(_calls(tcs))
    if name != "send_email" or args is None:
        return False, f"name={name}"
    return args.get("to") == "alice@corp.com", f"to={args.get('to')}"


def chk_parallel(tcs):
    n = len([c for c in _calls(tcs) if c[0] == "get_weather" and c[1] is not None])
    return n >= 2, f"{n} weather calls"


def chk_array(tcs):
    name, args = _first(_calls(tcs))
    if name != "add_to_cart" or args is None:
        return False, f"name={name}"
    items = args.get("items")
    return isinstance(items, list) and len(items) == 2, f"items={items}"


def chk_restraint(tcs):
    n = len(_calls(tcs))
    return n == 0, f"{n} calls (want 0)"


CASES = [
    ("simple_call",   "simple",    [WEATHER], "What's the weather in Istanbul right now?", chk_simple),
    ("enum_adherence","enum",      [WEATHER], "Weather in Berlin in fahrenheit please.", chk_enum),
    ("tool_selection","select",    [WEATHER, SEND_EMAIL, SEARCH],
     "Email alice@corp.com a note titled 'Lunch' saying 'See you at 1pm'.", chk_select),
    ("parallel_calls","parallel",  [WEATHER], "Compare the weather in Tokyo and Paris.", chk_parallel),
    ("array_args",    "nested",    [ADD_NUMS], "Add 2 apples and 3 bananas to my cart.", chk_array),
    ("no_call_needed","restraint", [WEATHER, SEND_EMAIL],
     "What is the capital of France? Just answer, don't use tools.", chk_restraint),
]


async def run(base_url: str, model: str) -> dict:
    results = []
    async with httpx.AsyncClient() as client:
        for cid, cat, tools, user, check in CASES:
            msgs = [{"role": "user", "content": user}]
            r = await chat_once(client, base_url, model, msgs, tools=tools, max_tokens=512)
            if not r["ok"]:
                results.append({"id": cid, "cat": cat, "passed": False, "reason": r["error"][:80]})
                continue
            tcs = r["message"].get("tool_calls")
            try:
                passed, reason = check(tcs)
                passed = bool(passed)
            except Exception as e:  # noqa: BLE001
                passed, reason = False, f"check-err {e}"
            results.append({"id": cid, "cat": cat, "passed": passed, "reason": str(reason)[:80]})
    n = len(results)
    p = sum(1 for r in results if r["passed"])
    return {"eval": "toolcalling", "passed": p, "total": n, "score": round(p / n, 3), "cases": results}
