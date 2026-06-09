"""F16: measure the claude-code-router (ccr) shim's overhead. Send an IDENTICAL
minimal request (a) via ccr (Anthropic API -> ccr -> proxy -> MLX) and (b) direct
to the proxy (OpenAI -> MLX). The proxy tallies tokens for both, so the delta in
proxy-observed prompt_tokens = what the shim adds per request. Attributes
claude-code's token-heaviness to shim vs its own multi-turn context growth.

Run ON tardis (live MLX on :11434): PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_f16.py
"""
from __future__ import annotations
import time, httpx, sys
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from bench.run_l2 import start_proxy, start_ccr, proxy_reset, proxy_stats, PROXY_URL, CCR_URL
from bench import config
from bench.results import write

PROMPT = "Reply with exactly one word: hello."


def main():
    served = httpx.get(f"{config.base_url(config.MLX_PORT)}/models", timeout=20).json()["data"][0]["id"]
    proxy = start_proxy(served); start_ccr()
    try:
        time.sleep(2)
        # (a) via ccr — Anthropic /v1/messages
        proxy_reset(); t0 = time.perf_counter()
        httpx.post(f"{CCR_URL}/v1/messages", timeout=120, headers={"x-api-key": "bench", "anthropic-version": "2023-06-01"},
                   json={"model": "local", "max_tokens": 16, "messages": [{"role": "user", "content": PROMPT}]})
        ccr_dt = time.perf_counter() - t0; ccr = proxy_stats()
        # (b) direct to proxy — OpenAI
        proxy_reset(); t0 = time.perf_counter()
        httpx.post(f"{PROXY_URL}/v1/chat/completions", timeout=120,
                   json={"model": "local", "max_tokens": 16, "messages": [{"role": "user", "content": PROMPT}]})
        dir_dt = time.perf_counter() - t0; direct = proxy_stats()
        res = {"layer": "L2", "scenario": "ccr_overhead",
               "ccr_prompt_tokens": ccr.get("prompt_tokens"), "direct_prompt_tokens": direct.get("prompt_tokens"),
               "shim_added_prompt_tokens": (ccr.get("prompt_tokens", 0) - direct.get("prompt_tokens", 0)),
               "ccr_latency_s": round(ccr_dt, 2), "direct_latency_s": round(dir_dt, 2),
               "shim_added_latency_s": round(ccr_dt - dir_dt, 2)}
        write(res)
        print("F16 ccr overhead:", {k: v for k, v in res.items() if k != "layer" and k != "scenario"})
    finally:
        proxy.terminate()
        import subprocess; subprocess.run(["ccr", "stop"], capture_output=True)


if __name__ == "__main__":
    main()
