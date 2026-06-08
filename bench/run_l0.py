"""L0 — engine bake-off. Identical model across engines, box single-tenant, each
engine at its BEST config, verified. Two phases with DIFFERENT optimal configs:
  - single-stream: parallel/concurrency = 1 (max per-request latency/throughput)
  - concurrency:   parallel/concurrency = 16 (real batching ceiling)
Fairness: cross-engine quant is matched-bpw — MLX-4bit(vanilla) vs GGUF-Q4_K_M
(~4.5bpw both); MLX-4bit-DWQ is reported separately as MLX's best. Every row
records its exact config (provenance) + cache-busting on (unique prefix/request).

Run ON tardis:  PYTHONUNBUFFERED=1 .venv/bin/python -m bench.run_l0 --phase both [--soak 480]
"""
from __future__ import annotations
import argparse
import asyncio
import os
import time

from . import config, quiesce, scenarios
from .metrics import Sampler
from .results import write
from .load.runner import run_level, warmup, soak
from .engines.mlx import MLX
from .engines.llamacpp import LlamaCpp
from .engines.ollama import Ollama, register_gguf
from .engines.lmstudio import LMStudio, LMS

CTX = 34816
GGUF = os.path.expanduser("~/models/gguf/qwen3-coder-30b/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")
MLX_4BIT = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"       # matched-bpw vs Q4_K_M
MLX_DWQ = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ"    # MLX best (bonus row)
LOGDIR = "/tmp/llmbench"
os.makedirs(LOGDIR, exist_ok=True)


def build_engines(parallel: int, only: set | None = None) -> list:
    """Engines configured for the given parallelism (1=single-stream, 16=concurrency)."""
    out = [
        ("mlx-4bit", "4bit", MLX(MLX_4BIT, config.MLX_PORT,
                                 decode_concurrency=parallel, prompt_concurrency=parallel)),
        ("mlx-4bit-dwq", "4bit-DWQ", MLX(MLX_DWQ, config.MLX_PORT,
                                         decode_concurrency=parallel, prompt_concurrency=parallel)),
    ]
    if os.path.exists(GGUF):
        out.append(("llama.cpp", "Q4_K_M", LlamaCpp(GGUF, config.LLAMACPP_PORT, ctx=CTX,
                                                    alias="qwen3-coder-30b", parallel=parallel)))
        out.append(("ollama", "Q4_K_M", Ollama("qwen3coder-q4:latest", config.OLLAMA_CHAT_PORT,
                                               parallel=parallel, ctx=CTX)))
    if os.path.exists(LMS):
        out.append(("lmstudio", "Q4_K_M(gguf)",
                    LMStudio("qwen3-coder-30b", config.LMSTUDIO_PORT, ctx=CTX)))
    if only:
        out = [e for e in out if e[0] in only]
    return out


def _start(engine_name, eng):
    if engine_name == "ollama":
        eng.start(log_path=f"{LOGDIR}/ollama.log", ready_timeout=120)
        register_gguf("qwen3coder-q4:latest", GGUF, eng.port)
    else:
        eng.start(log_path=f"{LOGDIR}/{engine_name}.log", ready_timeout=600)


def _verify(engine_name) -> str:
    """Pull config-took-effect evidence from the engine log (the gate)."""
    log = f"{LOGDIR}/{engine_name}.log"
    if not os.path.exists(log):
        return ""
    txt = open(log, errors="ignore").read()
    ev = []
    if "GPULayers:" in txt:
        import re
        m = re.search(r"GPULayers:\d+\[[^\]]*\]", txt)
        if m: ev.append(m.group(0))
    if "FlashAttention:Enabled" in txt or "flash_attn" in txt.lower():
        ev.append("FA")
    return "; ".join(ev)


async def single_stream(engine_name, quant, eng, soak_s):
    base, model = eng.base_url, eng.served_id()
    print(f"\n=== [single] {engine_name} ({quant}) :{eng.port} ===")
    _start(engine_name, eng)
    await warmup(base, model, n=2)
    ev = _verify(engine_name)
    print(f"  config: {eng.config_str()[:90]}  evidence: {ev or '(speed-anchored)'}")

    # perf scenarios FIRST on a cold chip (the numbers that matter), soak after.
    for label, msgs, mx, kind in scenarios.perf_scenarios():
        with Sampler(proc_match=eng.proc_match) as s:
            lvl = await run_level(base, model, msgs, concurrency=1, n=3, max_tokens=mx,
                                  extra=scenarios.IGNORE_EOS if kind == "decode" else None)
        ex = lvl.samples[0] if lvl.samples else {}
        write({
            "layer": "L0", "phase": "single", "engine": engine_name, "quant": quant,
            "model": "qwen3-coder-30b", "scenario": label, "kind": kind, "concurrency": 1,
            "ok": lvl.ok, "fail": lvl.fail,
            "ttft_p50": round(lvl.ttft_p50, 3), "ttft_min": round(lvl.ttft_min, 3),
            "decode_tps": round(lvl.decode_tps_median, 1),
            "prefill_tps": round(ex.get("prefill_tps", 0), 0),
            "prompt_tokens": ex.get("prompt_tokens", 0), "out_tokens": ex.get("completion_tokens", 0),
            "mem_used_gb": round(s.stats.used_peak_gb, 1), "mem_wired_gb": round(s.stats.wired_peak_gb, 1),
            "config": eng.config_str(), "evidence": ev, "errors": lvl.errors,
        })
        print(f"  {label:11s} ok={lvl.ok}/{lvl.fail} ttft={lvl.ttft_p50:.2f}s(min {lvl.ttft_min:.2f}) "
              f"decode={lvl.decode_tps_median:.1f} prefill={ex.get('prefill_tps',0):.0f} "
              f"wired={s.stats.wired_peak_gb:.1f}G {lvl.errors if lvl.errors else ''}")

    # soak last (confirmatory thermal; chip warm but throttle already ~0%)
    if soak_s > 0:
        sk = await soak(base, model, scenarios.decode_messages(), duration_s=soak_s,
                        max_tokens=256, extra=scenarios.IGNORE_EOS)
        write({"layer": "L0", "phase": "single", "engine": engine_name, "quant": quant,
               "model": "qwen3-coder-30b", "scenario": "soak", "kind": "thermal",
               "config": eng.config_str(), **sk})
        print(f"  soak: first={sk.get('decode_tps_first_min')} last={sk.get('decode_tps_last_min')} "
              f"throttle={sk.get('throttle_pct')}%")
    eng.stop(); time.sleep(5)


async def concurrency(engine_name, quant, eng):
    base, model = eng.base_url, eng.served_id()
    print(f"\n=== [concurrency] {engine_name} ({quant}) :{eng.port} ===")
    _start(engine_name, eng)
    await warmup(base, model, n=2)
    ev = _verify(engine_name)
    print(f"  config: {eng.config_str()[:90]}  evidence: {ev or '(speed-anchored)'}")
    for c in scenarios.CONCURRENCY_LEVELS:
        lvl = await run_level(base, model, scenarios.decode_messages(), concurrency=c,
                              n=c * 4, max_tokens=128, extra=scenarios.IGNORE_EOS)
        write({
            "layer": "L0", "phase": "concurrency", "engine": engine_name, "quant": quant,
            "model": "qwen3-coder-30b", "scenario": "concurrency", "kind": "throughput",
            "concurrency": c, "ok": lvl.ok, "fail": lvl.fail,
            "ttft_p50": round(lvl.ttft_p50, 3), "ttft_p90": round(lvl.ttft_p90, 3),
            "system_tps": round(lvl.system_tps, 1),
            "decode_tps_per_stream": round(lvl.decode_tps_median, 1),
            "config": eng.config_str(), "evidence": ev, "errors": lvl.errors,
        })
        print(f"  c={c:<2d} system={lvl.system_tps:6.1f} t/s  ttft_p50={lvl.ttft_p50:.2f} "
              f"ttft_p90={lvl.ttft_p90:.2f} ok={lvl.ok}/{lvl.fail} {lvl.errors if lvl.errors else ''}")
    eng.stop(); time.sleep(5)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--phase", choices=["single", "concurrency", "both"], default="both")
    ap.add_argument("--soak", type=float, default=480, help="cold soak seconds (single phase; 0 to skip)")
    ap.add_argument("--only", default="", help="comma engine names")
    args = ap.parse_args()
    only = set(args.only.split(",")) if args.only else None

    with quiesce.quiesced():
        quiesce.assert_idle()
        if args.phase in ("single", "both"):
            for name, quant, eng in build_engines(parallel=1, only=only):
                try:
                    await single_stream(name, quant, eng, args.soak)
                except Exception as e:  # noqa: BLE001
                    print(f"  !! {name} single failed: {type(e).__name__}: {e}")
                    write({"layer": "L0", "phase": "single", "engine": name, "scenario": "ERROR", "error": str(e)})
                    try: eng.stop()
                    except Exception: pass
        if args.phase in ("concurrency", "both"):
            for name, quant, eng in build_engines(parallel=16, only=only):
                try:
                    await concurrency(name, quant, eng)
                except Exception as e:  # noqa: BLE001
                    print(f"  !! {name} concurrency failed: {type(e).__name__}: {e}")
                    write({"layer": "L0", "phase": "concurrency", "engine": name, "scenario": "ERROR", "error": str(e)})
                    try: eng.stop()
                    except Exception: pass
    print("\nL0 done -> results/runs.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
