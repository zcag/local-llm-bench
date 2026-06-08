"""L0 — engine bake-off. Identical model (Qwen3-Coder-30B-A3B) across every
engine, identical scenarios, the box single-tenant. Each engine at its best:
full GPU offload, flash-attention, fixed large context, temp=0 + seed.

Run ON tardis:  uv run python -m bench.run_l0 [--soak 480]
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

CTX = 34816  # must exceed the 32k prefill scenario
GGUF = os.path.expanduser("~/models/gguf/qwen3-coder-30b/Qwen3-Coder-30B-A3B-Instruct-Q4_K_M.gguf")
MLX_MODEL = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ"
LOGDIR = "/tmp/llmbench"
os.makedirs(LOGDIR, exist_ok=True)


def build_engines() -> list:
    """Engine instances whose artifacts exist on this box (others skipped)."""
    out = [("mlx", "4bit-DWQ", MLX(MLX_MODEL, config.MLX_PORT))]
    if os.path.exists(GGUF):
        out.append(("llama.cpp", "Q4_K_M", LlamaCpp(GGUF, config.LLAMACPP_PORT, ctx=CTX,
                                                    alias="qwen3-coder-30b")))
        out.append(("ollama", "Q4_K_M", Ollama("qwen3coder-q4:latest", config.OLLAMA_CHAT_PORT)))
    if os.path.exists(LMS):
        out.append(("lmstudio", "Q4_K_M(gguf)",
                    LMStudio("qwen3-coder-30b", config.LMSTUDIO_PORT, ctx=CTX)))
    return out


async def bench_engine(engine_name: str, quant: str, eng, soak_s: float, soak_only: bool = False):
    base = eng.base_url
    model = eng.served_id()
    print(f"\n=== {engine_name} ({quant}) :{eng.port} ===")

    # ollama: bring up serve, register the gguf, set big context
    if engine_name == "ollama":
        os.environ["OLLAMA_CONTEXT_LENGTH"] = str(CTX)
        eng.start(log_path=f"{LOGDIR}/ollama.log", ready_timeout=120)
        print("  registering gguf into ollama…")
        register_gguf("qwen3coder-q4:latest", GGUF, eng.port)
    else:
        eng.start(log_path=f"{LOGDIR}/{engine_name}.log", ready_timeout=600)
    print("  ready, warmup…")
    await warmup(base, model, n=2)

    if soak_only:
        if soak_s > 0:
            print(f"  soak {soak_s:.0f}s…")
            sk = await soak(base, model, scenarios.decode_messages(), duration_s=soak_s,
                            max_tokens=256, extra=scenarios.IGNORE_EOS)
            write({"layer": "L0", "engine": engine_name, "quant": quant,
                   "model": "qwen3-coder-30b", "scenario": "soak", "kind": "thermal", **sk})
            print(f"  soak: first={sk.get('decode_tps_first_min')} last={sk.get('decode_tps_last_min')} "
                  f"throttle={sk.get('throttle_pct')}% per_min={sk.get('per_min')}")
        eng.stop()
        time.sleep(5)
        return

    # 1) single-stream perf scenarios (decode + prefill at lengths) + memory
    for label, msgs, mx, kind in scenarios.perf_scenarios():
        with Sampler(proc_match=eng.proc_match) as s:
            lvl = await run_level(base, model, msgs, concurrency=1, n=3, max_tokens=mx,
                                  extra=scenarios.IGNORE_EOS if kind == "decode" else None)
        ex = lvl.samples[0] if lvl.samples else {}
        row = {
            "layer": "L0", "engine": engine_name, "quant": quant, "model": "qwen3-coder-30b",
            "scenario": label, "kind": kind, "concurrency": 1,
            "ok": lvl.ok, "fail": lvl.fail,
            "ttft_p50": round(lvl.ttft_p50, 3),
            "decode_tps": round(lvl.decode_tps_median, 1),
            "prefill_tps": round(ex.get("prefill_tps", 0), 0),
            "prompt_tokens": ex.get("prompt_tokens", 0),
            "out_tokens": ex.get("completion_tokens", 0),
            "mem_used_gb": round(s.stats.used_peak_gb, 1),
            "mem_wired_gb": round(s.stats.wired_peak_gb, 1),
            "proc_rss_gb": round(s.stats.proc_rss_peak_gb, 1),
            "errors": lvl.errors,
        }
        write(row)
        print(f"  {label:11s} ok={lvl.ok}/{lvl.fail} ttft={lvl.ttft_p50:.2f}s "
              f"decode={lvl.decode_tps_median:.1f} prefill={ex.get('prefill_tps',0):.0f} "
              f"wired={s.stats.wired_peak_gb:.1f}G {lvl.errors if lvl.errors else ''}")

    # 2) concurrency sweep on decode (system throughput + latency under load)
    for c in scenarios.CONCURRENCY_LEVELS:
        n = c * 4
        lvl = await run_level(base, model, scenarios.decode_messages(), concurrency=c,
                              n=n, max_tokens=128, extra=scenarios.IGNORE_EOS)
        row = {
            "layer": "L0", "engine": engine_name, "quant": quant, "model": "qwen3-coder-30b",
            "scenario": "concurrency", "kind": "throughput", "concurrency": c,
            "ok": lvl.ok, "fail": lvl.fail,
            "ttft_p50": round(lvl.ttft_p50, 3), "ttft_p90": round(lvl.ttft_p90, 3),
            "system_tps": round(lvl.system_tps, 1),
            "decode_tps_per_stream": round(lvl.decode_tps_median, 1),
            "errors": lvl.errors,
        }
        write(row)
        print(f"  c={c:<2d} system={lvl.system_tps:6.1f} t/s  ttft_p50={lvl.ttft_p50:.2f} "
              f"ttft_p90={lvl.ttft_p90:.2f} ok={lvl.ok}/{lvl.fail} {lvl.errors if lvl.errors else ''}")

    # 3) soak — sustained decode, thermal throttle
    if soak_s > 0:
        print(f"  soak {soak_s:.0f}s…")
        sk = await soak(base, model, scenarios.decode_messages(), duration_s=soak_s,
                        max_tokens=256, extra=scenarios.IGNORE_EOS)
        write({"layer": "L0", "engine": engine_name, "quant": quant,
               "model": "qwen3-coder-30b", "scenario": "soak", "kind": "thermal", **sk})
        print(f"  soak: first={sk.get('decode_tps_first_min')} last={sk.get('decode_tps_last_min')} "
              f"throttle={sk.get('throttle_pct')}%")

    eng.stop()
    time.sleep(5)  # let memory release before next engine


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--soak", type=float, default=480, help="soak seconds per engine (0 to skip)")
    ap.add_argument("--soak-only", action="store_true", help="skip perf+concurrency, only soak")
    ap.add_argument("--only", default="", help="comma list: mlx,llama.cpp,ollama,lmstudio")
    args = ap.parse_args()

    engines = build_engines()
    if args.only:
        keep = set(args.only.split(","))
        engines = [e for e in engines if e[0] in keep]
    print("engines:", [e[0] for e in engines])

    with quiesce.quiesced():
        quiesce.assert_idle()
        for name, quant, eng in engines:
            try:
                await bench_engine(name, quant, eng, args.soak, soak_only=args.soak_only)
            except Exception as e:  # noqa: BLE001 — one engine failing shouldn't sink the run
                print(f"  !! {name} failed: {type(e).__name__}: {e}")
                write({"layer": "L0", "engine": name, "scenario": "ERROR", "error": str(e)})
                try:
                    eng.stop()
                except Exception:
                    pass
    print("\nL0 done -> results/runs.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
