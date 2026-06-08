"""L1 — model x quant. Fixed engine (MLX, the L0 winner among MLX-capable), the
box single-tenant. Each model: perf (decode + prefill at context) + quality
aspects (coding / tool-calling / long-context). temp=0 + seed throughout.

Run ON tardis:  PYTHONUNBUFFERED=1 uv run python -m bench.run_l1 [--only tag,tag] [--he-limit N]
"""
from __future__ import annotations
import argparse
import asyncio
import json
import os
import time

from . import config, quiesce, scenarios
from .metrics import Sampler
from .results import write
from .load.runner import run_level, warmup
from .engines.mlx import MLX
from .zoo import ZOO
from .evals import toolcalling, longcontext, humaneval


async def perf(base, model, tag):
    for label, msgs, mx, kind in scenarios.perf_scenarios():
        with Sampler(proc_match="mlx_lm.server") as s:
            lvl = await run_level(base, model, msgs, concurrency=1, n=3, max_tokens=mx,
                                  extra=scenarios.IGNORE_EOS if kind == "decode" else None)
        ex = lvl.samples[0] if lvl.samples else {}
        write({"layer": "L1", "model_tag": tag, "model": model, "scenario": label, "kind": kind,
               "ttft_p50": round(lvl.ttft_p50, 3), "decode_tps": round(lvl.decode_tps_median, 1),
               "prefill_tps": round(ex.get("prefill_tps", 0), 0),
               "prompt_tokens": ex.get("prompt_tokens", 0), "out_tokens": ex.get("completion_tokens", 0),
               "mem_used_gb": round(s.stats.used_peak_gb, 1), "mem_wired_gb": round(s.stats.wired_peak_gb, 1),
               "ok": lvl.ok, "fail": lvl.fail})
        print(f"    {label:11s} decode={lvl.decode_tps_median:.1f} prefill={ex.get('prefill_tps',0):.0f} "
              f"wired={s.stats.wired_peak_gb:.1f}G ok={lvl.ok}/{lvl.fail}")


DETAIL_DIR = os.path.join(os.path.dirname(__file__), "..", "results", "l1_evals")


async def quality(base, model, tag, he_limit):
    os.makedirs(DETAIL_DIR, exist_ok=True)
    for name, coro in [
        ("toolcalling", toolcalling.run(base, model)),
        ("longcontext", longcontext.run(base, model)),
        ("humaneval+", humaneval.run(base, model, limit=he_limit)),
    ]:
        try:
            res = await coro
        except Exception as e:  # noqa: BLE001
            res = {"eval": name, "error": f"{type(e).__name__}: {e}"}
        # full detail (incl. per-case) to a side file; compact summary to main jsonl
        with open(os.path.join(DETAIL_DIR, f"{tag}__{name.replace('+','plus')}.json"), "w") as f:
            json.dump({"model_tag": tag, "model": model, **res}, f, indent=2)
        compact = {k: v for k, v in res.items() if k not in ("cases", "raw")}
        write({"layer": "L1", "model_tag": tag, "model": model, **compact})
        print(f"    {name:12s} -> " +
              str({k: v for k, v in res.items()
                   if k in ("score", "pass@1_base", "pass@1_plus", "passed", "total", "by_length", "error")}))


async def bench_model(tag, repo, he_limit):
    print(f"\n=== {tag}  ({repo}) ===")
    eng = MLX(repo, config.MLX_PORT)
    eng.start(log_path=f"/tmp/llmbench/l1-{tag}.log", ready_timeout=900)
    print("  ready, warmup…")
    await warmup(eng.base_url, repo, n=2)
    await perf(eng.base_url, repo, tag)
    await quality(eng.base_url, repo, tag, he_limit)
    eng.stop()
    time.sleep(6)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="", help="comma list of model_tags")
    ap.add_argument("--he-limit", type=int, default=None, help="HumanEval+ problem cap (default full 164)")
    args = ap.parse_args()
    zoo = ZOO if not args.only else [z for z in ZOO if z[0] in set(args.only.split(","))]
    print("models:", [z[0] for z in zoo])

    with quiesce.quiesced():
        quiesce.assert_idle()
        for tag, repo, sz, note in zoo:
            try:
                await bench_model(tag, repo, args.he_limit)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {tag} failed: {type(e).__name__}: {e}")
                write({"layer": "L1", "model_tag": tag, "scenario": "ERROR", "error": str(e)})
    print("\nL1 done -> results/runs.jsonl")


if __name__ == "__main__":
    asyncio.run(main())
