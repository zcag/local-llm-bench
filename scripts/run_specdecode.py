"""Spec-decode L0 row: does mlx_lm speculative decoding (--draft-model) speed up
single-stream decode on the DWQ-30B? (Coder-Next is hybrid-cache-incompatible, so
this is the standard 30B-A3B-4bit-DWQ + a small Qwen3-0.6B draft.) Measures decode
tok/s baseline vs with-draft, same prompt. Quiesced.

Run ON tardis: PYTHONUNBUFFERED=1 .venv/bin/python scripts/run_specdecode.py
"""
from __future__ import annotations
import asyncio, sys
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from bench import config, quiesce, scenarios
from bench.engines.mlx import MLX
from bench.results import write
from bench.load.runner import run_level, warmup

TARGET = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ"
DRAFT = "mlx-community/Qwen3-0.6B-4bit"


async def measure(eng, label, draft):
    await warmup(eng.base_url, eng.served_id(), n=2)
    lvl = await run_level(eng.base_url, eng.served_id(), scenarios.decode_messages(),
                          concurrency=1, n=3, max_tokens=256, extra=scenarios.IGNORE_EOS)
    write({"layer": "L0", "phase": "spec-decode", "engine": "mlx_lm", "scenario": "decode_256",
           "model": "30b-a3b-4bit-dwq", "draft": draft or "(none)",
           "decode_tps": round(lvl.decode_tps_median, 1), "ttft_p50": round(lvl.ttft_p50, 3),
           "config": eng.config_str()})
    print(f"  {label}: decode={lvl.decode_tps_median:.1f} t/s  ttft={lvl.ttft_p50:.2f}s ok={lvl.ok}/{lvl.fail} {lvl.errors}")
    return lvl.decode_tps_median


async def main():
    with quiesce.quiesced():
        quiesce.assert_idle()
        # baseline (no draft)
        base = MLX(TARGET, config.MLX_PORT)
        base.start(log_path="/tmp/llmbench/spec-base.log", ready_timeout=600)
        b = await measure(base, "baseline (no draft)", None)
        base.stop(); await asyncio.sleep(5)
        # with draft
        try:
            spec = MLX(TARGET, config.MLX_PORT, draft_model=DRAFT)
            spec.start(log_path="/tmp/llmbench/spec-draft.log", ready_timeout=600)
            s = await measure(spec, f"spec-decode (draft {DRAFT})", DRAFT)
            spec.stop()
            print(f"\n  speedup: {s/b:.2f}x ({b:.1f} -> {s:.1f} t/s)")
            write({"layer": "L0", "phase": "spec-decode", "scenario": "summary",
                   "baseline_tps": round(b, 1), "spec_tps": round(s, 1), "speedup": round(s/b, 2)})
        except Exception as e:  # noqa: BLE001
            print(f"  spec-decode FAILED: {type(e).__name__}: {e}")
            write({"layer": "L0", "phase": "spec-decode", "scenario": "ERROR", "error": str(e)})
    print("\nspec-decode done")


if __name__ == "__main__":
    asyncio.run(main())
