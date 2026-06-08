"""Validate the control plane SAFELY before any real run:
quiesce (take live services down) -> harness-managed MLX on a scratch port ->
one real request -> stop -> restore -> confirm :11434 and :11435 are back.

Uses the lighter 30B-DWQ (faster load) to prove the adapter, not Coder-Next.
Run ON tardis:  uv run python scripts/test_control.py
"""
from __future__ import annotations
import asyncio
import subprocess
import sys
import httpx

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from bench import config, quiesce
from bench.engines.mlx import MLX
from bench.load.runner import run_level

SCRATCH_PORT = 11440
MODEL = "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ"


def listeners() -> str:
    p = subprocess.run(["lsof", "-nP", "-iTCP", "-sTCP:LISTEN"], capture_output=True, text=True)
    return "\n".join(l for l in p.stdout.splitlines()
                     if any(x in l for x in ("11434", "11435", "11440", ":3000")))


async def main():
    print("=== BEFORE ===\n" + listeners())

    with quiesce.quiesced():
        idle = quiesce.assert_idle()
        print(f"=== QUIESCED (idle {idle}) — ports free ===\n" + (listeners() or "(none)"))

        eng = MLX(MODEL, SCRATCH_PORT)
        print(f"starting {eng.name} :{SCRATCH_PORT} {MODEL} …")
        eng.start(log_path="/tmp/llmbench-ctl.log", ready_timeout=300)
        print("ready. firing one request…")
        lvl = await run_level(eng.base_url, eng.served_id(),
                              [{"role": "user", "content": "Say hi in one word."}],
                              concurrency=1, n=1, max_tokens=16)
        print(f"  ok={lvl.ok} fail={lvl.fail} ttft={lvl.ttft_p50:.2f}s "
              f"decode={lvl.decode_tps_median:.1f} t/s errors={lvl.errors}")
        eng.stop()
        print("engine stopped.")

    # restored on context exit
    print("=== AFTER (restored) ===\n" + listeners())
    for name, port in (("mlx", config.MLX_PORT), ("embed", config.EMBED_PORT)):
        up = _up(port)
        print(f"  {name} :{port} -> {'UP' if up else 'DOWN'}")


def _up(port: int) -> bool:
    try:
        return httpx.get(f"http://127.0.0.1:{port}/v1/models", timeout=5).status_code == 200
    except Exception:
        try:
            return httpx.get(f"http://127.0.0.1:{port}/api/tags", timeout=5).status_code == 200
        except Exception:
            return False


if __name__ == "__main__":
    asyncio.run(main())
