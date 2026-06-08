"""Validate the eval pipeline against the live MLX server (small limits).
Run ON tardis:  .venv/bin/python scripts/test_evals.py
"""
from __future__ import annotations
import asyncio
import sys
import httpx

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from bench import config
from bench.evals import toolcalling, longcontext, humaneval

BASE = config.base_url(config.MLX_PORT)
# Pin the exact served id — mlx_lm.server swaps models if you request a different
# cached id, which would disrupt the live server. Coder-Next is the live default.
MODEL = "mlx-community/Qwen3-Coder-Next-mxfp4"


async def main():
    model = MODEL
    print("model:", model)

    if "--he-only" not in sys.argv:
        tc = await toolcalling.run(BASE, model)
        print("\ntoolcalling:", {k: tc[k] for k in ("passed", "total", "score")})
        lc = await longcontext.run(BASE, model)
        print("\nlongcontext:", {k: lc[k] for k in ("passed", "total", "score")}, "by_len:", lc["by_length"])

    he = await humaneval.run(BASE, model)   # full 164 (evalplus requires the complete set)
    print("\nhumaneval+ (full 164):",
          {k: he.get(k) for k in ("pass@1_base", "pass@1_plus", "n", "error")})
    if he.get("raw"):
        print("raw tail:", he["raw"])


if __name__ == "__main__":
    asyncio.run(main())
