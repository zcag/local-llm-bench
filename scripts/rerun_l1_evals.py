"""L1 non-coding eval pass for ALL 8 models: bfcl-style tool-calling + ifeval +
long-context (with the F14 token-calc fix). Each model served by the engine that
PARSES its tool format — MLX for the Qwen3-Coder family, llama.cpp --jinja for
gpt-oss / devstral / qwen2.5 (whose harmony / [TOOL_CALLS] formats mlx_lm's OpenAI
endpoint doesn't surface as tool_calls, F5). One model up at a time, quiesced.

Run ON tardis: PYTHONUNBUFFERED=1 .venv/bin/python scripts/rerun_l1_evals.py
"""
from __future__ import annotations
import asyncio, json, os, sys
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from bench import config, quiesce
from bench.engines.mlx import MLX
from bench.engines.llamacpp import LlamaCpp
from bench.evals import bfcl, ifeval, longcontext
from bench.results import write, RESULTS_DIR

G = os.path.expanduser("~/models/gguf")
# (tag, engine, model-id-or-gguf-path)
MODELS = [
    ("coder-next-mxfp4",       "mlx", "mlx-community/Qwen3-Coder-Next-mxfp4"),
    ("30b-a3b-4bit",           "mlx", "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"),
    ("30b-a3b-4bit-dwq",       "mlx", "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ"),
    ("30b-a3b-6bit",           "mlx", "mlx-community/Qwen3-Coder-30B-A3B-Instruct-6bit"),
    ("30b-a3b-8bit",           "mlx", "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"),
    ("gpt-oss-20b",            "llamacpp", f"{G}/gpt-oss-20b/gpt-oss-20b-mxfp4.gguf"),
    ("devstral-2507-8bit",     "llamacpp", f"{G}/devstral-2507/Devstral-Small-2507-Q8_0.gguf"),
    ("qwen2.5-coder-32b-8bit", "llamacpp", f"{G}/qwen2.5-coder-32b/Qwen2.5-Coder-32B-Instruct-Q8_0.gguf"),
]
DETAIL = os.path.join(RESULTS_DIR, "l1_evals")


def make_engine(engine, model_or_path, tag):
    if engine == "mlx":
        return MLX(model_or_path, config.MLX_PORT), model_or_path
    if not os.path.exists(model_or_path):
        return None, None
    return LlamaCpp(model_or_path, config.LLAMACPP_PORT, ctx=34816, alias=tag, parallel=1), tag


async def one(tag, engine_name, model_or_path):
    eng, served = make_engine(engine_name, model_or_path, tag)
    if eng is None:
        print(f"  !! {tag}: artifact missing {model_or_path}"); return
    print(f"\n=== {tag} via {engine_name} ===")
    eng.start(log_path=f"/tmp/llmbench/l1eval-{tag}.log", ready_timeout=900)
    try:
        for name, coro in [("bfcl_style", bfcl.run(eng.base_url, served)),
                           ("ifeval_style", ifeval.run(eng.base_url, served)),
                           ("longcontext", longcontext.run(eng.base_url, served))]:
            res = await coro
            with open(os.path.join(DETAIL, f"{tag}__{name}.json"), "w") as f:
                json.dump({"model_tag": tag, "engine": engine_name, **res}, f, indent=2)
            write({"layer": "L1", "model_tag": tag, "engine": engine_name, "served_via": engine_name,
                   **{k: v for k, v in res.items() if k != "cases"}})
            print(f"  {name}: score={res.get('score')} ({res.get('passed')}/{res.get('total')}) "
                  f"{res.get('by_category') or res.get('by_length') or ''}")
    finally:
        eng.stop(); await asyncio.sleep(5)


async def main():
    with quiesce.quiesced():
        quiesce.assert_idle()
        for tag, engine_name, mp in MODELS:
            try:
                await one(tag, engine_name, mp)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {tag}: {type(e).__name__}: {e}")
                write({"layer": "L1", "model_tag": tag, "engine": engine_name, "scenario": "ERROR", "error": str(e)})
    print("\nL1 non-coding eval pass done")


if __name__ == "__main__":
    asyncio.run(main())
