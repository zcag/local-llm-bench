"""F5 fix: gpt-oss / devstral / qwen2.5-coder produce tool-calls in formats
mlx_lm.server's OpenAI endpoint doesn't parse (harmony / Mistral [TOOL_CALLS]).
llama.cpp with --jinja parses them via the model's chat template. Re-serve each
via llama.cpp and re-score toolcalling + longcontext so those cells are real
(engine×model). Verifies whether the failures were serving-path or the model.

Run ON tardis: PYTHONUNBUFFERED=1 .venv/bin/python scripts/rerun_tools_llamacpp.py
"""
from __future__ import annotations
import asyncio, json, os, sys
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from bench import config, quiesce
from bench.engines.llamacpp import LlamaCpp
from bench.evals import toolcalling, longcontext
from bench.results import write, RESULTS_DIR

G = os.path.expanduser("~/models/gguf")
MODELS = [
    ("gpt-oss-20b",          f"{G}/gpt-oss-20b/gpt-oss-20b-mxfp4.gguf"),
    ("devstral-2507-8bit",   f"{G}/devstral-2507/Devstral-Small-2507-Q8_0.gguf"),
    ("qwen2.5-coder-32b-8bit", f"{G}/qwen2.5-coder-32b/Qwen2.5-Coder-32B-Instruct-Q8_0.gguf"),
]
DETAIL = os.path.join(RESULTS_DIR, "l1_evals")


async def one(tag, gguf):
    if not os.path.exists(gguf):
        print(f"  !! {tag}: gguf missing {gguf}"); return
    eng = LlamaCpp(gguf, config.LLAMACPP_PORT, ctx=34816, alias=tag, parallel=1)
    print(f"\n=== {tag} via llama.cpp --jinja ===")
    eng.start(log_path=f"/tmp/llmbench/f5-{tag}.log", ready_timeout=600)
    try:
        for name, res in [("toolcalling", await toolcalling.run(eng.base_url, eng.served_id())),
                          ("longcontext", await longcontext.run(eng.base_url, eng.served_id()))]:
            with open(os.path.join(DETAIL, f"{tag}__{name}_llamacpp.json"), "w") as f:
                json.dump({"model_tag": tag, "engine": "llama.cpp", **res}, f, indent=2)
            write({"layer": "L1", "model_tag": tag, "engine": "llama.cpp", "served_via": "llama.cpp-jinja",
                   **{k: v for k, v in res.items() if k != "cases"}})
            print(f"  {name}: score={res.get('score')} ({res.get('passed')}/{res.get('total')})")
    finally:
        eng.stop(); await asyncio.sleep(5)


async def main():
    with quiesce.quiesced():
        quiesce.assert_idle()
        for tag, gguf in MODELS:
            try:
                await one(tag, gguf)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {tag}: {type(e).__name__}: {e}")
                write({"layer": "L1", "model_tag": tag, "engine": "llama.cpp", "scenario": "ERROR", "error": str(e)})
    print("\nF5 tool/longctx re-score done")


if __name__ == "__main__":
    asyncio.run(main())
