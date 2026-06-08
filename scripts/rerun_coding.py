"""F2 fix: re-run HumanEval+ AND MBPP+ for ALL zoo models at concurrency=1, with
samples saved — so coding scores are independent (no queue-timeout artifact) and
auditable (prove generations differ across models). Quiesced, one model at a time.
Rewrites each model's coding rows in runs.jsonl + detail files.

Run ON tardis: PYTHONUNBUFFERED=1 .venv/bin/python scripts/rerun_coding.py [--only tag,tag]
"""
from __future__ import annotations
import argparse, asyncio, json, os, sys
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from bench import config, quiesce
from bench.engines.mlx import MLX
from bench.evals import humaneval
from bench.results import write, RESULTS_DIR
from bench.zoo import ZOO

DETAIL = os.path.join(RESULTS_DIR, "l1_evals")


async def one_model(tag, repo):
    eng = MLX(repo, config.MLX_PORT)
    print(f"\n=== {tag} ({repo}) ===")
    eng.start(log_path=f"/tmp/llmbench/coding-{tag}.log", ready_timeout=900)
    try:
        for ds in ("humaneval", "mbpp"):
            res = await humaneval.run(eng.base_url, repo, concurrency=1,
                                      save_dir=os.path.join(DETAIL, f"{tag}__{ds}_samples"), dataset=ds)
            with open(os.path.join(DETAIL, f"{tag}__{ds}plus.json"), "w") as f:
                json.dump({"model_tag": tag, "model": repo, **res}, f, indent=2)
            write({"layer": "L1", "model_tag": tag, "model": repo, "rerun": True,
                   **{k: v for k, v in res.items() if k not in ("cases", "raw")}})
            print(f"  {ds}+ -> base={res.get('pass@1_base')} plus={res.get('pass@1_plus')} n={res.get('n')}")
    finally:
        eng.stop()
        await asyncio.sleep(5)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    args = ap.parse_args()
    zoo = ZOO if not args.only else [z for z in ZOO if z[0] in set(args.only.split(","))]
    print("models:", [z[0] for z in zoo])
    with quiesce.quiesced():
        quiesce.assert_idle()
        for tag, repo, sz, note in zoo:
            try:
                await one_model(tag, repo)
            except Exception as e:  # noqa: BLE001
                print(f"  !! {tag}: {type(e).__name__}: {e}")
                write({"layer": "L1", "model_tag": tag, "scenario": "ERROR", "error": str(e), "rerun": True})
    print("\ncoding re-run done")


if __name__ == "__main__":
    asyncio.run(main())
