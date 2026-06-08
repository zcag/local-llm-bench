"""Re-run HumanEval+ for one model at concurrency 1 (no MLX queue-timeout
artifact), quiesced + dedicated MLX. Rewrites the model's humaneval+ row in
results/runs.jsonl and its detail file. Run ON tardis:
  PYTHONUNBUFFERED=1 .venv/bin/python scripts/rerun_humaneval.py <tag> <hf_repo>
"""
from __future__ import annotations
import asyncio
import json
import os
import sys

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from bench import config, quiesce
from bench.engines.mlx import MLX
from bench.evals import humaneval
from bench.results import write, RESULTS_DIR

DETAIL = os.path.join(RESULTS_DIR, "l1_evals")


async def main():
    tag, repo = sys.argv[1], sys.argv[2]
    with quiesce.quiesced():
        quiesce.assert_idle()
        eng = MLX(repo, config.MLX_PORT)
        print(f"starting {repo}…")
        eng.start(log_path=f"/tmp/llmbench/rerun-{tag}.log", ready_timeout=900)
        res = await humaneval.run(eng.base_url, repo, concurrency=1,
                                  save_dir=os.path.join(DETAIL, f"{tag}__samples"))
        eng.stop()
    print("result:", {k: res.get(k) for k in ("pass@1_base", "pass@1_plus", "n")})
    with open(os.path.join(DETAIL, f"{tag}__humanevalplus.json"), "w") as f:
        json.dump({"model_tag": tag, "model": repo, **res}, f, indent=2)
    write({"layer": "L1", "model_tag": tag, "model": repo, "rerun": True,
           **{k: v for k, v in res.items() if k not in ("cases", "raw")}})


if __name__ == "__main__":
    asyncio.run(main())
