"""L2 — agent-harness shootout. Fixed model (L1 winner) served by MLX behind the
measurement proxy; every available harness runs the SAME polyglot tasks. We
record task success (pytest), tokens (proxy), and wall-clock per (harness, task).

Run ON tardis:  PYTHONUNBUFFERED=1 .venv/bin/python -m bench.run_l2 \
                   --model mlx-community/Qwen3-Coder-Next-mxfp4 --tasks 15
"""
from __future__ import annotations
import argparse
import os
import subprocess
import tempfile
import time
import httpx

from . import config, quiesce, l2_tasks
from .results import write
from .engines.mlx import MLX
from .harnesses.aider import Aider
from .harnesses.opencode import OpenCode

PROXY_PORT = 11500
PROXY_URL = f"http://127.0.0.1:{PROXY_PORT}"
HARNESS_BASE = f"{PROXY_URL}/v1"
ALL_HARNESSES = [Aider(), OpenCode()]   # goose/crush/claude-code added after live validation
LOGDIR = "/tmp/llmbench"


def start_proxy(model: str) -> subprocess.Popen:
    os.makedirs(LOGDIR, exist_ok=True)
    f = open(f"{LOGDIR}/proxy.log", "ab")
    p = subprocess.Popen(
        [".venv/bin/python", "-m", "bench.proxy", "--upstream", str(config.MLX_PORT),
         "--port", str(PROXY_PORT), "--model", model],
        stdout=f, stderr=f)
    for _ in range(30):
        try:
            if httpx.get(f"{PROXY_URL}/__bench/stats", timeout=3).status_code == 200:
                return p
        except Exception:
            time.sleep(0.5)
    raise RuntimeError("proxy did not start")


def proxy_reset():
    httpx.post(f"{PROXY_URL}/__bench/reset", timeout=10)


def proxy_stats() -> dict:
    return httpx.get(f"{PROXY_URL}/__bench/stats", timeout=10).json()


def run_one(harness, task_id, model):
    workdir = tempfile.mkdtemp(prefix=f"l2-{harness.name}-{task_id}-")
    meta = l2_tasks.prepare(task_id, workdir)
    proxy_reset()
    res = harness.run(workdir, meta["instruction"], meta["solution_files"], HARNESS_BASE, model)
    stats = proxy_stats()
    graded = l2_tasks.grade(workdir, meta["test_files"])
    row = {
        "layer": "L2", "harness": harness.name, "task": task_id, "model": model,
        "passed": graded["passed"], "harness_ok": res.get("ok"),
        "secs": res.get("secs"), "rc": res.get("rc"),
        "prompt_tokens": stats.get("prompt_tokens"), "completion_tokens": stats.get("completion_tokens"),
        "requests": stats.get("requests"),
        "fail_tail": "" if graded["passed"] else graded["tail"][-200:],
        "harness_err": res.get("error", ""),
    }
    write(row)
    flag = "PASS" if graded["passed"] else "fail"
    print(f"  [{harness.name:9}] {task_id:22} {flag}  {res.get('secs')}s "
          f"tok={stats.get('prompt_tokens')}+{stats.get('completion_tokens')} "
          f"reqs={stats.get('requests')} {res.get('error','')}")
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-Coder-Next-mxfp4")
    ap.add_argument("--tasks", type=int, default=15)
    ap.add_argument("--only", default="", help="comma harness names")
    args = ap.parse_args()

    harnesses = [h for h in ALL_HARNESSES if not args.only or h.name in args.only.split(",")]
    avail = [h for h in harnesses if h.available()]
    print("harnesses available:", [h.name for h in avail],
          "| missing:", [h.name for h in harnesses if not h.available()])
    tasks = l2_tasks.task_ids(args.tasks)
    print(f"tasks ({len(tasks)}):", tasks)

    with quiesce.quiesced():
        quiesce.assert_idle()
        eng = MLX(args.model, config.MLX_PORT)
        print(f"starting MLX {args.model}…")
        eng.start(log_path=f"{LOGDIR}/l2-mlx.log", ready_timeout=900)
        proxy = start_proxy(args.model)
        try:
            for h in avail:
                print(f"\n=== {h.name} ===")
                for t in tasks:
                    try:
                        run_one(h, t, args.model)
                    except Exception as e:  # noqa: BLE001
                        print(f"  !! {h.name}/{t}: {type(e).__name__}: {e}")
                        write({"layer": "L2", "harness": h.name, "task": t, "scenario": "ERROR",
                               "error": str(e)})
        finally:
            proxy.terminate()
            eng.stop()
    print("\nL2 done -> results/runs.jsonl")


if __name__ == "__main__":
    main()
