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

import json
from . import config, quiesce, l2_tasks
from .results import write
from .engines.mlx import MLX
from .harnesses.aider import Aider
from .harnesses.opencode import OpenCode
from .harnesses.goose import Goose
from .harnesses.crush import Crush
from .harnesses.claudecode import ClaudeCode

PROXY_PORT = 11500
PROXY_URL = f"http://127.0.0.1:{PROXY_PORT}"
HARNESS_BASE = f"{PROXY_URL}/v1"
CCR_PORT = 3456
CCR_URL = f"http://127.0.0.1:{CCR_PORT}"
# opencode excluded: won't run headless against a hermetic custom-provider config
# (initializes then hangs, 0 model requests, no error). The daily `lcld` use relies
# on opencode's own persisted auth/config, which a clean benchmark deliberately avoids.
ALL_HARNESSES = [Aider(), Goose(), Crush(), ClaudeCode()]
LOGDIR = "/tmp/llmbench"


def harness_base(h) -> str:
    # claude-code talks Anthropic -> ccr shim -> proxy -> MLX (proxy still tallies)
    return CCR_URL if h.name == "claude-code" else HARNESS_BASE


def start_ccr() -> subprocess.Popen | None:
    """claude-code-router: Anthropic API -> our OpenAI proxy."""
    cfgdir = os.path.expanduser("~/.claude-code-router")
    os.makedirs(cfgdir, exist_ok=True)
    cfg = {
        "Providers": [{
            "name": "local",
            "api_base_url": f"{PROXY_URL}/v1/chat/completions",
            "api_key": "bench",
            "models": ["local"],
        }],
        "Router": {"default": "local,local"},
        "HOST": "127.0.0.1", "PORT": CCR_PORT,
    }
    with open(os.path.join(cfgdir, "config.json"), "w") as f:
        json.dump(cfg, f)
    subprocess.run(["ccr", "restart"], capture_output=True, text=True)
    for _ in range(20):
        try:
            httpx.get(f"{CCR_URL}/", timeout=3)
            return True
        except Exception:
            time.sleep(0.5)
    return True  # ccr may not answer GET /; rely on per-task run


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
    res = harness.run(workdir, meta["instruction"], meta["solution_files"], harness_base(harness),
                      model, timeout=900)   # agentic harnesses (claude-code/crush) need headroom
    stats = proxy_stats()
    graded = l2_tasks.grade(workdir, meta["test_files"], task_id)
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
    return {**row, "_stdout": (res.get("stdout", "") + " || ERR: " + res.get("stderr", ""))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit-DWQ")
    ap.add_argument("--tasks", type=int, default=15)
    ap.add_argument("--only", default="", help="comma harness names")
    ap.add_argument("--validate", action="store_true", help="1 task per harness, verbose")
    args = ap.parse_args()

    harnesses = [h for h in ALL_HARNESSES if not args.only or h.name in args.only.split(",")]
    avail = [h for h in harnesses if h.available()]
    print("harnesses available:", [h.name for h in avail],
          "| missing:", [h.name for h in harnesses if not h.available()])
    tasks = l2_tasks.task_ids(1 if args.validate else args.tasks)
    print(f"tasks ({len(tasks)}):", tasks)

    with quiesce.quiesced():
        quiesce.assert_idle()
        eng = MLX(args.model, config.MLX_PORT)
        print(f"starting MLX {args.model}…")
        eng.start(log_path=f"{LOGDIR}/l2-mlx.log", ready_timeout=900)
        proxy = start_proxy(args.model)
        start_ccr()
        try:
            for h in avail:
                print(f"\n=== {h.name} ({harness_base(h)}) ===")
                for t in tasks:
                    try:
                        row = run_one(h, t, args.model)
                        if args.validate:
                            print(f"     harness_ok={row['harness_ok']} rc={row['rc']} "
                                  f"err={row['harness_err']}")
                            print("     stdout tail:", repr(row.get("_stdout", ""))[:400])
                    except Exception as e:  # noqa: BLE001
                        print(f"  !! {h.name}/{t}: {type(e).__name__}: {e}")
                        write({"layer": "L2", "harness": h.name, "task": t, "scenario": "ERROR",
                               "error": str(e)})
        finally:
            proxy.terminate()
            subprocess.run(["ccr", "stop"], capture_output=True)
            eng.stop()
    print("\nL2 done -> results/runs.jsonl")


if __name__ == "__main__":
    main()
