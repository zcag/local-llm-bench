"""Append-only JSONL results. One line per (engine, model, scenario, level)."""
from __future__ import annotations
import json
import os
import socket
import subprocess
import time
from typing import Any

RESULTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")


def _git_rev() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=os.path.dirname(RESULTS_DIR), stderr=subprocess.DEVNULL,
        ).decode().strip()
    except Exception:
        return "nogit"


def write(record: dict[str, Any], path: str | None = None) -> str:
    os.makedirs(RESULTS_DIR, exist_ok=True)
    path = path or os.path.join(RESULTS_DIR, "runs.jsonl")
    record = {
        "ts": time.time(),
        "host": socket.gethostname(),
        "rev": _git_rev(),
        **record,
    }
    with open(path, "a") as f:
        f.write(json.dumps(record) + "\n")
    return path
