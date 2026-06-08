"""L2 task set — Aider polyglot Python exercises (recognized, test-backed). Each
task: a stub file the harness must implement + a hidden test file used to grade.
Same tasks for every harness → apples-to-apples agentic comparison.
"""
from __future__ import annotations
import json
import os
import shutil
import subprocess
import sys

POLY = os.path.join(os.path.dirname(__file__), "..", "tasks", "polyglot", "python", "exercises", "practice")

# A fixed, deterministic subset (sorted) so every harness sees identical tasks.
def task_ids(limit: int | None = None) -> list[str]:
    ids = sorted(d for d in os.listdir(POLY) if os.path.isdir(os.path.join(POLY, d)))
    return ids[:limit] if limit else ids


def _config(ex_dir: str) -> dict:
    with open(os.path.join(ex_dir, ".meta", "config.json")) as f:
        return json.load(f)


def prepare(task_id: str, workdir: str) -> dict:
    """Copy ONLY the stub solution file(s) + instructions into workdir — the test
    file is deliberately withheld so no harness can read or run the hidden tests
    it's graded on (keeps agentic harnesses honest vs one-shot ones). Returns
    {instruction, solution_files, test_files} (test_files copied in at grade time)."""
    ex = os.path.join(POLY, task_id)
    cfg = _config(ex)
    sol = cfg["files"]["solution"]
    tst = cfg["files"]["test"]
    os.makedirs(workdir, exist_ok=True)
    for rel in sol:
        src = os.path.join(ex, rel)
        dst = os.path.join(workdir, rel)
        os.makedirs(os.path.dirname(dst) or workdir, exist_ok=True)
        shutil.copy(src, dst)
    with open(os.path.join(ex, ".docs", "instructions.md")) as f:
        instruction = f.read()
    return {"instruction": instruction, "solution_files": sol, "test_files": tst, "task_id": task_id}


def grade(workdir: str, test_files: list[str], task_id: str, timeout: int = 120) -> dict:
    """Copy the withheld test file(s) in, then run pytest. Pass = exit 0."""
    ex = os.path.join(POLY, task_id)
    for rel in test_files:
        dst = os.path.join(workdir, rel)
        os.makedirs(os.path.dirname(dst) or workdir, exist_ok=True)
        shutil.copy(os.path.join(ex, rel), dst)
    try:
        proc = subprocess.run(
            [sys.executable, "-m", "pytest", "-q", "--no-header", *test_files],
            cwd=workdir, capture_output=True, text=True, timeout=timeout,
        )
        out = (proc.stdout + proc.stderr)[-600:]
        return {"passed": proc.returncode == 0, "tail": out}
    except subprocess.TimeoutExpired:
        return {"passed": False, "tail": "pytest timeout"}
