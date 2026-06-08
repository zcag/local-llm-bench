"""Agent-harness adapter contract. Each harness drives the SAME model via the
measurement proxy on the SAME tasks; we measure task success (pytest), tokens
(from the proxy), and wall-clock. Each harness is configured to its recommended
non-interactive mode and documented.
"""
from __future__ import annotations
import abc
import shutil
import subprocess
import time


class Harness(abc.ABC):
    name = "harness"
    bin = ""              # executable to check for

    def available(self) -> bool:
        return bool(shutil.which(self.bin)) if self.bin else False

    @abc.abstractmethod
    def command(self, workdir: str, instruction: str, solution_files: list[str],
                base_url: str, model: str) -> tuple[list[str], dict]:
        """Return (argv, env_overrides) to run the harness once, non-interactively,
        editing the solution file(s) in workdir per the instruction."""

    def run(self, workdir, instruction, solution_files, base_url, model, timeout=600) -> dict:
        import os
        argv, env_over = self.command(workdir, instruction, solution_files, base_url, model)
        env = os.environ.copy()
        env.update(env_over)
        t0 = time.perf_counter()
        try:
            proc = subprocess.run(argv, cwd=workdir, capture_output=True, text=True,
                                  timeout=timeout, env=env)
            return {"ok": proc.returncode == 0, "secs": round(time.perf_counter() - t0, 1),
                    "rc": proc.returncode, "stdout": proc.stdout[-1500:], "stderr": proc.stderr[-800:]}
        except subprocess.TimeoutExpired:
            return {"ok": False, "secs": round(time.perf_counter() - t0, 1),
                    "rc": -1, "error": "timeout", "stdout": "", "stderr": ""}
        except Exception as e:  # noqa: BLE001
            return {"ok": False, "secs": round(time.perf_counter() - t0, 1),
                    "rc": -1, "error": f"{type(e).__name__}: {e}"}
