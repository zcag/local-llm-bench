"""Engine adapter contract: the harness owns the engine lifecycle during a run.

Each adapter knows how to start ONE model on a known port, health-check it,
expose its OpenAI-compatible /v1 base_url, and stop cleanly. The harness boots
the engine, runs the load, tears it down — so memory/thermals start from a known
state every time and nothing else is contending for the box.
"""
from __future__ import annotations
import abc
import subprocess
import time
import httpx


class Engine(abc.ABC):
    name: str = "engine"

    def __init__(self, model: str, port: int):
        self.model = model
        self.port = port
        self.proc: subprocess.Popen | None = None

    @property
    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}/v1"

    @property
    @abc.abstractmethod
    def proc_match(self) -> str:
        """Substring matching the engine process(es) for memory sampling."""

    @abc.abstractmethod
    def _command(self) -> list[str]:
        ...

    def config_str(self) -> str:
        """Exact config used — recorded into every result row for provenance (F18)."""
        try:
            return " ".join(self._command())
        except NotImplementedError:
            return f"{self.name} model={self.model} port={self.port}"

    def _env(self) -> dict | None:
        return None

    def start(self, log_path: str, ready_timeout: float = 300) -> None:
        f = open(log_path, "ab")
        self.proc = subprocess.Popen(self._command(), stdout=f, stderr=f, env=self._env())
        self.wait_ready(ready_timeout)

    def served_id(self) -> str:
        """Model id to send in requests (some engines rename)."""
        return self.model

    def wait_ready(self, timeout: float) -> None:
        deadline = time.time() + timeout
        last = ""
        while time.time() < deadline:
            if self.proc and self.proc.poll() is not None:
                raise RuntimeError(f"{self.name} exited early (code {self.proc.returncode}); see log")
            try:
                r = httpx.get(f"{self.base_url}/models", timeout=5)
                if r.status_code == 200:
                    return
            except Exception as e:  # noqa: BLE001
                last = str(e)
            time.sleep(2)
        raise TimeoutError(f"{self.name} not ready in {timeout}s ({last})")

    def stop(self) -> None:
        if not self.proc:
            return
        self.proc.terminate()
        try:
            self.proc.wait(timeout=30)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait(timeout=10)
        self.proc = None

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.stop()
