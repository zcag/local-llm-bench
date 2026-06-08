"""Ollama chat adapter — dedicated `ollama serve` on its own port for the run.

Separate from the embed-only instance on :11435. keep_alive=-1 so the model
stays resident (no eviction mid-benchmark). Model must already be pulled.
"""
from __future__ import annotations
import os
import subprocess
import tempfile
from .base import Engine


def register_gguf(name: str, gguf_path: str, port: int) -> None:
    """Register a local .gguf as an ollama model `name` (idempotent). Requires a
    running `ollama serve` on `port`. Ollama copies the blob into its store."""
    env = os.environ.copy()
    env["OLLAMA_HOST"] = f"127.0.0.1:{port}"
    existing = subprocess.run(["ollama", "list"], env=env, capture_output=True, text=True).stdout
    if name in existing:
        return
    with tempfile.NamedTemporaryFile("w", suffix=".Modelfile", delete=False) as f:
        f.write(f"FROM {gguf_path}\n")
        mf = f.name
    subprocess.run(["ollama", "create", name, "-f", mf], env=env,
                   capture_output=True, text=True, check=True)
    os.unlink(mf)


class Ollama(Engine):
    name = "ollama"

    def __init__(self, model: str, port: int, parallel: int = 1, ctx: int = 34816):
        super().__init__(model, port)
        self.parallel = parallel   # NUM_PARALLEL: 1 for single-stream, 16 for concurrency phase
        self.ctx = ctx

    @property
    def proc_match(self) -> str:
        return "ollama"

    def _env(self) -> dict:
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"127.0.0.1:{self.port}"
        env["OLLAMA_KEEP_ALIVE"] = "-1"
        env["OLLAMA_FLASH_ATTENTION"] = "1"
        env["OLLAMA_NUM_PARALLEL"] = str(self.parallel)
        env["OLLAMA_CONTEXT_LENGTH"] = str(self.ctx)   # explicit (was fragile via parent env)
        return env

    def _command(self) -> list[str]:
        return ["ollama", "serve"]
