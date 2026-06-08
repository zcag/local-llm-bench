"""Ollama chat adapter — dedicated `ollama serve` on its own port for the run.

Separate from the embed-only instance on :11435. keep_alive=-1 so the model
stays resident (no eviction mid-benchmark). Model must already be pulled.
"""
from __future__ import annotations
import os
from .base import Engine


class Ollama(Engine):
    name = "ollama"

    @property
    def proc_match(self) -> str:
        return "ollama"

    def _env(self) -> dict:
        env = os.environ.copy()
        env["OLLAMA_HOST"] = f"127.0.0.1:{self.port}"
        env["OLLAMA_KEEP_ALIVE"] = "-1"
        env["OLLAMA_FLASH_ATTENTION"] = "1"
        return env

    def _command(self) -> list[str]:
        return ["ollama", "serve"]
