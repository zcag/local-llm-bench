"""LM Studio adapter — driven by the `lms` CLI (its server is a singleton daemon,
not a Popen child). LM Studio's engines are llama.cpp (GGUF) + MLX under the hood,
so this measures the LM Studio serving wrapper's overhead vs the raw engines.

Model key = "<publisher>/<repo>" as LM Studio sees it under ~/.lmstudio/models.
To benchmark the *identical* GGUF file, symlink our download into that tree first
(see scripts: link_lmstudio_model).
"""
from __future__ import annotations
import os
import subprocess
import time
import httpx
from .base import Engine

LMS = os.path.expanduser("~/.lmstudio/bin/lms")


class LMStudio(Engine):
    name = "lmstudio"

    def __init__(self, model: str, port: int, identifier: str | None = None, ctx: int = 16384):
        super().__init__(model, port)   # model = LM Studio model key
        self.identifier = identifier or "bench-model"
        self.ctx = ctx

    @property
    def proc_match(self) -> str:
        return "LM Studio"

    def served_id(self) -> str:
        return self.identifier

    def _command(self):  # not used — lifecycle overridden
        raise NotImplementedError

    def start(self, log_path: str, ready_timeout: float = 300) -> None:
        subprocess.run([LMS, "server", "start", "--port", str(self.port)],
                       capture_output=True, text=True, check=True)
        # load the model fully onto GPU, fixed context
        subprocess.run(
            [LMS, "load", self.model, "--identifier", self.identifier,
             "--gpu", "max", "--context-length", str(self.ctx), "-y"],
            capture_output=True, text=True, check=True)
        self.wait_ready(ready_timeout)

    def stop(self) -> None:
        subprocess.run([LMS, "unload", "--all"], capture_output=True, text=True)
        subprocess.run([LMS, "server", "stop"], capture_output=True, text=True)
