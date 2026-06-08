"""llama-server adapter (GGUF + Metal). Model is a path to a .gguf in HF cache.

`-ngl 999` offloads all layers to Metal; `--flash-attn` on; `-c` context.
Served model id under /v1 defaults to the file basename — we pin it with --alias.
"""
from __future__ import annotations
from .base import Engine


class LlamaCpp(Engine):
    name = "llama.cpp"

    def __init__(self, model: str, port: int, ctx: int = 16384, alias: str | None = None):
        super().__init__(model, port)   # model = path to .gguf
        self.ctx = ctx
        self.alias = alias or "llamacpp-model"

    @property
    def proc_match(self) -> str:
        return "llama-server"

    def served_id(self) -> str:
        return self.alias

    def _command(self) -> list[str]:
        return [
            "llama-server",
            "-m", self.model,
            "--alias", self.alias,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "-ngl", "999",
            "--flash-attn", "on",
            "-c", str(self.ctx),
            "--jinja",            # use the model's chat template (tool-calling)
        ]
