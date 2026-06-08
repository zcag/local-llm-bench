"""llama-server adapter (GGUF + Metal). Model is a path to a .gguf in HF cache.

`-ngl 999` offloads all layers to Metal; `--flash-attn` on; `-c` context.
Served model id under /v1 defaults to the file basename — we pin it with --alias.
"""
from __future__ import annotations
from .base import Engine


class LlamaCpp(Engine):
    name = "llama.cpp"

    def __init__(self, model: str, port: int, ctx: int = 16384, alias: str | None = None,
                 parallel: int = 16):
        super().__init__(model, port)   # model = path to .gguf
        self.ctx = ctx
        self.alias = alias or "llamacpp-model"
        self.parallel = parallel  # slots for the concurrency sweep (default auto was 4 — unfair at c8/16)

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
            "-c", str(self.ctx),                  # shared budget (unified) — fits 32k single-stream
            "--parallel", str(self.parallel),     # enough slots to actually batch at c=16
            "--kv-unified",                       # one shared KV cache: full ctx for 1 stream, shared for N
            "-b", "2048", "-ub", "2048",          # ubatch 2048 for Metal prefill throughput (F8)
            "--jinja",            # use the model's chat template (tool-calling)
        ]
