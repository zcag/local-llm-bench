"""mlx_lm.server adapter — uses the pinned venv on tardis."""
from __future__ import annotations
import os
from .base import Engine

MLX_PY = os.path.expanduser("~/srv/mlx/.venv/bin/python")


class MLX(Engine):
    name = "mlx_lm.server"

    def __init__(self, model: str, port: int, decode_concurrency: int = 1,
                 prompt_concurrency: int = 1, draft_model: str | None = None):
        super().__init__(model, port)
        # mlx_lm.server CAN continuous-batch via these — default 1 serializes
        # (which produced the earlier false "MLX can't batch" finding).
        self.decode_concurrency = decode_concurrency
        self.prompt_concurrency = prompt_concurrency
        self.draft_model = draft_model  # speculative decoding (works on DWQ-30B, not Coder-Next)

    @property
    def proc_match(self) -> str:
        return "mlx_lm.server"

    def _command(self) -> list[str]:
        cmd = [
            MLX_PY, "-m", "mlx_lm.server",
            "--model", self.model,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--log-level", "INFO",
            "--decode-concurrency", str(self.decode_concurrency),
            "--prompt-concurrency", str(self.prompt_concurrency),
        ]
        if self.draft_model:
            cmd += ["--draft-model", self.draft_model]
        return cmd
