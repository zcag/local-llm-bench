"""mlx_lm.server adapter — uses the pinned venv on tardis."""
from __future__ import annotations
import os
from .base import Engine

MLX_PY = os.path.expanduser("~/srv/mlx/.venv/bin/python")


class MLX(Engine):
    name = "mlx_lm.server"

    @property
    def proc_match(self) -> str:
        return "mlx_lm.server"

    def _command(self) -> list[str]:
        return [
            MLX_PY, "-m", "mlx_lm.server",
            "--model", self.model,
            "--host", "127.0.0.1",
            "--port", str(self.port),
            "--log-level", "INFO",
        ]
