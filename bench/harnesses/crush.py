"""Charmbracelet crush — non-interactive `crush run`. Custom OpenAI-compatible
provider via a per-workdir crush.json pointing at the proxy.
"""
from __future__ import annotations
import json
import os
from .base import Harness


class Crush(Harness):
    name = "crush"
    bin = "crush"

    def command(self, workdir, instruction, solution_files, base_url, model):
        cfg = {
            "$schema": "https://charm.land/crush.json",
            "providers": {
                "bench": {
                    "type": "openai",
                    "base_url": base_url,
                    "api_key": "bench",
                    "models": [{"id": "local", "name": "local",
                                "context_window": 32768, "default_max_tokens": 4096}],
                }
            },
        }
        with open(os.path.join(workdir, "crush.json"), "w") as f:
            json.dump(cfg, f)
        argv = ["crush", "run", "--quiet",
                instruction + f"\n\nImplement the solution in {', '.join(solution_files)} "
                "in the current directory. Use your tools to edit the file."]
        return argv, {"CRUSH_DISABLE_AUTOUPDATE": "1", "CRUSH_YOLO": "1"}
