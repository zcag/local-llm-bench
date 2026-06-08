"""opencode (sst) — non-interactive `opencode run`. A per-workdir opencode.json
defines a custom OpenAI-compatible provider pointing at the proxy. The proxy
rewrites the model id, so the alias here can be simple.
"""
from __future__ import annotations
import json
import os
from .base import Harness

ALIAS = "local"


class OpenCode(Harness):
    name = "opencode"
    bin = "opencode"

    def command(self, workdir, instruction, solution_files, base_url, model):
        cfg = {
            "$schema": "https://opencode.ai/config.json",
            "provider": {
                "bench": {
                    "npm": "@ai-sdk/openai-compatible",
                    "name": "bench",
                    "options": {"baseURL": base_url, "apiKey": "bench"},
                    "models": {ALIAS: {"name": ALIAS}},
                }
            },
        }
        with open(os.path.join(workdir, "opencode.json"), "w") as f:
            json.dump(cfg, f)
        argv = ["opencode", "run", "--model", f"bench/{ALIAS}", instruction]
        return argv, {"OPENCODE_DISABLE_AUTOUPDATE": "1"}
