"""aider — non-interactive single-message run. Points at the proxy as an
OpenAI-compatible backend. --yes auto-confirms, --no-auto-commits keeps git out
of it, one --message then exit. Edit format is aider's default (diff) for the model.
"""
from __future__ import annotations
from .base import Harness


class Aider(Harness):
    name = "aider"
    bin = "aider"

    def command(self, workdir, instruction, solution_files, base_url, model):
        argv = [
            "aider",
            "--model", f"openai/{model}",
            "--edit-format", "whole",   # most robust for weak local models (diff often fails to apply)
            "--no-auto-commits", "--no-gitignore", "--yes",
            "--no-show-model-warnings", "--no-check-update", "--no-stream",
            "--message", instruction,   # identical shared prompt (no per-harness scaffolding)
            *solution_files,
        ]
        env = {
            "OPENAI_API_BASE": base_url,
            "OPENAI_API_KEY": "bench",
            "AIDER_ANALYTICS": "false",
        }
        return argv, env
