"""Claude Code pointed at the LOCAL model via the claude-code-router (ccr) shim,
which translates the Anthropic API that `claude` speaks into OpenAI calls to the
proxy. ccr runs as a server (started by the orchestrator); this adapter just runs
`claude -p` non-interactively with edits auto-approved, env pointed at ccr.

The ccr endpoint is passed via base_url here (already the ccr port, not the proxy).
"""
from __future__ import annotations
from .base import Harness


class ClaudeCode(Harness):
    name = "claude-code"
    bin = "claude"

    def command(self, workdir, instruction, solution_files, base_url, model):
        argv = [
            "claude", "-p", instruction,   # identical shared prompt
            "--dangerously-skip-permissions",
        ]
        env = {
            "ANTHROPIC_BASE_URL": base_url,     # ccr server, e.g. http://127.0.0.1:3456
            "ANTHROPIC_API_KEY": "bench",
            "ANTHROPIC_MODEL": "local",
            "DISABLE_TELEMETRY": "1",
            "DISABLE_AUTOUPDATER": "1",
        }
        return argv, env
