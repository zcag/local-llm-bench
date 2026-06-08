"""Block goose — non-interactive `goose run -t`. Configured for an OpenAI-compatible
backend (the proxy) via env. goose's developer extension edits files in cwd.
"""
from __future__ import annotations
from .base import Harness


class Goose(Harness):
    name = "goose"
    bin = "goose"

    def command(self, workdir, instruction, solution_files, base_url, model):
        # base_url ends with /v1; goose's openai provider appends /v1 itself, so
        # hand it the host root.
        host = base_url[:-3] if base_url.endswith("/v1") else base_url
        argv = ["goose", "run", "--no-session", "-t",
                instruction + f"\n\nEdit the file(s) {', '.join(solution_files)} in the current "
                "directory to implement this. Do not create new files."]
        env = {
            "GOOSE_PROVIDER": "openai",
            "GOOSE_MODEL": "local",
            "OPENAI_API_KEY": "bench",
            "OPENAI_HOST": host,
            "OPENAI_BASE_URL": base_url,
            "GOOSE_DISABLE_KEYRING": "1",
        }
        return argv, env
