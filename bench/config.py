"""Static config: endpoints, paths, the box under test."""
from __future__ import annotations
import os

# Run ON tardis (localhost) so network latency doesn't pollute TTFT/throughput.
MLX_PORT = 11434
EMBED_PORT = 11435
LLAMACPP_PORT = 11436   # harness-launched engines get their own ports
LMSTUDIO_PORT = 11437
OLLAMA_CHAT_PORT = 11438

HOST = "127.0.0.1"
HF_CACHE = os.path.expanduser("~/.cache/huggingface/hub")

# launchd units the quiesce step takes down so the box is single-tenant during runs.
LAUNCHD_UNITS = ["io.cagdas.mlx", "io.cagdas.ollama-embed", "io.cagdas.ollama-embed-prewarm"]
# OrbStack compose group for open-webui (stopped during runs).
OPENWEBUI_DIR = os.path.expanduser("~/srv/open-webui")

# The box (for the report header).
BOX = {
    "host": "tardis",
    "chip": "Apple M4 Pro (14c CPU / 20c GPU)",
    "unified_mem_gb": 64,
    "wired_limit_mb": 57344,
    "os": "macOS 26.4 (Darwin 25.x)",
}

def base_url(port: int) -> str:
    return f"http://{HOST}:{port}/v1"
