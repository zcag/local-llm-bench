#!/bin/bash
# Pull the L1 MLX model zoo into the HF cache on tardis. Run when the box is idle
# (NOT during a benchmark). hf download is resumable, so re-running is safe.
set -uo pipefail
HF=~/srv/mlx/.venv/bin/hf
REPOS=(
  "mlx-community/Qwen3-Coder-30B-A3B-Instruct-4bit"
  "mlx-community/Qwen3-Coder-30B-A3B-Instruct-6bit"
  "mlx-community/Qwen3-Coder-30B-A3B-Instruct-8bit"
  "mlx-community/gpt-oss-20b-MXFP4-Q8"
  "mlx-community/Devstral-Small-2507-8bit"
  "mlx-community/Qwen2.5-Coder-32B-Instruct-8bit"
)
# (coder-next-mxfp4 and 30b-a3b-4bit-DWQ already cached)
for r in "${REPOS[@]}"; do
  echo "=== $r ==="
  "$HF" download "$r" >/dev/null 2>&1 && echo "  ok" || echo "  FAILED: $r"
done
echo "=== cache size ==="; du -sh ~/.cache/huggingface/hub
