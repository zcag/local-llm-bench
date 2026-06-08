# Reproduce

Built for an **Apple Silicon Mac** (tested on M4 Pro / 64 GB / macOS). Most of it is
engine-agnostic; the MLX bits are Apple-only.

## 1. Environment
```bash
git clone <this-repo> && cd local-llm-bench
uv venv && uv pip install httpx psutil transformers rich evalplus pytest
# engines you want to compare (install the ones you have):
brew install llama.cpp ollama                 # llama.cpp + ollama
brew install --cask lm-studio                 # LM Studio (run once to bootstrap `lms`)
# mlx_lm in its own venv (Apple): pip install mlx-lm
```

## 2. Models
- **MLX models** resolve from the HuggingFace cache (`mlx-community/...`).
- **GGUF** for the cross-engine comparison: download one `Q4_K_M` and point llama.cpp /
  ollama / LM Studio at the *same file* (see `bench/run_l0.py` for the path it expects).
- The L2 polyglot tasks: `git clone https://github.com/Aider-AI/polyglot-benchmark
  tasks/polyglot`.

## 3. Coding-eval grader (one-time)
EvalPlus's sandbox needs Linux. Build the arm64 container (OrbStack runs it natively):
```bash
docker build -t llmbench-evalplus grading/
```

## 4. Run
```bash
# L0 — engine bake-off (single-stream + concurrency, configs verified from logs)
python -m bench.run_l0 --phase both --soak 300
# L1 — model × quant (perf + coding + tools + ifeval + long-context)
python -m bench.run_l1
# L2 — agent-harness shootout (needs aider/opencode/goose/crush/claude-code installed)
python -m bench.run_l2 --tasks 10
# L3 — embed/RAG (needs the embed model served, e.g. ollama on :11435)
python scripts/run_l3.py
# tabulate
python -m bench.report
```

Results append to `results/runs.jsonl` (one row per measurement, with the exact config
used). Per-eval detail lands in `results/l1_evals/`.

## Notes / gotchas (the ones that cost us hours)
- **Single-tenant:** quiesce other consumers of the box during measured runs — the
  harness binds engines to `127.0.0.1` and takes down background services so nothing
  else perturbs the numbers.
- **macOS multiprocessing** defaults to `spawn`; the grader runs in a container to avoid
  the fallout (see [pitfalls.md](pitfalls.md) #6).
- **Verify, don't assume:** after each engine starts, check its log confirms full GPU
  offload + the batching/flags you intended. The harness records this per row.
- See [pitfalls.md](pitfalls.md) before trusting any number you produce.
