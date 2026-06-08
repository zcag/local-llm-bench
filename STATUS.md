# Status

**Phase 1 (scaffold + harness): measurement path validated.** See [PLAN.md](PLAN.md) for full design.

## Done
- Python harness (`bench/`), run with `uv` ON tardis (localhost load-gen, no network noise):
  - `load/client.py` — async OpenAI-compat streamed request → TTFT, decode tok/s, prefill tok/s, exact token counts (server `usage`, tokenizer fallback).
  - `load/runner.py` — concurrency sweep (warmup + N repeats + p50/p90 TTFT + aggregate system tok/s).
  - `metrics.py` — background mem sampler (used + wired + engine RSS; no sudo). Thermal = decode-decay over soak.
  - `results.py` — append-only JSONL with ts/host/git-rev.
  - `scripts/smoke.py` — non-destructive validation vs live MLX.
- Deploy: `rsync ~/proj/llm-bench/ tardis:proj/llm-bench/` then `uv run`.

## Validated baseline (Coder-Next-mxfp4, live MLX, concurrency 1)
| scenario | TTFT p50 | decode t/s | prefill t/s | mem used | wired |
|---|---|---|---|---|---|
| short (25→36) | 0.16s | 70.2 | 150 | 55.8 | 45.9 |
| 3.6k prompt (→32) | 5.32s | 64.2 | 682 | 57.5 | 48.7 |

## tardis inventory (2026-06-08)
- Running: `io.cagdas.mlx` (:11434, mlx_lm 0.31.3, Coder-Next-mxfp4), `io.cagdas.ollama-embed` (:11435), open-webui (OrbStack :3000).
- Engines installed: **only ollama 0.23.1**. Need: llama.cpp, LM Studio, (MLC stretch).
- Models cached (MLX): Coder-Next-mxfp4, Qwen3-Coder-30B-A3B-4bit-DWQ. Everything else TBD-download.
- Control: MLX via launchd (KeepAlive=true → `bootout` to stop). `uv` present.

## Next
1. **Engine adapters + quiesce** — `bench/engines/{mlx,llamacpp,ollama,lmstudio}.py` (start/stop/health/model swap) + `quiesce.py` (bootout launchd units, stop open-webui, assert idle, restore).
2. **Pin output length** in decode scenarios (~256 forced tokens) for clean decode-rate.
3. **L0 engine bake-off** — install llama.cpp + LM Studio, pull GGUF Qwen3-Coder-30B-A3B, run the cross-engine grid.

## Open decisions
- Model list for L1 (confirm: Coder-Next, 30B-A3B quant sweep 4/6/8, gpt-oss-20b, GLM-Air, Devstral, Qwen2.5-Coder-32B). Big downloads — confirm before pulling.
- SWE-bench slice: default fixed ~30-task verified subset.
- Publish target: public repo+blog under Cagdas's name vs private.
