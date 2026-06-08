# Status

**Phase 1 done: harness + control plane validated.** See [PLAN.md](PLAN.md) for full design.
Next: install engines, pull model zoo, build workload/eval layer (L0→L3).

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

## Done (Phase 1)
- Engine adapters (`bench/engines/{base,mlx,llamacpp,ollama}.py`) — harness owns engine lifecycle.
- `quiesce.py` — takes box single-tenant, **always restores** (bootout→sleep→bootstrap; docker path resolved; missing-bin tolerant). Validated: full take-down → harness MLX → restore.
- Datapoint: 30B-A3B-4bit-DWQ ~185 t/s decode vs Coder-Next-mxfp4 ~70 t/s.
- tardis disk: 304 GB free, hf cache 55 GB.

## Next
1. **Install engines**: `brew install llama.cpp`; LM Studio (`lms` CLI); (MLC stretch). Write/test ollama + lmstudio adapters against real models.
2. **Pin output length** in decode scenarios (~256 forced tokens, ignore-eos where supported) for clean decode-rate.
3. **L0 engine bake-off** — pull GGUF Qwen3-Coder-30B-A3B, run cross-engine grid (mlx / llama.cpp / ollama×2 / lmstudio) at concurrency {1,2,4,8,16} + 30-min soak.
4. **Workload/eval layer** — wire Aider polyglot, SWE-bench subset, BFCL, tool-call suite, needle; LLM-judge via Claude API (grader-only).

## Public repo (live)
github.com/zcag/local-llm-bench (public, MIT, main). README + docs/ (methodology,
pitfalls, reproduce) + raw results published. **`git push origin main` after each
commit** so the public repo stays in sync as the re-run finalizes. Tela writeup link
in README is a TBD placeholder — wire the circular link once the URL exists.

## BREADTH-RESTORATION QUEUE (locked contract — see PLAN.md; no contender dropped)
Execute in order across autonomous wakeups. Mark [x] when done. GPU-bound items are serial.

### Harnesses (L2)
- [~] aider, goose, crush, claude-code — current L2 run (--tasks 10) in progress
- [ ] opencode — fixed (tool_call:true); validate then run --only opencode --tasks 10
- [ ] aider BEST config — auto-test loop + try edit-formats (diff/whole/architect); re-run
- [ ] cline / continue / roo — investigate headless drivers; add if possible, else PROVE GUI-wall + document

### Engines (L0)
- [ ] **RE-RUN L0 concurrency sweep — CONFIG ERROR (invalidated old data):** all engines ran with concurrency=1/auto-4. Re-run with batching maxed: mlx `--decode-concurrency 16 --prompt-concurrency 16`, ollama `NUM_PARALLEL=16`, llama.cpp `--parallel 16 --kv-unified`, lmstudio (find slots flag). Single-stream L0 still valid. (adapters already fixed)
- [ ] verify-config-with-evidence for EVERY engine before counting its run (CONFIG.md ledger): confirm full GPU offload + the intended batching/FA actually took effect (parse logs)
- [x] MLC-LLM — **WALL (proven)**: tvm runtime crashes on import on macOS/arm64 + Qwen3-MoE unsupported. Documented in CONFIG/FINDINGS. (no silent drop)
- [x] ollama-MLX backend — **n/a (documented)**: ollama 0.23.1 has no flag to select MLX for an arbitrary GGUF; the `ollama` row is its real serving path for this model.
- [ ] spec-decode — mlx_lm --draft-model (Qwen3-0.6B) on DWQ-30B; new L0 row (speedup vs base)

### Evals (L1 — add axes across all 8 models)
- [ ] tool-calling FIX — serve gpt-oss/devstral/qwen2.5 via llama.cpp --jinja (grammar tool-parse); re-score tool axis (engine×model cell)
- [ ] MBPP+ — evalplus --dataset mbpp, container-graded, all 8 models
- [ ] BFCL — Berkeley Function-Calling Leaderboard subset, all 8 models
- [ ] IFEval — verifiable instruction-following, all 8 models
- [ ] qwen2.5-coder-32b HumanEval+ rerun (concurrency 1) — pending from earlier

### Suites
- [ ] SWE-bench — attempt (verified subset); PROVE if x86-on-ARM Docker is a real wall

### L3
- [ ] embed/RAG — qwen3-embedding:0.6b throughput + recall@k
- [ ] real repo tasks — 2-3 small tasks from ~/proj via the L2-winner harness

### Synthesis (closing step, once all layers gated + done)
- [ ] full exec-summary in FINDINGS.md: recommended stack + per-layer winners + surprises + which contenders hit real walls + "independently audited" note
- [ ] **create the tela writeup section** (Tela MCP: create_page under a suitable space) from the final synthesis — narrative for the blog
- [ ] **wire circular links**: put the real tela URL into README.md + docs/README.md (replace the "(link TBD)" placeholders); set repo homepage to tela; (tela side links back to github.com/zcag/local-llm-bench). commit + `git push origin main`
- [ ] robot ping Cagdas with audited headline results + the tela + github links

## Open decisions
- Model list for L1 (confirm: Coder-Next, 30B-A3B quant sweep 4/6/8, gpt-oss-20b, GLM-Air, Devstral, Qwen2.5-Coder-32B). Big downloads — confirm before pulling.
- SWE-bench slice: default fixed ~30-task verified subset.
- Publish target: public repo+blog under Cagdas's name vs private.
