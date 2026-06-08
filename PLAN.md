# Local LLM Stack — End-to-End Benchmark

## CONTRACT (locked): breadth is the point
Every engine, model, and harness is a CONTENDER — none may be dropped from the
comparison. Economize only on DEPTH (task counts, repeat counts) and only when it
doesn't erase a comparison. A contender that "doesn't work" is investigated and
made to work at its best config; if it's a genuine wall, that's PROVEN with
evidence and documented — never silently cut. Full-breadth matrix:
- Engines: mlx_lm · llama.cpp · ollama-GGUF · ollama-MLX · LM Studio · MLC-LLM
  (+ spec-decode config variant where supported)
- Models: all 8, with tool-calling made to work per model (engine×model cell)
  Evals: HumanEval+ · MBPP+ · BFCL · IFEval · long-context · tool-calling
- Harnesses: aider(best-config) · opencode · goose · crush · claude-code
  (+ attempt cline/continue/roo — prove if GUI-walled)
- Suites: attempt SWE-bench (prove if x86-on-ARM is a wall)
- Depth concessions: polyglot Python + partial task count; n=3 repeats.

---


**Goal:** publishable/blog-grade, reproducible benchmark of the whole local-LLM stack on `tardis`,
from inference engine → model/quant → agent harness → real end-to-end workload, plus the embed/RAG path.
**Not** a quick "which is faster" — a defensible, charted, version-pinned comparison.

## The box (single device under test)
Mac mini 2024 · M4 Pro (14c CPU / 20c GPU) · **64 GB unified** · `iogpu.wired_limit_mb=57344` (~56 GB for weights+KV) · macOS Tahoe 26.4.
Implications: weights+KV must fit ~56 GB; **thermals throttle** on sustained load (burst ≠ sustained — we measure both); only one box, so "concurrency" = many requests at one server, not a cluster.

## Layer cake (each layer isolatable)

### L0 — Inference engine bake-off ("the backend")
Same model + comparable quant, swap the engine. Fair cross-engine model = **Qwen3-Coder-30B-A3B** (exists as both MLX and GGUF); Coder-Next is MLX-only so it rides along in MLX-only rows.

Engines:
- `mlx_lm.server` (incumbent, MLX)
- `llama.cpp` / `llama-server` (GGUF, Metal)
- Ollama — llama.cpp backend **and** its newer MLX backend (two rows)
- LM Studio server (MLX + GGUF)
- MLC-LLM (Metal) — stretch / optional

Metrics: prefill tok/s · decode tok/s · **TTFT** · throughput @ concurrency {1,2,4,8,16} · peak RSS + wired mem · cold load time · max context before OOM · OpenAI-compat fidelity · tool-call wire-format correctness · crash/stall rate over a 30-min soak · **burst vs sustained decode (thermal)**.

### L1 — Model × quant (fix engine = L0 MLX winner)
Models (all must fit ≤56 GB; MoE A3B keeps active params small):
- Qwen3-Coder-Next-mxfp4 (80B-A3B, ~39G) — incumbent
- Qwen3-Coder-30B-A3B — quant sweep **4 / 6 / 8-bit DWQ** (the quality↓/speed↑/mem↓ curve)
- gpt-oss-20b; gpt-oss-120b-MoE (mxfp4 ~63G — **likely won't fit**, test/flag)
- GLM-4.x-Air
- Devstral-Small-2507 (24B)
- Qwen2.5-Coder-32B (dense baseline)

Aspects per model (the "different aspects"): coding correctness · tool-calling reliability (schema adherence, parallel/multi-tool) · instruction following · long-context retention (needle + repo-scale) · structured/JSON output (the ```json-fence footgun) · reasoning · **decode speed at realistic context fill** (not the empty-context vanity number).

### L2 — Agent harness shootout (HEADLINE) — fix L1 winner model
Same model, same tasks, swap the agent that drives it:
- **opencode** (incumbent)
- **aider** — test its edit formats (whole / diff / udiff / architect)
- **Claude Code on local** — via `claude-code-router` or LiteLLM Anthropic-compat shim (**the shim is a pinned variable**, see Risks)
- **crush** (charmbracelet)
- **cline** / roo-code
- **continue.dev**
- **goose** (Block)

Metrics: end-to-end **task success rate** (tests pass) · edit/diff-apply success · tool-loop count · **tokens per task** (efficiency — a great model wasted by a chatty harness) · wall-clock per task · retry/derail/giveup rate · cost-equivalent (if billed). This layer measures *the harness*, holding the model constant — so harness prompt/context-management differences are the signal, not noise.

### L3 — Real end-to-end + embed/RAG
- **Your real tasks:** small fixes/features from `datak` / `migros` / `robot` / infra, run through the winning L2 config. The only number that reflects daily reality.
- **Embed path:** tela's `qwen3-embedding:0.6b` (:11435) — embed throughput + retrieval quality (recall@k on a hand-labeled query set against the tela corpus).

## Methodology spine (trust)
- **Workload sets:** Aider polyglot · SWE-bench-verified subset · BFCL (tool-calling) · needle+repo-QA (long ctx) · your real tasks.
- **Driver:** one orchestrator — starts/stops each engine, health-checks, fires the load, records tok/s·TTFT·mem·thermals per run. Fixed temp/seed. **N repeats for variance** (local inference is noisier than people report — report median + spread, not a single number).
- **Grading:** functional tests where possible; LLM-as-judge (**Claude via API, grader-only, never in the measured path**) for open-ended.
- **Reproducibility:** pin engine+model+quant+shim versions; capture `iogpu` + thermal state; log everything to structured JSONL → charts.
- **Reuse, don't reinvent:** lean on existing harnesses — `llama-bench` (engine raw), `llmperf`/`guidellm`/`vllm benchmark_serving` (OpenAI-endpoint load), **aider's own benchmark repo**, **SWE-bench official harness**, **BFCL runner**. We orchestrate + normalize their outputs, not re-implement evals.

## Phasing (full matrix, made tractable)
1. **Scaffold + harness** — driver, metrics capture, engine start/stop adapters, results schema. Smoke-test on one engine+model.
2. **L0 engine bake-off** — raw + serving throughput, the fair 30B-A3B cross-engine grid. → pick MLX serving winner.
3. **L1 model×quant** — quality+speed aspects on the L0 winner engine. → pick model + quant.
4. **L2 harness shootout** — the headline grid on the L1 winner. (Includes standing up the Claude-Code-on-local shim.)
5. **L3 real tasks + embed/RAG.**
6. **Synthesis** — charts, the writeup, "recommended daily-driver config" + "fastest" + "best-quality" callouts.

## Risks / sharp edges
- **Coder-Next is MLX-only & spec-decode-incompatible** (hybrid cache) — it can't appear in GGUF engine rows; keep cross-engine claims to models that exist in both formats.
- **Claude-Code-on-local needs a translation shim** (Anthropic↔OpenAI). The shim adds latency + can mangle tool-calls; pin its version and measure shim overhead separately so it's not blamed on the model.
- **Thermal throttling** on an M4 Pro under sustained load — a 30-min soak will read differently than a 30-sec burst. Both reported.
- **Harness fairness** — different default system prompts/tool sets/context mgmt. We hold model+tasks constant and treat the rest as the harness's measured behavior, but document each harness's config.
- **Memory ceiling** — gpt-oss-120b and 8-bit big models may not fit under 56 GB wired; OOM is a result, not a failure.
- **The ```json-fence bug** is a known Coder-Next quirk — structured-output scores must note whether a model needs post-strip.

## Open questions (decide before scaffolding)
- Driver language: Python (best eval-harness ecosystem — aider bench, SWE-bench, BFCL are all Python) — assume **Python** unless objected.
- SWE-bench subset size (full verified = expensive locally; propose a fixed ~30-task slice).
- Do we publish under your name (blog/GitHub) — affects how much polish + a public results repo.
