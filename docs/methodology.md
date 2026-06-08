# Methodology

## Why layers
A "local LLM" is not one thing — it's an **inference engine** serving a **quantized
model** driven by an **agent harness**, plus an **embedding model** for RAG. A number
quoted without saying which of these is being varied is meaningless. So we isolate
four layers and vary one at a time, holding the rest fixed and at their best config.

## L0 — engine bake-off
Same model across every engine (mlx_lm.server, llama.cpp, ollama, LM Studio, MLC),
two phases because the *best config differs by regime*:
- **single-stream** (`parallel=1`): max per-request latency/throughput.
- **concurrency** (`parallel=16`): real batching ceiling.

Metrics: decode tok/s, prefill tok/s at 4k/16k/32k, **TTFT** (cache-busted so it's
honest), system throughput + p50/p90 TTFT under concurrency {1,2,4,8,16}, peak wired
memory, and an 8-min soak for thermal throttle.

**Fairness:** cross-engine quant is matched bit-budget (MLX-4bit-vanilla ≈ GGUF-Q4_K_M,
~4.5 bpw); MLX-4bit-DWQ reported separately as MLX's best. Full GPU offload + flash
attention verified from each engine's log, not assumed.

## L1 — model × quant
Fixed engine (MLX), 8 models including a 4/6/8-bit + DWQ quant sweep of one model.
Per model, measured *aspects*:
- **coding correctness** — HumanEval+ and MBPP+ (via the EvalPlus harness, graded in
  an arm64-Linux container because its sandbox doesn't work on macOS).
- **tool-calling** — a transparent BFCL-category suite (simple / multiple / parallel /
  parallel-multiple / irrelevance), AST-style arg-checked.
- **instruction following** — IFEval-style verifiable constraints (format, length,
  casing, keywords), programmatically graded.
- **long-context retention** — needle-in-a-haystack at 4k–32k, multiple depths.
- **speed/memory at realistic context fill** (not the empty-prompt vanity number).

Decoding is greedy (temp 0, fixed seed) for reproducibility. Some models (gpt-oss
harmony, Mistral `[TOOL_CALLS]`) need an engine that *parses* their tool format —
served via llama.cpp `--jinja` for those cells so the tool axis is real, not a
serving-path artifact.

## L2 — agent harness shootout
Same model (the L1 winner), same tasks (Aider polyglot Python exercises), driven by
each harness: aider, opencode, goose, crush, and Claude Code (via the
[claude-code-router](https://github.com/musistudio/claude-code-router) Anthropic→OpenAI
shim). Fairness rules:
- **Test-blind:** each harness gets only the spec + stub; the hidden test is withheld
  until grading. (Early versions leaked the tests into the workdir — see pitfalls.)
- **Identical prompt + caps:** same instruction text, same context/output budget for
  every harness, so token counts are comparable.
- **Uniform token accounting:** all harnesses point at a measurement proxy
  (`bench/proxy.py`) that forwards to the engine and tallies tokens — so efficiency is
  measured the same way regardless of how each harness prompts internally.

Metrics: task pass rate (pytest), tokens/task, wall-clock/task.

## L3 — embed / RAG
The live embedding model (qwen3-embedding:0.6b): embeddings/sec throughput + recall@1
and @3 on a hand-built labelled corpus.

## The verified-config gate
The core discipline: **a result is invalid until its config is proven applied (log
evidence) and its number passes an anchor check** — decode tok/s vs the box's memory
bandwidth ceiling, coding pass@1 vs published values, TTFT vs prompt÷prefill-rate.
Each result row records the exact engine command/flags used. See
[`../CONFIG.md`](../CONFIG.md) for the ledger.
