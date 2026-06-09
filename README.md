# local-llm-bench

**A rigorous, end-to-end benchmark of a *whole* local-LLM stack on Apple Silicon —
inference engine × model/quant × agent harness — built to be fair, reproducible,
and honest about its own mistakes.**

Most "local LLM benchmarks" measure one model on one engine and quote a tokens/sec
number with no config and no checks. This measures the **whole stack as layers**,
holds every contender at its *best, verified* configuration, and — the part most
benchmarks skip — **audits itself and publishes the bugs it caught in its own first
results.** If you run models locally on a Mac, the tooling here is meant to be
lifted and reused.

> 📊 **Narrative writeup:** **[Read it on the blog](https://tela.cagdas.io/spaces/10/pages/229/benchmarking-a-whole-local-llm-stack-on-an-m4-pro-and)**
> 📁 **This repo** is the tooling + raw data + methodology the writeup draws on.
> 🔬 **Status:** results are being finalized under a verified-config re-run (see
> [`FIXES.md`](FIXES.md)); the harness and methodology are stable. Numbers in
> [`FINDINGS.md`](FINDINGS.md) marked *provisional* are mid-re-run.

## What it measures — four isolatable layers

| Layer | Question | How |
|---|---|---|
| **L0 — engine** | Which serving engine is fastest/leanest for the same model? | mlx_lm · llama.cpp · ollama · LM Studio · MLC, single-stream **and** under concurrency, at matched bit-budget |
| **L1 — model × quant** | Which model/quant is the best quality/speed/memory tradeoff? | 8 models incl. a 4/6/8-bit + DWQ sweep; HumanEval+ · MBPP+ · BFCL-style tools · IFEval · long-context · decode/prefill/memory |
| **L2 — agent harness** | Same model, which agent driver gets the most done per token? | aider · opencode · goose · crush · claude-code (via an Anthropic→OpenAI shim), on polyglot tasks, token usage tallied by a proxy |
| **L3 — embed/RAG** | How good/fast is the embedding model? | throughput + recall@k on a labelled retrieval set |

The box under test: **Apple M4 Pro, 64 GB unified, macOS** (single-user inference rig).

## The part worth stealing: fairness as a gate, not a hope

Every result must pass a check before it's trusted — and we got this *wrong* on the
first pass in instructive ways. See **[`docs/pitfalls.md`](docs/pitfalls.md)** for the
full list; highlights:

- **"MLX can't batch" was false** — we'd run every engine with concurrency *disabled*
  by default. MLX batches fine; it just doesn't gain aggregate throughput from it.
- **TTFT numbers were prefix-cache hits** — 0.14 s reported where the prefill rate
  implied 25 s. Fixed by busting the cache per request.
- **A Mistral model and a Qwen model scored identically to 3 decimals** — a
  concurrent-generation timeout artifact, not real. Fixed by serial generation.
- **The agentic harnesses could read the hidden tests** they were graded on.
- **Quant comparison wasn't apples-to-apples** (MLX-4bit-DWQ vs GGUF-Q4_K_M differ
  in bit-budget).

These were caught by an **independent adversarial audit** + anchor checks (does the
number match physics / a published value?), and every fix is tracked in
[`FIXES.md`](FIXES.md) with the evidence. **The methodology is the contribution.**

## Reuse it

The harness is small, dependency-light Python. See [`docs/reproduce.md`](docs/reproduce.md).

```bash
uv venv && uv pip install httpx psutil transformers rich evalplus pytest
# L0 engine bake-off (two phases, configs verified from logs):
python -m bench.run_l0 --phase both
# L1 model×quant; L2 agent-harness shootout; L3 embed/RAG:
python -m bench.run_l1 ; python -m bench.run_l2 --tasks 10 ; python scripts/run_l3.py
```

Pieces you can lift independently: the **measurement proxy** (`bench/proxy.py`, uniform
token accounting + model rewrite in front of any OpenAI endpoint), the **engine
adapters** (`bench/engines/`), the **transparent eval suites** (`bench/evals/` —
BFCL-style tool-calling, IFEval, needle long-context, EvalPlus wrapper, embed/RAG),
and the **agent-harness adapters** (`bench/harnesses/`).

## Repo layout

```
bench/            the harness (engines, evals, harnesses, proxy, runners)
scripts/          re-run / setup / grading helpers
grading/          arm64-Linux container for EvalPlus (its sandbox breaks on macOS)
results/          raw data — runs.jsonl (one row per measurement) + l1_evals/ detail
docs/             methodology, pitfalls/audit, per-layer notes, reproduce
PLAN.md           design + the locked "breadth + verified-config" contract
CONFIG.md         the config ledger — best config + evidence, per contender
FINDINGS.md       results (with retraction markers where re-running)
FIXES.md          audit findings + fix status
```

## Methodology & docs

Start at **[`docs/`](docs/)** → methodology, the pitfalls/audit writeup, per-layer
details, and how to reproduce on your own Apple Silicon box.

## License

MIT — see [`LICENSE`](LICENSE). The polyglot exercises used by L2 come from the
[Aider polyglot benchmark](https://github.com/Aider-AI/polyglot-benchmark) (fetched
separately, not vendored here).
