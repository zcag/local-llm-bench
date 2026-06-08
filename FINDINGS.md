# Findings

Running log of results as each layer completes. Numbers from `results/runs.jsonl`
(reproduce: `uv run python -m bench.report`). Box: M4 Pro, 64 GB, wired 56 GB.

## Contenders that hit real walls (attempted, documented — not silently dropped)
- **MLC-LLM (engine):** `import tvm` crashes on this macOS/arm64 box
  (`tvm::ffi::Error`); the nightly wheels are version-mismatched and even past that,
  MLC needs per-model compilation and doesn't support the Qwen3-Next-MoE architecture.
  Proven wall — see [CONFIG.md](CONFIG.md).
- _(others appended here as encountered: e.g. SWE-bench x86-Docker-on-ARM, GUI-only
  harnesses if they can't run headless.)_

## L0 — Engine bake-off (Qwen3-Coder-30B-A3B, Q4, identical model across engines)

**v2 (post-audit, verified).** Cross-engine quant is now MATCHED-bpw: MLX-4bit
(vanilla, ~4.5bpw) vs GGUF-Q4_K_M (~4.5bpw) — same budget, no DWQ advantage.
MLX-4bit-DWQ reported as a bonus row (MLX's best). Two phases, each at the
engine's best config for that regime (single-stream parallel=1; concurrency
parallel=16). TTFT cache-busted (unique prefix/request) so it's honest. Configs +
GPU-offload evidence recorded per row.

### Single-stream (1 user, matched-bpw) — MLX wins, and NOT because of DWQ
| engine | quant | decode t/s | prefill@4k | prefill@32k | TTFT@32k | wired GB |
|---|---|---|---|---|---|---|
| **mlx_lm** | 4bit | **89.3** | 848 | 757 | 25.0s | **19.9** |
| mlx_lm | 4bit-DWQ | 90.3 | 843 | 733 | 25.2s | 19.9 |
| llama.cpp | Q4_K_M | 70.7 | 789 | 568 | 32.9s | 25.8 |
| ollama | Q4_K_M | 65.0 | 718 | 354 | 52.0s | 24.5 |
| LM Studio | Q4_K_M | 57.3 | 708 | 534 | 34.6s | 24.8 |

MLX ~26% faster decode AND ~6 GB leaner than llama.cpp **at the same bit-budget**
(the earlier worry that DWQ caused the win is disproven — vanilla 4bit ties DWQ).
TTFT now physically consistent (≈ prompt_tokens / prefill_tps).

### Concurrency (parallel=16, batching enabled) — the corrected picture
System throughput (tok/s) as concurrent requests rise:

| engine | c=1 | c=2 | c=4 | c=8 | c=16 |
|---|---|---|---|---|---|
| **llama.cpp** | 63 | 86 | 98 | 105 | **134** |
| ollama | 37 | 64 | 80 | 89 | 101 |
| LM Studio | 63 | 87 | 96 | 96 | 96 |
| mlx_lm | 78 | 82 | 82 | 78 | **76 (flat)** |

**Verified finding (corrects BOTH earlier wrong claims):** MLX *does* batch —
its log shows `Prompt Cache: 10 sequences` concurrently, so it is NOT serializing
(my original "MLX can't batch" was wrong, a config artifact). But MLX's batched
decode gives **no aggregate-throughput gain** (flat ~80 t/s 1→16), while GGUF
engines' batching scales (llama.cpp 63→134). So the accurate statement is:
*MLX batches but batching doesn't raise aggregate throughput on this box;
llama.cpp's does.*

### Verdict
- **Single-user (this box's job): MLX wins** — fastest decode+prefill, leanest
  memory, at matched bit-budget. The single-stream advantage is real.
- **Multi-user aggregate throughput: llama.cpp wins** (134 vs MLX's flat 80 at
  c=16) — because GGUF batching scales and MLX's doesn't, NOT because MLX serializes.
- LM Studio ≈ llama.cpp single-stream-wise but plateaus earlier under load (96 vs 134).

### Soak / thermal (8-min sustained decode)
| engine | first-min t/s | last-min t/s | throttle |
|---|---|---|---|
| mlx_lm.server | 90.3 | 90.0 | **0.3%** |
| llama.cpp | 70.7 | 70.6 | **0.1%** |

**This M4 Pro does not thermally throttle on sustained single-stream decode** —
burst ≈ sustained. The earlier worry (that a 30s speed run would overstate daily
use) doesn't hold here; the steady-state numbers match the burst numbers.

## L1 — Model × quant (engine = MLX, single-stream, temp 0 + seed)

| model | decode t/s | prefill@32k | wired GB | HumanEval+ base/plus | tool | longctx |
|---|---|---|---|---|---|---|
| coder-next-mxfp4 (80B-A3B) | 67 | 580 | 45–51 | 0.909 / 0.872 | ✓ | ✓ |
| 30b-a3b-4bit | 89 | 749 | 20–25 | 0.823 / 0.793 | 5/6 | ✓ |
| **30b-a3b-4bit-DWQ** | **90** | 748 | **20–25** | **0.933 / 0.902** | ✓ | ✓ |
| 30b-a3b-6bit | 69 | 729 | 28–33 | 0.933 / 0.902 | ✓ | ✓ |
| 30b-a3b-8bit | 59 | 743 | 35–40 | 0.933 / 0.902 | ✓ | ✓ |
| gpt-oss-20b | 74 | 684 | 15–17 | 0.890 / 0.878 | ✗* | ✗* |
| devstral-2507-8bit (dense 24B) | 13 | 239 | 23–31 | 0.823 / 0.793 | ✗* | ✓ |
| qwen2.5-coder-32b-8bit (dense 32B) | 7 | 154 | 38–51 | re-running† | ✗* | 0.67 |

(tool = toolcalling score, ✓ = 6/6; longctx = needle 4k–32k, ✓ = 12/12)

> [!CAUTION]
> **L1 coding scores UNDER RE-VERIFICATION (audit F2).** Different architectures
> tied to 3 decimals (30b-a3b-4bit == devstral both 0.823/0.793; 6/8/DWQ-4bit all
> 0.933/0.902). Same-model quant ties may be real saturation, but the cross-model
> tie is a red flag and the original runs didn't save samples to prove independence.
> Re-running all 8 with saved generations before trusting the quant conclusions
> below. Memory + decode-throughput numbers are audited-sound; coding pass@1 is not
> yet. The gpt-oss/devstral/qwen2.5 tool+longctx scores are serving-path artifacts
> (F5), not capability — being re-served via an engine that parses their formats.

### Conclusions (PROVISIONAL — pending F2 re-verification)
- **Winner for this box: `30b-a3b-4bit-DWQ`.** Fastest decode (90 t/s), leanest
  memory (20–25 GB), and top-tier quality (HumanEval+ 0.902, tools ✓, 32k ✓) —
  matching 6/8-bit quality at 4-bit cost. **DWQ is decisively worth it**: vanilla
  4-bit drops to 0.793 and 5/6 tools; DWQ recovers it for free.
- **Quant curve (Qwen3-Coder-30B):** quality saturates at 4-bit-DWQ; 6/8-bit add
  memory + halve speed (90→59 t/s) for **no quality gain**. Don't pay for 8-bit.
- **The 80B flagship isn't worth it here:** Coder-Next scores *lower* than DWQ-4bit
  on HumanEval+ (0.872 vs 0.902), is slower (67 vs 90), and uses 2× memory.
- **MoE (A3B) crushes dense on speed:** A3B models 59–90 t/s; dense devstral-24B
  13 t/s, qwen2.5-32b **7 t/s** — 5–13× slower. Dense models are non-starters for
  interactive single-user use on this box regardless of quality.
- **\*Serving-path caveat (not a model verdict):** gpt-oss (harmony channels) and
  devstral/qwen2.5 (Mistral/`[TOOL_CALLS]`) emit tool-calls in formats
  `mlx_lm.server`'s OpenAI endpoint does **not** parse into `tool_calls` — so they
  score ~1/6 on tools and gpt-oss leaks `<|channel|>analysis<|message|>` as content
  (tanking its needle test). They *can* code; they just can't tool-call through
  this serving path. **For agentic use on mlx_lm.server, use the Qwen3-Coder family.**
- **†qwen2.5 HumanEval+** first run hit 0.0 — a harness artifact (concurrent
  generation timeouts at 7 t/s, since MLX doesn't batch), not the model. Re-run at
  concurrency 1 deferred to end of run (changes no ranking — disqualified on 7 t/s
  speed regardless; verified separately that it generates correct code).

## L2 — Agent harness shootout (model = 30b-a3b-4bit-DWQ, via measurement proxy)

Same model, same polyglot tasks, token usage tallied uniformly by a proxy in
front of MLX. Harnesses: **aider, goose, crush, claude-code** (the latter via the
claude-code-router shim translating Anthropic→OpenAI→proxy).

**opencode excluded** (documented): won't run headless against a hermetic
custom-provider config — initializes then hangs with 0 model requests and no error.
The daily `lcld` setup relies on opencode's own persisted auth/config, which a clean
reproducible benchmark deliberately avoids.

Early validation signal (1 hard task, affine-cipher) — token cost varies wildly
for identical work: **aider ~28k**, **goose ~16k**, **claude-code ~179k**,
**crush ~241k** prompt tokens. Full results pending.
_full run pending._

## L3 — Real tasks + embed/RAG
_pending._
