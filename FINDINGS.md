# Findings

Running log of results as each layer completes. Numbers from `results/runs.jsonl`
(reproduce: `uv run python -m bench.report`). Box: M4 Pro, 64 GB, wired 56 GB.

## L0 — Engine bake-off (Qwen3-Coder-30B-A3B, Q4, identical model across engines)

Apples-to-apples note: MLX uses 4bit-DWQ (~16 GB), the GGUF engines use the
*same* Q4_K_M file (18.6 GB) loaded by llama.cpp / ollama / LM Studio. DWQ vs
Q4_K_M differ in quant method (documented; bits-per-weight comparable).

### Single-stream (1 user) — MLX wins
| engine | decode t/s | prefill t/s (4k) | TTFT (short) | wired GB |
|---|---|---|---|---|
| **mlx_lm.server** | **90.3** | **817** | 0.07s | 19.9 |
| llama.cpp | 70.8 | 693 | 0.03s | 25.0 |
| ollama | 64.7 | 726 | 0.06s | 24.5 |
| LM Studio | 62.4 | 678 | 0.12s | 24.7 |

MLX is ~28% faster single-stream decode than the next best, best prefill, and
leanest memory at short context (it grows with context: 25 GB at 32k prompt).

### Under concurrency — the inversion
System throughput (tok/s) and TTFT p50 as concurrent requests rise:

| engine | c=1 | c=4 | c=16 | TTFT@c=4 | TTFT@c=16 |
|---|---|---|---|---|---|
| **llama.cpp** | 68 | **97** | **98** | 0.24s | 15.9s |
| LM Studio | 57 | 96 | 97 | 0.50s | 16.3s |
| mlx_lm.server | 87 | 87 | 86 (flat) | 4.5s | **22.5s** |
| ollama | 61 | 63 | 63 (flat) | 6.1s | 30.4s |

**Key finding: `mlx_lm.server` does no continuous batching.** Concurrent requests
serialize — throughput stays flat at the single-stream rate and TTFT balloons
(22.5s at c=16). llama.cpp / LM Studio batch properly: throughput climbs to ~98
t/s and TTFT stays sub-second through c=4. ollama also batches poorly.

### Verdict
- **This box's job is single-user (one person, one agent).** → **MLX is the right
  engine**: fastest decode + prefill, lowest single-stream latency. Confirms the
  current production choice, and grounds L1 on MLX.
- **If this were a multi-tenant server**, llama.cpp (or LM Studio, same llama.cpp
  core, ~+1s wrapper latency) would win on aggregate throughput.
- LM Studio ≈ llama.cpp on throughput (it *is* llama.cpp underneath) but adds a
  small serving overhead and a higher cold TTFT.

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

### Conclusions
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

## L2 — Agent harness
_pending._

## L3 — Real tasks + embed/RAG
_pending._
