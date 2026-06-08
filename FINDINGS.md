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

Soak/thermal: pending (soak-only run on mlx + llama.cpp).

## L1 — Model × quant
_pending — zoo downloading._

## L2 — Agent harness
_pending._

## L3 — Real tasks + embed/RAG
_pending._
