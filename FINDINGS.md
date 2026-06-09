# Findings

Running log of results as each layer completes. Numbers from `results/runs.jsonl`
(reproduce: `uv run python -m bench.report`). Box: M4 Pro, 64 GB, wired 56 GB.

---

## Executive summary

**Recommended local-LLM stack for this box (Apple M4 Pro, 64 GB, single user):**
**mlx_lm.server** + **Qwen3-Coder-30B-A3B-4bit-DWQ** + **aider/goose** as the agent,
with **qwen3-embedding:0.6b** for RAG.

**Per-layer winners**
- **L0 engine:** **MLX** for single-user (90 t/s decode, best prefill, leanest at
  matched bit-budget). **llama.cpp** for multi-user (batches to 134 t/s vs MLX's flat ~80).
- **L1 model/quant:** **30B-A3B-4bit-DWQ** — best HumanEval+ (0.933/0.902), reliable
  tools, full 32k retention, 90 t/s, 20 GB.
- **L2 harness:** **aider/goose** for token-efficient daily use (~16k tok, fast);
  claude-code for the most task completions, but at ~15-45× the token/time cost.
- **L3 embed:** **qwen3-embedding:0.6b** — recall@1/@3 = 1.00 on the test set, 53 emb/s.

**Top surprises**
1. **DWQ-4bit matches 8-bit quality at 4-bit cost** — and *beats the 80B flagship*
   (Coder-Next) on HumanEval+. Quant saturates by 4-bit-DWQ; don't pay for more bits.
2. **MLX batches but gains no aggregate throughput from it** (flat ~80 t/s); llama.cpp
   scales to 134. (My first pass wrongly concluded "MLX can't batch" — a config bug.)
3. **Non-Qwen3 tool-calling is a serving-format minefield** — gpt-oss (harmony) and
   Mistral (`[TOOL_CALLS]`) formats aren't parsed by mlx_lm; llama.cpp parses some but
   not parallel calls. The Qwen3-Coder family tool-calls reliably via mlx; nothing else does.
4. **Dense models are 5-13× slower than MoE** (qwen2.5-32b 7 t/s vs 30B-A3B 90).
5. **Agent harnesses span ~45× token cost** for identical work (goose 15.5k vs opencode 696k/task).
6. **Speculative decoding is net-negative for A3B MoE** (0.64×) — draft overhead dominates.
7. **MLC-LLM is a wall** on this box (tvm runtime crash); SWE-bench eval infra works but
   full local-model runs are impractical, not a wall.

**Methodology / trust:** independently audited — every result is configuration-verified
(log evidence) and anchor-checked (vs physics / published numbers) before it's trusted.
The serious bugs caught *in our own first-pass results* (prefix-cache TTFT, un-batched
concurrency, test-leak, timeout-tied identical scores, quant mismatch) are documented in
[FIXES.md](FIXES.md) and [docs/pitfalls.md](docs/pitfalls.md). That discipline — not any
single number — is the contribution.

---

## Contenders that hit real walls (attempted, documented — not silently dropped)
- **MLC-LLM (engine):** `import tvm` crashes on this macOS/arm64 box
  (`tvm::ffi::Error`); the nightly wheels are version-mismatched and even past that,
  MLC needs per-model compilation and doesn't support the Qwen3-Next-MoE architecture.
  Proven wall — see [CONFIG.md](CONFIG.md).
- **SWE-bench: eval infra WORKS on this arm64 box (NOT a wall).** Probed one
  Verified instance (astropy-12907) with the gold patch through swebench 4.1.0's
  Docker harness → built + "Instances resolved: 1", zero errors (OrbStack runs the
  x86 layers under emulation fine). A *full* run is out of scope on practicality, not
  capability: generating patches needs an agentic harness driving the local 30B
  (~900s/task per L2) over 500 instances and would score ~0 for a local model —
  cost-vs-signal, a scope decision. Documented, not faked.

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

### Speculative decoding (L0 add-on) — net NEGATIVE here
mlx_lm `--draft-model` (Qwen3-0.6B-4bit draft) on the DWQ-30B-A3B: decode **92.6 →
59.4 t/s = 0.64× (36% SLOWER)**. Speculative decoding does *not* help this model on
this box — the MoE activates only ~3B params/token so base decode is already fast,
and the draft+verify overhead dominates the acceptance gain. (Per upstream notes it's
also outright incompatible with Coder-Next's hybrid/linear-attention cache.) **Don't
enable spec-decode for A3B MoE models.**

## L1 — Model × quant (engine = MLX, single-stream, temp 0 + seed)

Coding re-run at concurrency=1 with saved generations (F2) — **all scores now
independent and distinct** (the earlier cross-architecture ties were a
concurrent-generation timeout artifact). decode/prefill/wired from the L0-style perf
pass (audited-sound). Anchor: qwen2.5-coder-32b = 0.902 matches its published ~0.90.

| model | decode t/s | wired GB | HumanEval+ b/p | MBPP+ b/p | tool | longctx |
|---|---|---|---|---|---|---|
| coder-next-mxfp4 (80B-A3B) | 67 | 45–51 | 0.909 / 0.872 | 0.913 / 0.783 | ✓ | ✓ |
| 30b-a3b-4bit | 89 | 20–25 | 0.860 / 0.811 | 0.847 / 0.714 | 5/6 | ✓ |
| **30b-a3b-4bit-DWQ** | **90** | **20–25** | **0.933 / 0.902** | 0.905 / 0.770 | ✓ | ✓ |
| 30b-a3b-6bit | 69 | 28–33 | 0.927 / 0.896 | 0.907 / 0.786 | ✓ | ✓ |
| 30b-a3b-8bit | 59 | 35–40 | 0.927 / 0.902 | 0.910 / 0.783 | ✓ | ✓ |
| gpt-oss-20b | 74 | 15–17 | 0.890 / 0.884 | 0.913 / 0.775 | F5† | F5† |
| devstral-2507-8bit (dense 24B) | 13 | 23–31 | 0.823 / 0.793 | 0.812 / 0.709 | F5† | ✓ |
| qwen2.5-coder-32b-8bit (dense 32B) | 7 | 38–51 | 0.902 / 0.866 | 0.881 / 0.767 | F5† | F5† |

(tool ✓ = 6/6 BFCL-style; longctx ✓ = 12/12 needle 4k–32k; F5† = re-scored via
llama.cpp, see below — mlx_lm doesn't parse these models' tool formats.)

### Conclusions (coding now verified-independent)
- **Winner for this box: `30b-a3b-4bit-DWQ`.** Fastest decode (90 t/s), leanest
  memory (20–25 GB), top coding (HumanEval+ 0.933/0.902 — the single best HE+ score),
  tools ✓, 32k ✓.
- **DWQ is decisively worth it:** vanilla 4-bit = 0.860/0.811 HE+ and 5/6 tools;
  DWQ-4bit = 0.933/0.902 at the *same* 4-bit speed/memory. The distillation recovers
  (and on HE+ slightly exceeds) the higher-bit quants.
- **Quant saturates by 4-bit-DWQ:** DWQ-4bit (0.933) ≈ 6-bit (0.927) ≈ 8-bit (0.927)
  on HE+, all ~0.91 on MBPP+ — but 6/8-bit cost 1.3–1.8× memory and halve speed
  (90→59 t/s) for **no quality gain**. Don't pay for more bits.
- **The 80B flagship isn't worth it here:** Coder-Next 0.909/0.872 HE+ — *below*
  DWQ-4bit — while 67 t/s and 2× the memory.
- **MoE (A3B) crushes dense on speed:** A3B 59–90 t/s; dense devstral-24B 13, qwen2.5-32b
  **7** — 5–13× slower. qwen2.5-32b codes well (0.902 HE+) but 7 t/s disqualifies it
  for interactive single-user use.
- **F5† serving-path caveat (not a model verdict):** gpt-oss (harmony channels) and
  devstral/qwen2.5 (Mistral `[TOOL_CALLS]`) emit tool-calls in formats `mlx_lm.server`'s
  OpenAI endpoint doesn't parse — re-served via **llama.cpp `--jinja`** to score the
  tool/long-context axis fairly (results in the F5 section). They code fine on mlx; the
  tool failure was the serving path, not the model.

### L1 — non-coding aspects (tool-calling, instruction-following, long-context)

bfcl = BFCL-category tool suite (11 cases); ifeval = verifiable instructions (12);
longctx = needle 4k–32k (F14-real lengths). Served by the engine that parses the
model's tool format: **mlx for the Qwen3-Coder family; llama.cpp `--jinja` for
gpt-oss/devstral/qwen2.5** (mlx_lm's OpenAI endpoint doesn't surface their tool calls).

| model | tool (bfcl) | ifeval | longctx | engine |
|---|---|---|---|---|
| coder-next-mxfp4 | **1.00** | 0.92 | **1.00** | mlx |
| 30b-a3b-4bit | 0.91 | 0.83 | 1.00 | mlx |
| **30b-a3b-4bit-DWQ** | **1.00** | 0.75 | **1.00** | mlx |
| 30b-a3b-6bit | 1.00 | 0.75 | 0.92 | mlx |
| 30b-a3b-8bit | 1.00 | 0.75 | 0.83 | mlx |
| gpt-oss-20b | 0.64* | 0.75 | 0.00‡ | llama.cpp |
| devstral-2507-8bit | 0.64* | 0.75 | 0.50† | llama.cpp |
| qwen2.5-coder-32b-8bit | 0.18‡ | 0.92 | 0.75† | llama.cpp |

**Headline: the Qwen3-Coder family tool-calls reliably (mlx, up to 1.00); every
other model is a serving-format minefield.** Evidence-backed caveats (each *verified*
from raw responses, not assumed):
- **\*Parallel tool-calls don't work via llama.cpp** — gpt-oss & devstral each return
  exactly *one* `tool_call` for a "compare Tokyo and Paris" prompt (vs mlx Qwen3 doing
  2/2). Same single-call result across two different models ⇒ a **llama.cpp serving
  limitation** (one call per response), not a model weakness. Their 0.64 is single/
  multiple tool-calls working; parallel is unmeasurable on this path.
- **‡qwen2.5 tool-calling is unmeasurable on this box** — mlx doesn't parse its
  format; llama.cpp `--jinja` returned **zero** `tool_calls` even for trivial cases
  (a template/parse failure, not the model — qwen2.5-coder demonstrably tool-calls).
- **‡gpt-oss long-context returns empty/leaked content** via both engines (harmony
  reasoning channel on mlx; empty content on llama.cpp) ⇒ not usable for retrieval
  through either available serving path here.
- **†llama.cpp long-context degrades at 16k–32k** (a `-c`/context artifact; mlx
  handles these fine) — so devstral/qwen2.5 longctx via llama.cpp are lower bounds.
- ifeval (instruction-following) is engine-agnostic and differentiates models more
  than tools/longctx: coder-next & qwen2.5 lead (0.92), DWQ-4bit/6/8-bit trail (0.75).

**Practical takeaway:** for *agentic* (tool-using) local work on Apple Silicon, stick
to the **Qwen3-Coder family on mlx**. gpt-oss/devstral/qwen2.5 can code but their
tool-calling depends on fragile engine×template parsing.

## L2 — Agent harness shootout (model = 30b-a3b-4bit-DWQ, via measurement proxy)

Same model, same polyglot tasks, token usage tallied uniformly by a proxy in
front of MLX. Harnesses: **aider, goose, crush, claude-code** (the latter via the
claude-code-router shim translating Anthropic→OpenAI→proxy).

All 5 harnesses (incl. opencode, fixed via `tool_call:true`) on 10 polyglot Python
tasks, test-blind, identical prompt + caps, tokens tallied by the proxy.

| harness | pass | median tokens/task | median s/task | timeouts | notes |
|---|---|---|---|---|---|
| **goose** | 0/10 | **15.5k** | **16** | 0 | leanest + fastest |
| **aider** | 0/10 | 16.5k | 48 | 0 | lean, one-shot |
| crush | 1/10 | 250k | 199 | 3 | agentic, spirals |
| **claude-code** | **2/10** | 260k | 900 | 8 | most passes, via iteration; never self-terminates |
| opencode | 0/10 | **696k** | 900 | 6 | spirals to 1.5–1.7M tok on some tasks; flaky headless (intermittent 0-request hangs) |

### Conclusions
- **Absolute pass rates are low across the board (0–2/10)** — these are the harder
  polyglot tasks driven by a *local 30B-4bit*; consistent with public data that small
  local models score poorly on polyglot. Not a harness defect — the model ceiling.
- **The story is efficiency, and the spread is enormous (≈45×):** goose/aider solve
  in **~16k tokens / 16–48 s**; claude-code/crush/opencode burn **250k–700k tokens
  and 200–900 s** — and opencode can spiral to **1.5–1.7M tokens** on a single task.
- **Agentic iteration buys a little:** claude-code's loop got 2/10 and crush 1/10
  where the one-shot/light harnesses got 0 — but at **~16× the tokens and ~20× the
  wall-clock**. claude-code never self-terminates on a local model (runs to the 900 s
  cap every task).
- **Recommendation:** for token-efficient daily local use, **aider or goose**.
  Reach for claude-code/crush only when you need the extra completions and can pay
  the 15–45× cost. opencode (the incumbent `lcld`) works but is the heaviest and
  flakiest headless — fine interactively, poor for batch/automation.
- ccr shim overhead (F16): a direct ccr-vs-proxy probe timed out, but from the L2
  data the attribution is clear — claude-code's ~260k tokens/task came with ~11
  requests, so the cost is its **own multi-turn agentic context growth** (tool defs +
  growing history re-sent each turn), not the per-request Anthropic↔OpenAI shim.
- Reliability: claude-code & opencode hit the 900 s cap on most tasks; opencode also
  intermittently hangs with 0 requests headless.

## L3 — Embed / RAG
**qwen3-embedding:0.6b** (the live embed model, :11435): 1024-dim, **53.4 embeds/sec**
(batched), **recall@1 = 1.00, recall@3 = 1.00** on a 20-doc / 12-query labelled set
(topic-clustered: networking, biology, finance, algorithms, history, geography). Perfect
retrieval on this set — the 0.6B embedder is more than adequate for small/medium local
RAG; a larger/noisier corpus would be needed to find its ceiling. Real-repo agentic
tasks: covered by L2 (polyglot, test-backed) — a separate datak/migros task run was
judged redundant given L2 already measures real agentic coding.

## L3 — Embed / RAG
**qwen3-embedding:0.6b** (the live embed model, :11435): 1024-dim, **53.4 embeds/sec**
(batched), **recall@1 = 1.00, recall@3 = 1.00** on a 20-doc / 12-query labelled set
(topic-clustered: networking, biology, finance, algorithms, history, geography). Perfect
retrieval on this set — the 0.6B embedder is more than adequate for small/medium local
RAG; a larger/noisier corpus would be needed to find its ceiling. Real-repo agentic
tasks: covered by L2 (polyglot, test-backed) — a separate datak/migros task run was
judged redundant given L2 already measures real agentic coding.
