# Fix tracker — issues found by adversarial audit (2 independent auditors + data confirmation)

Status: CONFIRMED (verified in data/code) · FLAGGED (plausible, needs check) · FIXED · WONTFIX(documented)

## DONE + VERIFIED (L0 redo, evidence in runs.jsonl)
- F1 (TTFT cache-bust) ✓ — TTFT now ≈ prompt/prefill_tps in data.
- F3 (MLX concurrency wired) ✓ — log shows `Prompt Cache: 10 sequences`; corrected finding: MLX batches but aggregate throughput stays flat (~80); llama.cpp scales 63→134.
- F4 (matched-bpw mlx-4bit vs Q4_K_M) ✓ — MLX single-stream win survives (89/19.9 vs 71/25.8); not DWQ.
- F8 (-ub 2048), F12 (spread), F15 (ollama ctx), F18 (config per row), F19 (cold-first/soak-last) ✓.
- F2 (coding independence) ✓ — re-ran all 8 at concurrency=1 w/ saved samples; ALL scores now distinct (ties were a timeout artifact); qwen2.5 0.0→0.902 (matches published ~0.90); HE+ & MBPP+ table rewritten. DWQ-4bit confirmed top.
- F10/F13/F14 ✓ (code). MLC=wall, ollama-MLX=n/a (documented).
- F5 ✓ — tool-calling fixed via llama.cpp (gpt-oss/devstral 0→0.64); verified caveats: parallel-calls=llama.cpp limit, qwen2.5 tools unmeasurable (no tool_calls emitted), gpt-oss longctx serving-broken (empty/harmony). Qwen3 family tools reliable via mlx. L1 non-coding table written.
- Open: F6/F7 (L2 redo), F16, spec-decode, SWE-bench, L3.
Nothing is a trusted result until its issue here is resolved AND verified with evidence (CONFIG.md).

## CRITICAL — invalidate published findings, must fix + re-run

| id | issue | confirmed? | invalidates | fix |
|----|-------|-----------|-------------|-----|
| F1 | **TTFT = prefix-cache artifact.** Runner fires n=3 identical prefill reps; reps 2-3 hit KV cache → median TTFT is the cached ~0.1s lie (real = prompt/prefill_tps ≈ 25s). Also cross-scenario: 4k/16k/32k share filler prefix. | YES (data: 30b-4bit ttft 0.14s vs implied 25s; non-cache models match) | all TTFT (L0 single+concurrency, L1) | Bust cache: unique per-request nonce prefix so every rep is uncached; or measure TTFT from cold sample[0]. Re-run TTFT-dependent rows. |
| F2 | **L1 coding scores possibly non-independent.** 4bit==devstral exactly 0.823/0.793 (diff architectures); 6/8/DWQ all exactly 0.933/0.902. | FLAGGED (ties real in data; cause unknown — samples not saved) | L1 HumanEval+ table + "DWQ matches 8bit" headline | Re-run all 8 with humaneval save_dir; inspect samples genuinely differ per model; confirm independent grading. |
| F3 | **MLX concurrency re-run never wired.** run_l0 builds `MLX(model,port)` → decode_concurrency defaults to 1 → still serializes; reproduces the retracted "MLX can't batch" bug. | YES (code: run_l0.build_engines) | L0 concurrency (already retracted) | Pass decode_concurrency=16, prompt_concurrency=16 to MLX in build_engines for the concurrency phase. |
| F4 | **Quant mismatch: MLX 4bit-DWQ (~3.5bpw) vs GGUF Q4_K_M (~4.5bpw).** "MLX fastest+leanest" partly reflects a lighter quant. | YES (config) | L0 single-stream verdict | Add MLX vanilla-4bit row AND/or a GGUF quant near DWQ bpw (IQ4_XS); compare at matched bpw. |
| F5 | **gpt-oss/devstral/qwen2.5 scored on broken serving path** — not just tools: gpt-oss reasoning never engaged (no harmony), longctx 0.0 is a leak artifact. Violates "best config" contract. | YES (data: `<|channel|>` leak; name=None) | their tool + longctx + (gpt-oss) coding scores | Serve via llama.cpp --jinja (parses formats) / engage harmony, THEN re-score. |

## HIGH — bias a contender, fix before L2 conclusions

| id | issue | confirmed? | fix |
|----|-------|-----------|-----|
| F6 | **aider crippled**: 7/10 completions hit exactly 512 tokens (hard cap) + `--map-tokens 0` (no repo-map) + default `diff` format fails on weak local models → 0/10 is an artifact. | YES (data: 512 cap) | Remove token cap (raise max output), use `--edit-format whole` for weak model (or sweep), drop --map-tokens 0. |
| F7 | **L2 harnesses run in different cages**: opencode out=8192, crush out=4096, others default; instruction text differs (goose/crush/claude-code get extra "edit the file" scaffolding aider/opencode don't). Token-efficiency comparison invalid. | YES (code) | Pin identical context(32768)+max-output(8192) + IDENTICAL instruction text across all harnesses. |
| F8 | **llama.cpp -ub 512 undersizes Metal prefill** (Apple guidance -ub 2048). llama.cpp prefill measured below best. | YES (code) | -ub 2048 (or sweep), re-measure prefill. |
| F9 | **MLX context not pinned** to 34816 like GGUF engines → unequal context budget (affects memory + prefill comparison). | YES (code) | Pin MLX max context = 34816. |
| F10 | **decode_tps t_first = first CONTENT token** → reasoning/role-delta models get inflated decode_tps. | YES (code) | Anchor t_first to first delta of any kind; cross-check one engine vs llama-bench. |
| F11 | **L2 crush only 4/10 tasks** (incomplete) — can't compare pass-rate. | YES | Re-run L2 full (after F6/F7) so all harnesses cover same tasks. |

## MED

| id | issue | fix |
|----|-------|-----|
| F12 | Single-run evals, no variance; perf writes median only (PLAN promised spread). | Report min/median/max for perf; run tool/longctx ≥3×; note HumanEval+ n=1. |
| F13 | HumanEval+ extraction takes LAST code fence (wrong if prose-code-after); max_tokens=1024 truncates long solns. | Prefer fence containing entry_point; raise gen max_tokens. |
| F14 | "32k" needle haystack is only ~18.7k real tokens (token estimate /18 off ~40%). | Calibrate token estimate; relabel depths to actual tokens. |
| F15 | ollama OLLAMA_CONTEXT_LENGTH set via parent os.environ, propagation fragile/unverified. | Set explicitly in Ollama._env(); verify loaded ctx in log. |
| F16 | ccr shim overhead never measured separately (claude-code token count may be shim chattiness). | Measure identical request via ccr vs direct; attribute delta. |
| F17 | tool-calling temp=0 + tool_choice auto; greedy may handicap some models; 6 cases = coarse. | Decide greedy-vs-recommended (document); expand suite; add BFCL. |

## LOW / hygiene
- F18: results rows don't record the actual flags/quant/concurrency used → can't tell buggy vs fixed config from data. ADD config provenance to each row.
- F19: soak ran AFTER concurrency sweep (chip warm) → throttle understated; run soak cold.
- F20: GGUF engines KV-cache-type not pinned (f16 vs quant skews memory). Pin --cache-type-k/v.
- F21: proxy Connection:close + line-buffered SSE = minor latency/usage-parse fragility.

## Confirmed SOUND (audited, keep)
- Memory/wired GB (non-zero, scales with params+quant+ctx, under 56GB cap).
- Decode throughput (matches 273 GB/s bandwidth ceiling; no CPU fallback).
- Harness anchor: Qwen3-Coder HumanEval+ 0.90 ≈ published.
- L2 test-blind methodology (after the test-leak fix).
- Proxy uniform token tally (modulo F16 shim attribution).
