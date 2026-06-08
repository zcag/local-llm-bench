# Pitfalls — how local-LLM benchmarks silently lie (and how we caught ours)

This is the most useful page in the repo. Every issue below **was present in our own
first-pass results** and would have shipped as a confident finding. They were caught
by an independent adversarial audit + anchor checks. If you benchmark a local stack,
assume you have some of these right now.

Each one: the trap → how it shows up → how to catch it. Full status in
[`../FIXES.md`](../FIXES.md).

## 1. The engine wasn't actually configured to batch
**Trap:** we concluded *"mlx_lm.server can't do continuous batching"* — a headline
finding. **Reality:** we'd run *every* engine with concurrency disabled by default
(mlx `--decode-concurrency 1`, ollama `OLLAMA_NUM_PARALLEL=1`, llama.cpp auto
`--parallel 4`). The sweep measured under-configured engines, not their batching.
**Catch:** check the engine's startup log for the concurrency/slot count it actually
loaded with; if "doesn't scale" — did you enable scaling? (Corrected finding: MLX
*does* batch — its log shows 10 concurrent sequences — but its batched decode doesn't
raise aggregate throughput, while llama.cpp's scales 63→134 tok/s.)

## 2. TTFT was a prefix-cache hit, not a cold measurement
**Trap:** time-to-first-token reported **0.14 s** for an 18,669-token prompt — while
the same row's prefill rate (749 tok/s) implies **~25 s**. **Reality:** we fired 3
identical prefill reps; reps 2–3 hit the KV prefix cache → near-zero TTFT → the median
was the cached lie. **Catch:** TTFT must ≈ prompt_tokens ÷ prefill_tok/s. If it's
wildly faster, you're measuring a cache hit. **Fix:** unique prefix per request so the
cache can't hit.

## 3. Two different models scored *identically* (a timeout artifact)
**Trap:** a Mistral-architecture 24B and a Qwen 30B both scored exactly 0.823/0.793 on
HumanEval+; three quant levels all scored exactly 0.933/0.902. **Reality:** generating
at concurrency 4 against a slow model, requests queued past the client timeout →
empty completions → depressed, coincidentally-equal scores. **Catch:** identical
floats to 3 decimals across *different architectures* is a red flag; save generations
and confirm they differ. **Fix:** serial generation (an engine with no continuous-batch
gain has nothing to lose by it), and a known-good model must hit its published number.

## 4. The agent could read the test it was graded on
**Trap:** L2 copied the exercise's hidden test file into the working directory. Agentic
harnesses (which read/run files) could see the exact assertions; one-shot harnesses
couldn't — apples-to-oranges. **Catch:** would a tool that runs `cat *_test.py` get an
unfair edge? **Fix:** withhold the test until grading; every harness solves blind.

## 5. The quant comparison wasn't apples-to-apples
**Trap:** "MLX is fastest and leanest" — but MLX served 4-bit-DWQ (~3.5 bpw) while the
GGUF engines served Q4_K_M (~4.5 bpw). Part of the win was a lighter quant. **Catch:**
compare at matched bits-per-weight; report distillation-tuned quants separately. (After
matching: MLX still wins single-stream — so the finding survived, but only once it was
fair to check.)

## 6. EvalPlus's sandbox silently fails on macOS
**Trap:** every coding solution scored 0 — including known-correct ones. **Reality:**
EvalPlus's `setrlimit`-based execution sandbox isn't supported on macOS; every run
timed out. **Catch:** grade a *known-correct* solution first; if it fails, the harness
is broken, not the model. **Fix:** run the grader in a native arm64-Linux container
(OrbStack), not under emulation. Also: EvalPlus **caches** results — a stale result
file masks every retry.

## 7. Harnesses measured in different cages
**Trap:** "claude-code uses 179k tokens vs aider's 28k" — but they had different output
caps, context windows, and extra per-harness prompt scaffolding. **Catch:** before
comparing token efficiency, confirm identical context/output budget and identical
instruction text. Also: aider on its default `diff` edit-format silently no-ops on weak
local models — use `whole` and verify it isn't output-capped.

## The meta-lesson
None of these were model facts — they were **configuration and methodology** errors,
each of which produced a plausible, publishable, *wrong* number. The only defense that
worked was: (1) hold each contender at a config you can prove took effect, (2)
anchor-check every number against physics or a published value, and (3) get
**independent** eyes to try to break your findings. "It ran and produced a number" is
not the same as "the number is true."
