# Config ledger — fair-comparison contract

Every contender must run at its BEST config, and that must be VERIFIED with
evidence (a log line, a probe), not assumed. A result is invalid until its row
here is `verified`. Source = where the optimal setting is established (docs/model
card/engine help), not a guess.

Legend: status = todo | applied | **verified** | wall(proven)

## Engines (L0)
| engine | best-config target | source | applied flags | verified-by / evidence | status |
|---|---|---|---|---|---|
| mlx_lm | all-GPU (inherent), correct chat template | MLX is Metal-only | default + --model | decode 90 t/s = GPU; template from tokenizer | **verified** |
| llama.cpp | full offload, FA on, **--parallel ≥ concurrency**, big batch | llama-server --help | -ngl 999 --flash-attn on -c 34816 --jinja | 70.8 t/s ⇒ full GPU; **BUT n_parallel=4 auto ⇒ concurrency c8/c16 capped → RE-RUN with --parallel 16 -b 2048 -ub 512** | todo (perf ok, concurrency unfair) |
| ollama-GGUF | full offload, FA on, num_ctx | ollama env | OLLAMA_FLASH_ATTENTION=1, CTX=34816, KEEP_ALIVE=-1 | log: GPULayers:49(0..48), FA Enabled | **verified** |
| ollama-MLX | (investigate: may only run registry models) | memory note | — | — | todo |
| LM Studio | GPU max, FA, context | lms load flags | --gpu max --context-length 34816 | needs verify (offload + FA in lms log) | todo |
| MLC-LLM | Metal, full offload | mlc docs | — | — | todo |
| spec-decode | mlx --draft-model on DWQ-30B | mlx_lm docs / memory | — | — | todo |

## Models (L1)
| model | best-config target | source | applied | verified-by | status |
|---|---|---|---|---|---|
| Qwen3-Coder family | own chat template + tool template; temp 0 for repro | Qwen model card | temp0 seed7, mlx template | tool 6/6 + 0.90 HE ⇒ template+tools work | **verified** |
| gpt-oss-20b | harmony format + reasoning channel + tool parse | gpt-oss card | NOT engaged via mlx_lm OpenAI ep | raw `<|channel|>analysis` leaked ⇒ harmony NOT parsed → must serve via an engine that parses harmony | todo (mis-served) |
| devstral/qwen2.5 | Mistral/Hermes tool template parsed to tool_calls | model cards | not parsed by mlx_lm | tool 1/6 ⇒ re-serve via llama.cpp --jinja | todo (mis-served) |
| sampling | each model's recommended decode params | model cards | temp0 greedy (repro choice) | — document tradeoff | todo (decide: greedy vs recommended) |

## Harnesses (L2)
| harness | best-config target | source | applied | verified-by | status |
|---|---|---|---|---|---|
| aider | best edit-format for model, NO artificial token cap | aider docs | default --message; **saw completion cap ~512 → investigate/raise** | 512-cap suspected unfair → verify + fix | todo (suspected cap) |
| opencode | model needs tool_call:true + limit | Cagdas's lcld config | added tool_call:true, limit 32768/8192 | needs validate (>0 reqs + edits) | todo |
| goose | openai provider, tools enabled | goose docs | env provider | made 4 reqs + tool use ⇒ works | applied (verify best) |
| crush | openai provider, yolo, max tokens | crush docs | --quiet CRUSH_YOLO | 16 reqs + edits ⇒ works | applied (verify best) |
| claude-code | via ccr shim, edits auto-approved | ccr docs | ANTHROPIC_BASE_URL=ccr, --dangerously-skip-permissions | PASS affine ⇒ works (token-heavy) | applied (verify best) |
| cline/continue/roo | headless driver | — | — | — | todo (attempt/prove wall) |

## Shared / fairness invariants
- temp=0, seed=7, identical scenarios, identical context window across a comparison.
- L2: every harness gets spec+stub ONLY (test withheld until grading) — **verified** (l2_tasks).
- L2 token accounting via proxy (uniform) — verified.
- Each run records its exact config (TODO: dump config into each results row).
