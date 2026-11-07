# Local models inside real Claude Code (M5 Max, 36 GB, November 2026)

The question: can a model that fits a 36 GB MacBook Pro (M5 Max, 32-core GPU) stand in for a hosted model inside
**unmodified Claude Code**, with its own prompt, tools and MCP servers? Short answer: **no**. The best setup
passed 39% of hidden tests and finished 3 of 10 small tasks, at about 13 minutes per task.

## Setup

- Server: [Rapid-MLX](https://github.com/raullenchai/Rapid-MLX) 1.0.1, Anthropic `/v1/messages`, one model loaded at a time.
- Harness: Claude Code 2.1.319 with the user's normal global config and every MCP tool loaded up front.
- Tasks: the same 10 Aider polyglot Python exercises as L2 (`bench/l2_tasks.py`), test-blind, the L2 prompt suffix,
  15 minutes each, graded by the withheld tests. Plus one MCP task: four tools in order (two MCP servers, Bash,
  Read), passed if all four are called and the reply carries the real commit hash.
- Runner: [`scripts/run_cap.py`](../scripts/run_cap.py). Raw rows: [`../results/claude-code-m5max.jsonl`](../results/claude-code-m5max.jsonl).

## Results

"Tests" counts withheld test cases passed out of 181 across the 10 tasks, which separates runs better than whole
tasks do.

| Rank | Model and setting | Tests | Tasks | Timeouts | MCP task |
|---|---|---|---|---|---|
| 1 | Qwen3.8-27B 4-bit (MTP), defaults | **39%** | 3/10 | 4 | pass |
| 2 | Qwen3.8-27B, `--effort low` | about 35% | 2/9 | 4 | pass |
| 3 | Qwen3.8-27B, `--effort medium`, web tools denied | 32% | 2/10 | 3 | pass |
| 4 | Qwen3.8-27B, `--setting-sources project` (no global CLAUDE.md) | 30% | 1/10 | 4 | fail (MCP not loaded) |
| 5 | Qwen3.8-27B, coding guidance via `--append-system-prompt` | 29% | 2/10 | 6 | pass |
| 5 | Qwen3-Coder-30B-A3B 4-bit DWQ | 29% | 2/10 | 6 | pass |
| 7 | Gemma 4 26B-A4B 4-bit | 21% | 0/10 | 2 | pass |
| 8 | Flint 40B-A2.5B 4-bit | 18% | 0/10 | 1 | fail |

The low-effort run was stopped before its last task; that task (`forth`) failed all 54 tests in every other run,
so the 35% counts it as failed.

What the failures were:
- **Flint** calls Claude Code's tools with names and arguments from another harness (`path` for `file_path`,
  `bash` for `Bash`) and loops on the errors. It decodes at about 80 tok/s. The score measures the mismatch, not
  the model.
- **Gemma 4** often pasted code into its reply instead of editing the file, and reused little of the prompt cache.
- **Qwen3-Coder-30B** re-read most of its prompt each step (about 90 s per step) and mostly ran out of time.
- **Qwen3.8** sometimes said its work passed when the hidden tests failed.
- Which tasks pass changes from run to run (`affine-cipher` passed, passed, timed out across three Qwen3.8 runs),
  so differences of one task are noise.

## Where the time goes

In the default Qwen3.8 run, 136K output tokens at a median 25.8 tok/s account for about 88 of 103 minutes. About
70% of the output is reasoning; capping it with lower effort moved much of that into reply text. Re-reading the
prompt was the smaller cost once the settings below were in.

Decode speed at about 36K context stayed at 26 to 35 tok/s across MTP depth 2 to 6, `--kv-cache-turboquant k8v4`,
`--stream-interval 16`, `--no-gc-control` and temperature 0 to 1.0. A 4K prompt gave 30 to 38 tok/s. Nothing tried
moved it beyond run-to-run noise.

## Settings that did matter (prompt cache, not score)

| Setting | Effect |
|---|---|
| `ENABLE_TOOL_SEARCH=false` | With on-demand loading the model called `ToolSearch` repeatedly and never the tool. Up front, the MCP task passed first try |
| `claude --exclude-dynamic-system-prompt-sections` | Two folders' prompts share 92% instead of about 1%. New folder: 128 s instead of 332 s |
| `RAPID_MLX_PREFIX_CACHE_SHUTDOWN_BUDGET=60` | The default 3.5 s silently skipped saving a large cache entry. With it, a prompt after an idle unload took 7 s, after a restart 2 s |
| `ANTHROPIC_DEFAULT_HAIKU_MODEL=not-served` | Background sessions always run an end-of-turn classifier on the haiku alias. Served locally it evicted the session's cache (next message about 6 min). A 404 avoids that |
| `CLAUDE_CODE_MAX_OUTPUT_TOKENS=16384` | The server reserves memory for the whole possible reply |
| `CLAUDE_CODE_ATTRIBUTION_HEADER=0` | Already known from L2: a per-session header otherwise busts the cache |

Memory: 80K-token prompts fit at the 27.1 GB Metal cap (27.2 GB peak), 103K ran out. Disabling MTP cost 37%
decode (39 to 24.5 tok/s on short prompts).

## Traps

- **One model at a time, and only after the last one has exited.** A stopped server sat for minutes in an
  uninterruptible state still holding 24 GB; starting a 15 GB server meanwhile swapped the Mac until it had to be
  rebooted. `run_cap.py` now refuses to start while another server holds memory.
- `ps` misses Metal memory: use `top -l 1 -stats command,mem`.
- `--max-prompt-tokens` rejected every request for Gemma 4, whose lane cannot count prompt tokens.
- A server's automatic KV estimate can refuse real prompts (Qwen3-Coder-30B); pass a measured
  `--metal-cap-kv-bytes-per-token` (140000 for it).
- If a session sends its prompt the moment it opens, MCP servers may not be connected yet; the tool list then
  changes on the next request and the prompt is re-read.

## Not tried

Ornith 1.5 35B-A3B, Nanbeige 4.3 33B, a Qwen3.8 coding fine-tune, High Power Mode, longer time limits. Given where
the time goes, none was likely to change the verdict.
