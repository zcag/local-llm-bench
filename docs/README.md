# Docs

Specialized docs for the local-LLM-stack benchmark. The narrative writeup lives on
the [tela blog](#) *(link TBD)*; this is the detailed-data + methods home it links to.

## Pages
- **[methodology.md](methodology.md)** — the four-layer design, scenarios, grading,
  and the verified-config gate. Start here.
- **[pitfalls.md](pitfalls.md)** — *the most useful page.* The fairness traps that
  silently corrupt local-LLM benchmarks, the ones we fell into, and how each was
  caught and fixed. If you only read one page, read this.
- **[reproduce.md](reproduce.md)** — run it on your own Apple Silicon box.
- Per-layer detail lives in [`../FINDINGS.md`](../FINDINGS.md) (results),
  [`../CONFIG.md`](../CONFIG.md) (the config ledger), and [`../FIXES.md`](../FIXES.md)
  (audit findings + fix status).

## Raw data
- [`../results/runs.jsonl`](../results/runs.jsonl) — one JSON object per measurement
  (engine/model/harness, scenario, metrics, **the exact config used**, evidence).
- [`../results/l1_evals/`](../results/l1_evals/) — per-model eval detail + saved
  generations (so coding-score independence is auditable).

## The contract (why this is different)
From [`../PLAN.md`](../PLAN.md): **breadth is the point — no contender is dropped;**
each is configured to its best and **verified with evidence** (a log line, a probe);
a contender that "doesn't work" is fixed or proven a wall, never silently cut; only
*depth* (task counts, repeats) is economized. Nothing counts as a result until its
config is verified and the number passes an anchor check (matches physics or a
published value).
