"""Validate the measurement path against the LIVE MLX server (non-destructive).

Hits :11434 at concurrency 1: a short-output decode test and a long-prompt
prefill test. Confirms usage reporting, TTFT, decode/prefill tok/s, mem sampling.
Run ON tardis:  uv run python scripts/smoke.py
"""
from __future__ import annotations
import asyncio
import sys
import httpx
from rich.console import Console
from rich.table import Table

sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from bench import config
from bench.metrics import Sampler
from bench.load.runner import run_level, warmup

console = Console()
BASE = config.base_url(config.MLX_PORT)


async def served_model() -> str:
    async with httpx.AsyncClient() as c:
        r = await c.get(f"{BASE}/models", timeout=20)
        r.raise_for_status()
        return r.json()["data"][0]["id"]


LONG_PROMPT = (
    "Below is a Python module. Summarize what each function does in one line.\n\n"
    + ("def f%d(x):\n    return x * %d + sum(range(%d))\n\n" % (0, 1, 10)) * 200
)


async def main():
    model = await served_model()
    console.print(f"[bold]Served model:[/bold] {model}")

    console.print("warmup…")
    await warmup(BASE, model, n=2)

    scenarios = [
        ("decode (short prompt, 256 out)",
         [{"role": "user", "content": "Write a Python function to check if a string is a palindrome. Just the code."}],
         256),
        ("prefill (long prompt, 32 out)",
         [{"role": "user", "content": LONG_PROMPT}],
         32),
    ]

    table = Table(title="MLX live smoke @ concurrency 1", show_lines=True)
    for col in ("scenario", "ok/fail", "ttft p50 (s)", "decode tok/s", "prefill tok/s",
                "prompt tok", "out tok", "mem used GB", "wired GB"):
        table.add_column(col)

    for name, msgs, mx in scenarios:
        with Sampler(proc_match="mlx_lm.server") as s:
            lvl = await run_level(BASE, model, msgs, concurrency=1, n=3, max_tokens=mx)
        ex = lvl.samples[0] if lvl.samples else {}
        table.add_row(
            name, f"{lvl.ok}/{lvl.fail}",
            f"{lvl.ttft_p50:.3f}", f"{lvl.decode_tps_median:.1f}",
            f"{ex.get('prefill_tps', 0):.0f}",
            str(ex.get("prompt_tokens", 0)), str(ex.get("completion_tokens", 0)),
            f"{s.stats.used_peak_gb:.1f}", f"{s.stats.wired_peak_gb:.1f}",
        )
        if lvl.errors:
            console.print(f"[red]errors:[/red] {lvl.errors}")

    console.print(table)
    console.print("[dim]usage reported by server[/dim]" if ex.get("prompt_tokens") else
                  "[yellow]no usage from server — will need local tokenizer counting[/yellow]")


if __name__ == "__main__":
    asyncio.run(main())
