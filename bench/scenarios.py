"""Standardized, deterministic perf scenarios — identical across every engine.

Decode rate is measured on a prompt that reliably generates well past max_tokens
(no early EOS skewing the average); prefill on synthetic filler of known length
with a tiny output. temp=0 + fixed seed everywhere. `ignore_eos` is sent as an
extra field — engines that support it honor it, engines that don't ignore the
unknown key and the long prompt still guarantees enough tokens.
"""
from __future__ import annotations

# A paragraph ~ 60 tokens; repeat to hit target prompt lengths for prefill tests.
_FILLER = (
    "The quick brown fox jumps over the lazy dog while the network packet "
    "traverses each layer of the protocol stack, encapsulated and routed "
    "across heterogeneous links toward its eventual destination host. "
)

DECODE_PROMPT = (
    "Write a thorough, detailed technical explanation of how the TCP/IP "
    "networking stack works, going layer by layer from the physical layer up "
    "to the application layer. Explain encapsulation, addressing, routing, "
    "congestion control, and reliability. Write at least 900 words."
)

IGNORE_EOS = {"ignore_eos": True}


def filler_messages(approx_tokens: int) -> list[dict]:
    """A user message of roughly `approx_tokens` tokens, asking for a 16-tok reply."""
    reps = max(1, int(approx_tokens / 60))
    body = _FILLER * reps
    return [{"role": "user", "content":
             body + "\n\nReply with exactly: ACK"}]


def decode_messages() -> list[dict]:
    return [{"role": "user", "content": DECODE_PROMPT}]


# (label, messages, max_tokens, measures)
def perf_scenarios() -> list[tuple]:
    return [
        ("decode_256",   decode_messages(),       256, "decode"),
        ("prefill_4k",   filler_messages(4000),   16,  "prefill"),
        ("prefill_16k",  filler_messages(16000),  16,  "prefill"),
        ("prefill_32k",  filler_messages(32000),  16,  "prefill"),
    ]


CONCURRENCY_LEVELS = [1, 2, 4, 8, 16]
