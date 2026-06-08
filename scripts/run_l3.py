"""L3 — embed/RAG retrieval + throughput. Needs the embed server (:11435) UP, so
run when NOT quiesced (after a chat-engine run). Writes one L3 row.
Run ON tardis: .venv/bin/python scripts/run_l3.py
"""
from __future__ import annotations
import sys
sys.path.insert(0, __file__.rsplit("/scripts/", 1)[0])
from bench.evals import embed
from bench.results import write

if __name__ == "__main__":
    res = embed.run()
    write({"layer": "L3", **{k: v for k, v in res.items() if k != "cases"}})
    print("embed/RAG:", {k: v for k, v in res.items()
                         if k in ("embeds_per_sec", "recall@1", "recall@3", "dim", "n_docs", "n_queries")})
