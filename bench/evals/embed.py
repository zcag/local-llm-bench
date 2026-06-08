"""L3 embed/RAG eval for qwen3-embedding:0.6b (the live embed model on :11435).
Measures (a) throughput: embeddings/sec on a batch; (b) retrieval quality:
recall@1 and @3 on a hand-built corpus where each query has a known-relevant doc.
Hits ollama's /api/embed. Deterministic, auto-graded.
"""
from __future__ import annotations
import time
import math
import httpx

EMBED_URL = "http://127.0.0.1:11435/api/embed"
MODEL = "qwen3-embedding:0.6b"

# Topic-clustered corpus; each query's known-relevant doc id is the gold label.
CORPUS = {
    "d_tcp": "TCP provides reliable ordered byte-stream delivery using sequence numbers, acknowledgements and retransmission.",
    "d_udp": "UDP is a connectionless datagram protocol with no delivery guarantees, used for low-latency traffic like DNS and video.",
    "d_bgp": "BGP is the path-vector routing protocol that exchanges reachability information between autonomous systems on the internet.",
    "d_photosynth": "Photosynthesis converts carbon dioxide and water into glucose and oxygen using light energy in chloroplasts.",
    "d_mitosis": "Mitosis is cell division producing two genetically identical daughter cells, with phases prophase through telophase.",
    "d_enzyme": "Enzymes are protein catalysts that lower the activation energy of biochemical reactions and are highly substrate-specific.",
    "d_inflation": "Inflation is the rate at which the general price level of goods and services rises, eroding purchasing power over time.",
    "d_interest": "A central bank raises interest rates to cool an overheating economy and curb inflation by making borrowing costlier.",
    "d_bond": "A bond is a fixed-income instrument representing a loan from an investor to a borrower, paying periodic coupons.",
    "d_quicksort": "Quicksort is a divide-and-conquer sorting algorithm with average O(n log n) time, partitioning around a pivot.",
    "d_hashmap": "A hash map stores key-value pairs and offers average O(1) lookups by hashing keys into buckets.",
    "d_btree": "A B-tree is a self-balancing tree keeping data sorted for logarithmic search, used in databases and filesystems.",
    "d_rome": "The Roman Empire reached its greatest extent under Trajan in 117 AD, spanning Britain to Mesopotamia.",
    "d_renaissance": "The Renaissance was a cultural movement from the 14th to 17th century reviving classical art, science and humanism.",
    "d_frenchrev": "The French Revolution began in 1789, overthrowing the monarchy and proclaiming liberty, equality and fraternity.",
    "d_everest": "Mount Everest, on the Nepal-China border, is Earth's highest peak above sea level at 8,849 metres.",
    "d_amazon": "The Amazon rainforest is the world's largest tropical rainforest, hosting unmatched biodiversity across the basin.",
    "d_sahara": "The Sahara is the largest hot desert on Earth, covering much of North Africa with extreme aridity.",
    "d_caffeine": "Caffeine is a stimulant that blocks adenosine receptors, increasing alertness and reducing perceived fatigue.",
    "d_vitc": "Vitamin C is a water-soluble antioxidant essential for collagen synthesis; deficiency causes scurvy.",
}

QUERIES = [
    ("How does reliable in-order data transfer work on the internet?", "d_tcp"),
    ("Which protocol routes between autonomous systems?", "d_bgp"),
    ("What connectionless protocol is used for DNS and streaming?", "d_udp"),
    ("How do plants make food from sunlight?", "d_photosynth"),
    ("What process splits a cell into two identical cells?", "d_mitosis"),
    ("Why does a central bank hike rates?", "d_interest"),
    ("What erodes the purchasing power of money over time?", "d_inflation"),
    ("Fast average-case comparison sort using a pivot?", "d_quicksort"),
    ("Data structure giving O(1) key lookups?", "d_hashmap"),
    ("When did the French monarchy get overthrown?", "d_frenchrev"),
    ("What is the tallest mountain on Earth?", "d_everest"),
    ("What stimulant blocks adenosine to keep you awake?", "d_caffeine"),
]


def _cos(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    na = math.sqrt(sum(x * x for x in a)); nb = math.sqrt(sum(y * y for y in b))
    return dot / (na * nb) if na and nb else 0.0


def _embed(client, texts):
    r = client.post(EMBED_URL, json={"model": MODEL, "input": texts}, timeout=120)
    r.raise_for_status()
    return r.json()["embeddings"]


def run() -> dict:
    ids = list(CORPUS)
    docs = [CORPUS[i] for i in ids]
    with httpx.Client() as client:
        # throughput: embed the whole corpus, timed
        t0 = time.perf_counter()
        dvecs = _embed(client, docs)
        dt = time.perf_counter() - t0
        eps = len(docs) / dt if dt else 0.0
        qvecs = _embed(client, [q for q, _ in QUERIES])

    r1 = r3 = 0
    cases = []
    for (q, gold), qv in zip(QUERIES, qvecs):
        sims = sorted(((_cos(qv, dv), i) for dv, i in zip(dvecs, ids)), reverse=True)
        top = [i for _, i in sims[:3]]
        hit1, hit3 = (top[0] == gold), (gold in top)
        r1 += hit1; r3 += hit3
        cases.append({"q": q, "gold": gold, "top3": top, "hit@1": hit1, "hit@3": hit3})
    n = len(QUERIES)
    return {"eval": "embed_rag", "model": MODEL, "dim": len(dvecs[0]) if dvecs else 0,
            "embeds_per_sec": round(eps, 1), "recall@1": round(r1 / n, 3), "recall@3": round(r3 / n, 3),
            "n_docs": len(docs), "n_queries": n, "cases": cases}


if __name__ == "__main__":
    import json
    print(json.dumps({k: v for k, v in run().items() if k != "cases"}, indent=2))
