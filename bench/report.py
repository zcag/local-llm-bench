"""Tabulate results/runs.jsonl into readable tables. `uv run python -m bench.report`."""
from __future__ import annotations
import json
import os
import sys

RESULTS = os.path.join(os.path.dirname(__file__), "..", "results", "runs.jsonl")


def load(path=RESULTS):
    if not os.path.exists(path):
        return []
    return [json.loads(l) for l in open(path) if l.strip()]


def l0(rows):
    perf = [r for r in rows if r.get("layer") == "L0" and r.get("kind") in ("decode", "prefill")]
    conc = [r for r in rows if r.get("layer") == "L0" and r.get("scenario") == "concurrency"]
    soak = [r for r in rows if r.get("layer") == "L0" and r.get("scenario") == "soak"]

    print("\n=== L0 single-stream perf (concurrency 1) ===")
    print(f"{'engine':10} {'scenario':12} {'ttft_s':>7} {'decode_t/s':>10} {'prefill_t/s':>11} "
          f"{'wired_GB':>8} {'ok/fail':>7}")
    for r in perf:
        print(f"{r['engine']:10} {r['scenario']:12} {r.get('ttft_p50',0):7.2f} "
              f"{r.get('decode_tps',0):10.1f} {r.get('prefill_tps',0):11.0f} "
              f"{r.get('mem_wired_gb',0):8.1f} {r['ok']}/{r['fail']}")

    print("\n=== L0 concurrency (system throughput) ===")
    print(f"{'engine':10} {'c':>3} {'system_t/s':>10} {'ttft_p50':>8} {'ttft_p90':>8} {'ok/fail':>7}")
    for r in conc:
        print(f"{r['engine']:10} {r['concurrency']:3} {r.get('system_tps',0):10.1f} "
              f"{r.get('ttft_p50',0):8.2f} {r.get('ttft_p90',0):8.2f} {r['ok']}/{r['fail']}")

    if soak:
        print("\n=== L0 soak / thermal throttle ===")
        print(f"{'engine':10} {'first_t/s':>9} {'last_t/s':>9} {'throttle%':>9} {'n':>4}")
        for r in soak:
            print(f"{r['engine']:10} {r.get('decode_tps_first_min',0):9.1f} "
                  f"{r.get('decode_tps_last_min',0):9.1f} {r.get('throttle_pct',0):9.1f} {r.get('n',0):4}")


def l1(rows):
    perf = {}
    evals = {}
    for r in rows:
        if r.get("layer") != "L1":
            continue
        key = r.get("model_tag") or r.get("model")
        if r.get("kind") == "decode":
            perf.setdefault(key, {})["decode"] = r.get("decode_tps", 0)
            perf.setdefault(key, {})["wired"] = r.get("mem_wired_gb", 0)
        if r.get("eval"):
            evals.setdefault(key, {})[r["eval"]] = r.get("score", 0)
    if not perf and not evals:
        return
    print("\n=== L1 model x quant ===")
    keys = sorted(set(perf) | set(evals))
    allev = sorted({e for v in evals.values() for e in v})
    hdr = f"{'model':40} {'decode_t/s':>10} {'wired_GB':>8} " + " ".join(f"{e:>12}" for e in allev)
    print(hdr)
    for k in keys:
        p = perf.get(k, {})
        ev = evals.get(k, {})
        line = f"{k:40} {p.get('decode',0):10.1f} {p.get('wired',0):8.1f} " + \
               " ".join(f"{ev.get(e,0):12.3f}" for e in allev)
        print(line)


if __name__ == "__main__":
    rows = load(sys.argv[1] if len(sys.argv) > 1 else RESULTS)
    print(f"{len(rows)} rows")
    l0(rows)
    l1(rows)
