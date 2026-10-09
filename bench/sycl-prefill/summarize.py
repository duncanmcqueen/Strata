#!/usr/bin/env python3
"""Aggregate a validated prefill JSONL into per-label statistics.

Usage: summarize.py results.jsonl
Only records with valid=true and a usable prefill timing enter an aggregate.  Failed/truncated runs are listed
separately and never averaged.  Reports run count, median, min and max for the primary metric per label, plus
the exact configuration identity carried by the records.
"""
import json
import statistics
import sys
from collections import defaultdict


def metric(d):
    # final prefill is the whole-workload metric; fall back to the inner summary only when the final is absent.
    if "final_prefill_ms" in d:
        return d["final_prefill_ms"], "final_prefill_ms"
    if "inner_prefill_ms" in d:
        return d["inner_prefill_ms"], "inner_prefill_ms"
    return None, None


def main():
    path = sys.argv[1]
    rows = []
    for line in open(path, errors="replace"):
        line = line.strip()
        if not line:
            continue
        try:
            rows.append(json.loads(line))
        except json.JSONDecodeError:
            rows.append({"valid": False, "invalid_reason": "unparseable json line", "raw": line})
    good = [r for r in rows if r.get("valid")]
    bad = [r for r in rows if not r.get("valid")]
    print(f"records: {len(rows)}  valid: {len(good)}  rejected: {len(bad)}")
    for r in bad:
        print(f"  REJECTED {r.get('label','?')}: {r.get('invalid_reason','?')} ({r.get('log','?')})")
    by = defaultdict(list)
    for r in good:
        v, key = metric(r)
        if v is not None:
            by[r.get("label", "?")].append((v, key, r))
    print("\nlabel                              metric            n    median       min       max")
    for label, vals in sorted(by.items()):
        xs = [v for v, _, _ in vals]
        _, key, _ = vals[0]
        print(f"{label:34s} {key:16s} {len(xs):3d}  {statistics.median(xs):9.1f} "
              f"{min(xs):9.1f} {max(xs):9.1f}")
    # configuration identity summary (from the first record of each distinct tuple)
    print("\nconfiguration identities:")
    seen = set()
    for r in good:
        k = (r.get("chunk_arg"), r.get("chunk_chosen"), r.get("cache_slots"), r.get("borrowed_slots"),
             r.get("resident_mode"))
        if k in seen:
            continue
        seen.add(k)
        print(f"  chunk_arg={r.get('chunk_arg')} chosen={r.get('chunk_chosen')} "
              f"slots={r.get('cache_slots')} borrowed={r.get('borrowed_slots')} resident={r.get('resident_mode')}")


if __name__ == "__main__":
    main()
