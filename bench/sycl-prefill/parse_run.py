#!/usr/bin/env python3
"""Parse a `strata generate` prefill log into one JSON object.

Usage: parse_run.py LOGFILE
Prints a single JSON line.  Fields:
  inner_prefill_ms / inner_tok_s   the engine's own "prefill N tokens in M chunks" line
  final_prefill_ms / final_tok_s / ttft_ms   the final "prefill ... -> ... tok/s" summary
  phases{name:{ms,pct}}            STRATA_PREFILL_TIMING phase table
"""
import json
import re
import sys

PHASES = [
    ("embed_steps", "embed+steps"), ("hc_read", "hc read"), ("gdn", "gdn"),
    ("qsa_proj", "qsa proj"), ("qsa_indexer", "qsa indexer"), ("qsa_score", "qsa score"),
    ("qsa_topk", "qsa topk"), ("qsa_select", "qsa select"),
    ("qsa_attn", "qsa attn"), ("router_shared", "router+shared"), ("host_grouping", "host grouping"),
    ("gather", "gather"), ("wait_copy", "wait copy"), ("dequant", "dequant"),
    ("gemm_gu", "gemm gate/up"), ("gemm_d", "gemm down"), ("combine", "combine"),
    ("ple", "ple"), ("gdn_conv", "gdn conv+gates"), ("gdn_rec", "gdn recurrence"),
    ("gdn_out", "gdn out proj"),
]


def parse(path):
    """Parse LOGFILE text into one dict.  Kept as a function so the aggregator can validate records."""
    out = {"log": path}
    txt = open(path, errors="replace").read()
    return parse_text(path, txt)


def parse_text(path, txt):
    out = {"log": path}
    m = re.search(r"prompt chunk auto: (\d+) tokens", txt)
    if m:
        out["chunk_chosen"] = int(m.group(1))
    # inner (engine) prefill summary
    m = re.search(r"prefill (\d+) tokens in (\d+) chunks, ([\d.]+) ms \(([\d.]+) tok/s\)"
                  r"(?:.*?experts streamed (\d+) \((\d+) by DMA, host ([\d.]+) ms\))?"
                  r"(?:.*?resident (\d+))?(?:.*?PLE ([\d.]+) ms)?", txt)
    if m:
        out["tokens"] = int(m.group(1))
        out["chunks"] = int(m.group(2))
        out["inner_prefill_ms"] = float(m.group(3))
        out["inner_tok_s"] = float(m.group(4))
        if m.group(5):
            out["experts_streamed"] = int(m.group(5))
            out["experts_dma"] = int(m.group(6))
            out["host_ms"] = float(m.group(7))
        if m.group(8):
            out["experts_resident"] = int(m.group(8))
        if m.group(9):
            out["ple_ms"] = float(m.group(9))
    # final prefill summary (primary metric)
    m = re.search(r"prefill\s+(\d+) tokens in ([\d.]+) ms\s*->\s*([\d.]+) tok/s"
                  r"\s*\(time to first token ([\d.]+) ms\)", txt)
    if m:
        out["final_tokens"] = int(m.group(1))
        out["final_prefill_ms"] = float(m.group(2))
        out["final_tok_s"] = float(m.group(3))
        out["ttft_ms"] = float(m.group(4))
    m = re.search(r"decode\s+(\d+) tokens in ([\d.]+) ms\s*->\s*([\d.]+) tok/s", txt)
    if m:
        out["decode_tokens"] = int(m.group(1))
        out["decode_ms"] = float(m.group(2))
        out["decode_tok_s"] = float(m.group(3))
    # phase table; historical logs carry the combined "qsa select" phase, new logs carry score + top-k.
    m = re.search(r"prefill timing:.*?GPU timeline (\d+) ms, wall (\d+) ms, host staging (\d+) ms: (.*)$",
                  txt, re.M)
    if m:
        out["gpu_timeline_ms"] = int(m.group(1))
        out["timing_wall_ms"] = int(m.group(2))
        out["host_staging_ms"] = int(m.group(3))
        ph = {}
        for name, label in PHASES:
            pm = re.search(re.escape(label) + r" (\d+) \(([\d.]+)%\)", m.group(4))
            if pm:
                ph[name] = {"ms": int(pm.group(1)), "pct": float(pm.group(2))}
        out["phases"] = ph
    # metadata
    for key, pat, cast in (
        ("cache_slots", r"expert cache (\d+) slots", int),
        ("borrowed_slots", r"prompt path borrows (\d+) cache slots", int),
        ("resident_mode", r"resident RAM mode: ([^\n]+)", str),
        ("pcie_probe", r"PCIe probe: ([^\n]+)", str),
        ("cache_gib", r"expert cache auto: ([\d.]+) GiB free", float),
    ):
        mm = re.search(pat, txt)
        if mm:
            out[key] = cast(mm.group(1))
    if "final_prefill_ms" not in out and "inner_prefill_ms" not in out:
        out["error"] = "no prefill summary found"
    return out


def main():
    path = sys.argv[1]
    print(json.dumps(parse(path)))


if __name__ == "__main__":
    main()
