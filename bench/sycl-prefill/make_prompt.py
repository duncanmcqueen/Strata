#!/usr/bin/env python3
"""Build a non-repeated prefill prompt from real repository text.

The provided /tmp prompt files are repeated 1,526-token blocks, which distort routing
and MTP acceptance.  This encodes distinct text (docs + engine sources) with the
model's own tokenizer (tools/strata_tokenizer.py) and writes comma-separated ids.

Usage: make_prompt.py <out.ids> <n_tokens> [files...]
"""
import pathlib
import sys

ROOT = pathlib.Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "tools"))
from strata_tokenizer import Tokenizer  # noqa: E402

GGUF = ("/home/dwmcqueen/models/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/IQ1_M/"
        "Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf")

DEFAULT_FILES = [
    "docs/DETAILS.md", "docs/INTEL_SYCL.md", "docs/AMD_HIP.md", "docs/INSTALL.md",
    "src/prefill/prefill.cpp", "src/sycl_runtime/blas.cpp", "src/program/generate.cpp",
    "src/core/expert_source.cpp", "src/kernels/sycl/iq_kernels.cpp", "include/strata/prefill/moe_mmq.hpp",
    "src/prefill/moe_mmq.cu", "src/prefill/sycl/kernels.cpp", "CMakeLists.txt", "README.md",
]


def main():
    out = pathlib.Path(sys.argv[1])
    want = int(sys.argv[2])
    files = sys.argv[3:] or DEFAULT_FILES
    tok = Tokenizer.from_gguf(GGUF)
    ids: list[int] = []
    for f in files:
        p = ROOT / f
        if not p.exists():
            continue
        text = p.read_text(errors="replace")
        ids.extend(tok.encode(text, parse_special=False))
        if len(ids) >= want:
            break
    ids = ids[:want]
    out.write_text(",".join(str(i) for i in ids))
    uniq = len(set(ids))
    print(f"{out}: {len(ids)} tokens, {uniq} unique ({100.0 * uniq / len(ids):.1f}%)")


if __name__ == "__main__":
    main()
