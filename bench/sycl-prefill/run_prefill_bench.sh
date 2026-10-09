#!/bin/bash
# Benchmark runner for the Intel SYCL prefill path (Arc Pro B70).
#
# Usage: run_prefill_bench.sh <tokens-file> <chunk> [extra engine args...]
#   chunk: auto | auto:16384 | auto:32768 | <N>
# Env passthrough: STRATA_PREFILL_TIMING, STRATA_SYCL_QSTATS, STRATA_PREFILL_RING,
#   STRATA_PREFILL_ISSUER, STRATA_PREFILL_AUTO_MAX, STRATA_RESIDENT_GIB (budget),
#   ONEAPI_DEVICE_SELECTOR, etc.  Output goes to stdout; the caller redirects.
set -u
ROOT=/home/dwmcqueen/Strata
BIN="${STRATA_BENCH_ENGINE:-$ROOT/engine/strata}"
TOK="${1:?tokens-file}"; CHUNK="${2:?chunk}"; shift 2
G=/home/dwmcqueen/models/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/IQ1_M
export LD_LIBRARY_PATH="$(dirname "$BIN"):$ROOT/engine:/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/mkl/latest/lib:${LD_LIBRARY_PATH:-}"
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 ZES_ENABLE_SYSMAN=1 NEO_FP64_EMULATION=1
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:0}"
CTX="${MAX_CONTEXT:-65536}"
MAXNEW="${MAX_NEW:-8}"
BUDGET_ARGS=""
if [ -n "${STRATA_RESIDENT_GIB:-}" ]; then BUDGET_ARGS="--resident-budget-gib ${STRATA_RESIDENT_GIB}"; fi
exec "$BIN" \
  --pack /home/dwmcqueen/Strata-data/packs/coder-iq1_m \
  --native "$G/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf" \
  --ple-gguf "$G/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf" \
  --expert-profile "$ROOT/data/expert-profile-coder.bin" \
  --expert-cache auto --prefill "$CHUNK" --spec 4 --spec-min-p 0.5 \
  --mtp /home/dwmcqueen/Strata-data/mtp/rt --max-context "$CTX" --spec-split --kv int8 \
  --resident-experts $BUDGET_ARGS --max-new "$MAXNEW" --tokens-file "$TOK" "$@"
