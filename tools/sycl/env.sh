# Source this in every SYCL build/test shell:  . tools/sycl/env.sh
# This machine's oneAPI install has no top-level setvars.sh, so set the compiler
# and oneMKL paths directly. ONEAPI_ROOT override: ONEAPI_ROOT=/elsewhere . tools/sycl/env.sh
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
export PATH="${ONEAPI_ROOT}/compiler/latest/bin:${PATH}"
export LD_LIBRARY_PATH="${ONEAPI_ROOT}/compiler/latest/lib:${ONEAPI_ROOT}/mkl/latest/lib:${LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="${ONEAPI_ROOT}/mkl/latest/lib/cmake/mkl:${CMAKE_PREFIX_PATH}"

# The inference card is the DISCRETE Arc.  The Level Zero order is not OpenVINO's
# GPU.0/GPU.1 and differs per machine, so do not hardcode an index:
#   dev box:  level_zero:0 Arc B580 (display), level_zero:1 Arc A770 (the test card)
#   B70 box:  level_zero:0 Arc Pro B70 (the test card), level_zero:1 integrated Arc Graphics
# Pick by name: a discrete Arc carries a model (A380/A750/A770/B580/B60/B70/Pro B70...);
# the integrated "Arc(TM) Graphics" has none and stays out of test runs.  With more than
# one discrete Arc, the historical A770 default (level_zero:1) is kept.  An explicit
# ONEAPI_DEVICE_SELECTOR, or no sycl-ls, wins.
if [ -z "${ONEAPI_DEVICE_SELECTOR:-}" ]; then
    _sycl_ls="${ONEAPI_ROOT}/compiler/latest/bin/sycl-ls"
    _arc_sel=""
    if [ -x "$_sycl_ls" ]; then
        _arcs=$("$_sycl_ls" --ignore-device-selectors 2>/dev/null \
            | grep -E '\[level_zero:gpu\]\[level_zero:[0-9]+\].*Arc\(TM\) (Pro )?[AB][0-9]' \
            | sed -E 's/.*\[level_zero:([0-9]+)\].*/\1/' | sort -n)
        if [ "$(printf '%s\n' "$_arcs" | grep -c .)" = "1" ]; then
            _arc_sel="$_arcs"                 # exactly one discrete Arc: it is the one
        fi
    fi
    export ONEAPI_DEVICE_SELECTOR="level_zero:${_arc_sel:-1}"
    unset _sycl_ls _arcs _arc_sel
fi

# >4 GB device allocations (the expert arena) and free-memory queries (cudaMemGetInfo).
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export ZES_ENABLE_SYSMAN=1

# DG2/BMG have no FP64 hardware; the A770 must report the fp64 aspect (emulated)
# or kernels with double arithmetic are rejected at launch. The AOT image side is
# -cl-fp64-gen-emu in cmake/sycl_backend.cmake.
export NEO_FP64_EMULATION=1
