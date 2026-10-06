# Source this in every SYCL build/test shell:  . tools/sycl/env.sh
# This machine's oneAPI install has no top-level setvars.sh, so set the compiler
# and oneMKL paths directly. ONEAPI_ROOT override: ONEAPI_ROOT=/elsewhere . tools/sycl/env.sh
ONEAPI_ROOT="${ONEAPI_ROOT:-/opt/intel/oneapi}"
export PATH="${ONEAPI_ROOT}/compiler/latest/bin:${PATH}"
export LD_LIBRARY_PATH="${ONEAPI_ROOT}/compiler/latest/lib:${ONEAPI_ROOT}/mkl/latest/lib:${LD_LIBRARY_PATH}"
export CMAKE_PREFIX_PATH="${ONEAPI_ROOT}/mkl/latest/lib/cmake/mkl:${CMAKE_PREFIX_PATH}"

# The inference card is the A770 (level_zero:1 here - check with sycl-ls, do not
# assume the order matches OpenVINO's GPU.0/GPU.1). The B580 (level_zero:0) drives
# the display and stays out of test runs.
export ONEAPI_DEVICE_SELECTOR="${ONEAPI_DEVICE_SELECTOR:-level_zero:1}"

# >4 GB device allocations (the expert arena) and free-memory queries (cudaMemGetInfo).
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1
export ZES_ENABLE_SYSMAN=1

# DG2/BMG have no FP64 hardware; the A770 must report the fp64 aspect (emulated)
# or kernels with double arithmetic are rejected at launch. The AOT image side is
# -cl-fp64-gen-emu in cmake/sycl_backend.cmake.
export NEO_FP64_EMULATION=1
