# Opt-in SYCL configuration (Intel Arc), modelled on hip_backend.cmake.  Unlike HIP,
# SYCL cannot compile the CUDA kernel sources: the device code is a real rewrite
# under src/kernels/sycl/, and the CUDA-shaped runtime the host sources speak is the
# strata_sycl_runtime library below (src/sycl_runtime/) behind the force-included
# include/strata/sycl_compat/cuda_runtime.h shim.
#
# Build with the oneAPI DPC++ compiler:
#   cmake -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx -DSTRATA_ENABLE_SYCL=ON ...
# Pin versions in tools/sycl/versions.txt; source tools/sycl/env.sh in every shell.
if(NOT CMAKE_CXX_COMPILER_ID MATCHES "IntelLLVM")
  message(WARNING
    "STRATA_ENABLE_SYCL expects the oneAPI DPC++ compiler (icx/icpx: -DCMAKE_C_COMPILER=icx "
    "-DCMAKE_CXX_COMPILER=icpx); CMAKE_CXX_COMPILER_ID is '${CMAKE_CXX_COMPILER_ID}'")
endif()

# AOT for the cards you own plus a spir64 JIT image as the fallback.  The ocloc device
# list matches tools/sycl/versions.txt (dg2-g10 = A770, bmg-g21 = B580).
set(STRATA_SYCL_TARGETS "spir64_gen,spir64" CACHE STRING "SYCL offload targets")
set(STRATA_SYCL_AOT_DEVICES "dg2-g10" CACHE STRING "ocloc AOT device list, e.g. dg2-g10 or dg2-g10,bmg-g21")
if(STRATA_SYCL_AOT_DEVICES STREQUAL "" OR STRATA_SYCL_AOT_DEVICES STREQUAL "none")
  # no AOT device named (a card setup has no ocloc name for): the spir64 JIT image only, compiled by the driver at
  # first use
  set(STRATA_SYCL_TARGETS "spir64" CACHE STRING "SYCL offload targets" FORCE)
elseif(STRATA_SYCL_TARGETS STREQUAL "spir64")
  # a cache left JIT-only by an earlier configure that named no device: AOT devices are named now
  set(STRATA_SYCL_TARGETS "spir64_gen,spir64" CACHE STRING "SYCL offload targets" FORCE)
endif()

# oneMKL provides the cuBLAS-shaped gemm (src/sycl_runtime/blas.cpp).  tools/sycl/env.sh
# puts <oneapi>/mkl/latest/lib/cmake/mkl on CMAKE_PREFIX_PATH; the -DMKL_DIR=... hint
# works too.  intel_thread (libiomp5 ships with the DPC++ compiler); the default tbb
# threading is not installed on every machine.
set(MKL_THREADING intel_thread)
find_package(MKL CONFIG REQUIRED)

add_library(strata_sycl_flags INTERFACE)
target_compile_options(strata_sycl_flags INTERFACE -fsycl "-fsycl-targets=${STRATA_SYCL_TARGETS}")
target_link_options(strata_sycl_flags INTERFACE -fsycl "-fsycl-targets=${STRATA_SYCL_TARGETS}")
# icpx defaults to -ffp-model=fast, whose vectorized host loops are NOT IEEE-exact
# (measured: x[i]*s off by 1 ulp on ~45% of elements vs g++/the GPU).  The parity
# tests' host oracles, and the engine's host-side reference math, assume IEEE.
target_compile_options(strata_sycl_flags INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-ffp-model=precise>")
# AOT-only options must name their backend or the spir64 JIT image would receive
# ocloc's -device flag.  The -options string is passed to ocloc verbatim:
#  - -ze-opt-greater-than-4GB-buffer-required: without it the AOT image refuses
#    kernels that touch buffers above 4 GB (the expert arena); the runtime side is
#    UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 (tools/sycl/env.sh).
#  - -cl-fp64-gen-emu: DG2/BMG have no FP64 hardware; this makes IGC emulate it.
#    Two ported kernels (quantize_q8_0, silu) use double arithmetic because their
#    CUDA reference does, and the parity contract is byte-exact.  The runtime side
#    is NEO_FP64_EMULATION=1 (tools/sycl/env.sh): without it the Level Zero device
#    does not report the fp64 aspect and kernel launch is rejected.
if(STRATA_SYCL_TARGETS MATCHES "spir64_gen")
  target_link_options(strata_sycl_flags INTERFACE
    "-Xsycl-target-backend=spir64_gen" "-device ${STRATA_SYCL_AOT_DEVICES} -options \"-cl-fp64-gen-emu -ze-opt-greater-than-4GB-buffer-required\"")
endif()

# The engine is a few executables and shared libraries (the runtime and the kernel families): setup copies them into
# one folder, and $ORIGIN lets each find the others there as well as in the build tree.  The oneAPI runtime libraries
# (libsycl, oneMKL, Level Zero) come from LD_LIBRARY_PATH, which setup's config carries.
list(APPEND CMAKE_BUILD_RPATH "$ORIGIN")

# Device code is split per kernel and the AOT (ocloc) jobs of one link run in parallel: one device module of
# every ported kernel took ocloc 41 minutes on one core (measured, 12-core host, after native_mmvq).
set(STRATA_SYCL_LINK_JOBS "8" CACHE STRING "Parallel ocloc jobs per SYCL link (-fsycl-max-parallel-link-jobs)")
target_compile_options(strata_sycl_flags INTERFACE "$<$<COMPILE_LANGUAGE:CXX>:-fsycl-device-code-split=per_kernel>")
target_link_options(strata_sycl_flags INTERFACE -fsycl-device-code-split=per_kernel
  "-fsycl-max-parallel-link-jobs=${STRATA_SYCL_LINK_JOBS}")

set(STRATA_SYCL_COMPAT_INCLUDE_DIR "${CMAKE_CURRENT_SOURCE_DIR}/include/strata/sycl_compat")
# SHARED, as strata_sycl_kernels is: the runtime holds the device list, streams, graphs and the last error, and
# one copy of that state must serve the kernel library and every executable that links it.
# A rebuilt runtime .so must not relink the kernel library (another full AOT compile) or every test executable:
# only the shared library's interface matters to them, and that is the shim header.
set(CMAKE_LINK_DEPENDS_NO_SHARED ON)
add_library(strata_sycl_runtime SHARED
  src/sycl_runtime/runtime.cpp
  src/sycl_runtime/memory.cpp
  src/sycl_runtime/graph.cpp
  src/sycl_runtime/device_profile.cpp
  src/sycl_runtime/blas.cpp)
target_include_directories(strata_sycl_runtime BEFORE PUBLIC
  "${STRATA_SYCL_COMPAT_INCLUDE_DIR}" "${CMAKE_CURRENT_SOURCE_DIR}/include")
target_compile_definitions(strata_sycl_runtime PUBLIC STRATA_USE_SYCL=1)
# The shim declares the CUDA runtime subset as plain host functions, force-included
# into every C++ translation unit exactly as the HIP shim is, so src/core/*.cpp, the
# parity tests and the ported kernel files compile untouched.
file(TO_CMAKE_PATH "${STRATA_SYCL_COMPAT_INCLUDE_DIR}/cuda_runtime.h" _strata_sycl_force)
target_compile_options(strata_sycl_runtime PUBLIC
  "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include ${_strata_sycl_force}>")
target_link_libraries(strata_sycl_runtime PUBLIC strata_sycl_flags MKL::MKL_SYCL)
# Which device-specific kernel families this build actually compiles.  The dispatch profile reads these so the
# selectors never advertise a kernel that is not in the binary.  XMX families stay unset (off) until a kernel is
# implemented AND qualified; the A770 build must never inherit a B70-only template.
target_compile_definitions(strata_sycl_runtime PRIVATE
  STRATA_SYCL_HAVE_TOPK_HIER=1
  STRATA_SYCL_HAVE_SCORES_TILED=1
  STRATA_SYCL_HAVE_PROMPT_ATTN_TILED=0
  STRATA_SYCL_HAVE_GROUPING_DEVICE=0
  STRATA_SYCL_HAVE_EXPERTS_BOUNDED=0)

# The SYCL build is incremental: the engine (strata_core, prefill, generate) stays
# CUDA/HIP-only until its sources are ported; what exists at each step is the runtime
# above plus the ported kernels (src/kernels/sycl/) and their parity tests.
message(STATUS "Strata: SYCL enabled, targets ${STRATA_SYCL_TARGETS}, AOT ${STRATA_SYCL_AOT_DEVICES}")
