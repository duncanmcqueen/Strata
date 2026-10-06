# Strata Intel Arc Porting Guide

Oct 2, 2026 · @Duncan

## Approach

Add a third, opt-in GPU backend, `STRATA_ENABLE_SYCL`, built with Intel's oneAPI DPC++ compiler for Level Zero. Keep every host source and test unchanged by putting a CUDA-shaped runtime shim underneath them, and rewrite only the device code. This is the same split the AMD HIP backend uses, with one difference: HIP can compile CUDA kernel syntax, SYCL cannot, so the kernels are real rewrites rather than renames.

The guide was written against Strata `main` at commit `99f3dbd` (2026-10-03). Line counts and call counts below come from a scan of that tree; recheck them after rebasing.

| Option | Effort | Strengths | Why not first |
| --- | --- | --- | --- |
| **Native SYCL backend (this guide)** | 6–10 weeks for decode, more for prefill | Full control of sub-groups, host memory, graphs; upstreamable | Every kernel is rewritten |
| chipStar (HIP to SPIR-V) | 1–2 weeks to know | Reuses the HIP backend as is | Open bugs with warp-synchronous code ([#1409](https://github.com/CHIP-SPV/chipStar/issues/1409)); graph and host-memory semantics unproven for Strata's spin waits |
| Vulkan compute | Larger than SYCL | Runs everywhere | No equivalent of mapped host memory spin-waits or a BLAS; nothing to reuse |

The rule throughout: **the public kernel headers in `include/strata/kernels/*.hpp` are the seam.** Each declares host functions such as `wait_flag_ge(const uint32_t*, uint32_t, void* stream)`. A SYCL file implements the same function with the same contract, and the existing parity test for it must pass before the next kernel starts.

## Toolchain and hardware setup

Work on Linux with the `xe` kernel driver. Pin every version below in a `tools/sycl/versions.txt` so test results stay comparable.

1. Install Intel's compute runtime (Level Zero + OpenCL), the Level Zero loader and dev headers, and the oneAPI Base Toolkit (DPC++ `icpx`, oneMKL, oneDPL). Source `setvars.sh` in every build shell.
2. Verify devices: `sycl-ls` must list each Arc card under `level_zero:gpu`. Note the index of the inference card; do not assume it matches OpenVINO's `GPU.0`/`GPU.1` order.
3. Pin the device for every run with `ONEAPI_DEVICE_SELECTOR=level_zero:<n>`. Leave the display card out of the test runs.
4. Allow large allocations: export `UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1` and compile device code with `-Xs "-ze-opt-greater-than-4GB-buffer-required"` (section on pitfalls explains why).
5. Raise the locked-memory limit (`ulimit -l unlimited` or `/etc/security/limits.d`) and reserve 2 MB huge pages for the expert arena, as the CUDA build expects.
6. Install `intel-gpu-tools` and `xpu-smi` (or `nvtop`) for GPU busy, power and memory readings during benchmarks.

| Card | Architecture | VRAM | AOT device name | Sub-group sizes |
| --- | --- | --- | --- | --- |
| Arc A770 | Xe-HPG (Alchemist, DG2) | 16 GB | `dg2-g10` (alias `acm-g10`) | 8, 16, 32 |
| Arc B580 | Xe2-HPG (Battlemage) | 12 GB | `bmg-g21` | 16, 32 |
| Arc Pro B70 | Xe2 (Battlemage) | 32 GB | confirm with `ocloc compile --help` | 16, 32 |

Compile ahead-of-time for the cards you own and add a `spir64` JIT image as a fallback. Treat the sub-group column as the expected value, and confirm it at startup by querying `info::device::sub_group_sizes`; the backend must refuse to start if 32 is missing, because Strata's kernels assume 32-wide warps.

## Build system

Copy the shape of `cmake/hip_backend.cmake`. The top-level `CMakeLists.txt` already gates `strata_core` and every GPU target on `STRATA_ENABLE_CUDA OR STRATA_ENABLE_HIP`; extend each of those conditions with `OR STRATA_ENABLE_SYCL`, and refuse a configure that enables two backends.

New files:

| Path | Purpose |
| --- | --- |
| `cmake/sycl_backend.cmake` | Finds `icpx`, oneMKL; sets `-fsycl`, AOT targets, large-buffer flag; defines `STRATA_USE_SYCL=1` |
| `include/strata/sycl_compat/cuda_runtime.h` (+ `cuda_runtime_api.h`, `cublas_v2.h`, `cuda_fp16.h`) | Declares the CUDA runtime subset Strata uses, as plain host functions |
| `src/sycl_runtime/*.cpp` | Implements that subset on SYCL and Level Zero (library `strata_sycl_runtime`) |
| `src/core/sycl/device.cpp`, `pinned.cpp` | Replace `src/core/device.cu` and `pinned.cu` |
| `src/kernels/sycl/*.cpp` | One file per `src/kernels/cuda/*.cu`, same public header |
| `src/prefill/sycl/*.cpp` | Prefill kernels, after decode works |
| `tests/sycl/*.cpp` | Backend-specific tests (testing section) |

The compat headers are force-included into every C++ translation unit, exactly as the HIP shim is, so `src/core/*.cpp`, `src/program/generate.cpp` and all `*_parity.cpp` tests compile untouched:

```cmake
# cmake/sycl_backend.cmake (sketch)
set(STRATA_SYCL_TARGETS "spir64_gen,spir64" CACHE STRING "SYCL targets")
set(STRATA_SYCL_AOT_DEVICES "dg2-g10" CACHE STRING "ocloc device list, e.g. dg2-g10,bmg-g21")
find_package(MKL CONFIG REQUIRED)
add_library(strata_sycl_flags INTERFACE)
target_compile_options(strata_sycl_flags INTERFACE -fsycl -fsycl-targets=${STRATA_SYCL_TARGETS})
target_link_options(strata_sycl_flags INTERFACE -fsycl -fsycl-targets=${STRATA_SYCL_TARGETS}
  -Xs "-device ${STRATA_SYCL_AOT_DEVICES} -options -ze-opt-greater-than-4GB-buffer-required")
set(_compat ${CMAKE_CURRENT_SOURCE_DIR}/include/strata/sycl_compat)
add_library(strata_sycl_runtime STATIC src/sycl_runtime/runtime.cpp src/sycl_runtime/graph.cpp
  src/sycl_runtime/blas.cpp src/sycl_runtime/memory.cpp)
target_include_directories(strata_sycl_runtime BEFORE PUBLIC ${_compat} ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_compile_definitions(strata_sycl_runtime PUBLIC STRATA_USE_SYCL=1)
target_compile_options(strata_sycl_runtime PUBLIC "$<$<COMPILE_LANGUAGE:CXX>:SHELL:-include ${_compat}/cuda_runtime.h>")
target_link_libraries(strata_sycl_runtime PUBLIC strata_sycl_flags MKL::MKL_SYCL ze_loader)
```

Configure and build:

```sh
source /opt/intel/oneapi/setvars.sh
cmake -S . -B build-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_ENABLE_SYCL=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_SYCL_AOT_DEVICES=dg2-g10 -DSTRATA_BUILD_TESTS=ON
cmake --build build-sycl -j
```

Keep the ggml-cpu dependency exactly as pinned (`GIT_TAG 3cf03257…`): the CPU expert path does not change. Skip `STRATA_PREFILL_MMQ`, `STRATA_MMQ_KQUANTS` and `STRATA_ORCA_Q4KS_MMQ` on SYCL until a later phase; they compile ggml-cuda kernels.

## Runtime layer

The HIP shim already enumerates every CUDA runtime and cuBLAS call Strata makes: about 90 names. Implement that same list in `strata_sycl_runtime` and nothing more. Handles become small structs: `cudaStream_t` wraps an in-order `sycl::queue` plus capture state; `cudaEvent_t` wraps a `sycl::event`; `cudaGraphExec_t` wraps an executable command graph.

| CUDA call group | SYCL / Level Zero implementation | Notes |
| --- | --- | --- |
| `cudaSetDevice`, `cudaGetDeviceCount`, `cudaGetDeviceProperties` | Device list filtered to Level Zero GPUs; one shared context | Honor `ONEAPI_DEVICE_SELECTOR` |
| `cudaDeviceGetAttribute` | `MultiProcessorCount` = Xe-core count; `MaxSharedMemoryPerBlockOptin` = `local_mem_size`; compute capability = a fixed sentinel | Grep every compute-capability branch so the sentinel selects portable paths |
| `cudaMalloc`, `cudaFree`, `cudaMemset(Async)` | `sycl::malloc_device`, `sycl::free`, `queue::memset` | Relaxed allocation limits required |
| `cudaMemcpy(Async)`, `cudaMemcpyPeerAsync` | `queue::memcpy`; peer copies out of scope for v1 | Return an error for peer copies |
| `cudaMemGetInfo` | `ext_intel_free_memory` (needs `ZES_ENABLE_SYSMAN=1`) | Fall back to total minus tracked allocations |
| `cudaHostAlloc` (Mapped), `cudaMallocHost`, `cudaHostGetDevicePointer` | `sycl::malloc_host`; device pointer = host pointer | Host USM is device-readable over PCIe |
| `cudaHostRegister` / `Unregister` | No SYCL equivalent; allocate the arena with `malloc_host` instead (new `PageBacking::PinnedBySycl`) | Call sites: `pinned.cu:336,356`, `expert_source.cpp:1435` |
| `cudaStreamCreate(WithFlags)`, `Synchronize`, `Query` | In-order queue; `wait()`; `ext_oneapi_empty()` | One queue per CUDA stream |
| `cudaEventRecord`, `cudaStreamWaitEvent`, `cudaEventQuery`, `ElapsedTime` | `ext_oneapi_submit_barrier()`; barrier with wait list; `info::event_command_status`; profiling info | Timing needs a profiling-enabled queue |
| `cudaStreamBeginCapture` … `cudaGraphLaunch`, `cudaGraphUpload`, `GetNodes` | `command_graph::begin_recording(q)`, `end_recording`, `finalize`, `queue::ext_oneapi_graph`; upload is a no-op | `sycl_ext_oneapi_graph`; see graph rules below |
| `cudaLaunchHostFunc` | `handler::host_task` on the same queue | Used by the verify window (`verify.cpp:1258`) |
| `cudaFuncSetAttribute` (dynamic shared memory) | No-op that validates the request against `local_mem_size` | Fail loudly if too large |
| `cudaGetLastError`, `PeekAtLastError`, `GetErrorString` | Thread-local last error set by catching `sycl::exception` and the queue's async handler | SYCL reports errors by exception |
| `cublasCreate`, `SetStream`, `GemmEx`, `SetWorkspace`, `SetMathMode` | oneMKL `blas::column_major::gemm` on the bound queue; workspace and math mode are no-ops | Map `CUDA_R_16F/16BF/32F` and the transpose flags |
| `cudaMemcpyToSymbol` | Replace `__constant__` tables with `device_global` or a device buffer | Port with the kernel that owns the symbol |

Three graph rules carry over verbatim from `include/strata/core/graph.hpp`, and they hold for SYCL graphs too:

1. Kernel arguments are frozen at record time. Anything that changes per token must be data in a device buffer.
2. Every buffer a recorded body touches must be allocated before recording and never moved.
3. Waits must poll the runtime (`cudaEventQuery`), not memory. On Level Zero, query the event; a host loop that only reads memory can starve submission.

A SYCL graph cannot record a queue that is already recording, and some commands (host tasks, memsets on host memory) may be unsupported inside a graph depending on the DPC++ release. Make `cudaStreamBeginCapture` return `cudaErrorStreamCaptureUnsupported` for anything the installed runtime rejects, so Strata's existing fallbacks take over.
