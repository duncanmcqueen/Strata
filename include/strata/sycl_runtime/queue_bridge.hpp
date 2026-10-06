#pragma once
// include/strata/sycl_runtime/queue_bridge.hpp - the SYCL kernel side's view of the
// runtime: how a ported kernel turns a CUDA-shaped `void* stream` into the
// sycl::queue it submits to.  Only STRATA_ENABLE_SYCL kernel/device sources include
// this; it pulls in <sycl/sycl.hpp> and must never be force-included.

#include <sycl/sycl.hpp>

namespace strata::sycl_runtime {

// The queue of a stream handle; nullptr selects the current device's default stream,
// matching the legacy-default-stream behaviour the CUDA sources rely on when they
// pass `stream == nullptr` and then cudaDeviceSynchronize.
sycl::queue& queue_from_stream(void* stream);

// The current device (cudaGetDevice) and its context - for USM allocations a ported
// file needs to make itself (constant-table replacements, scratch).
sycl::device device_current();
sycl::context context_shared();

}  // namespace strata::sycl_runtime
