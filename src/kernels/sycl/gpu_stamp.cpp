// src/kernels/sycl/gpu_stamp.cpp - the verify window's stage-profiler stamp (verify_kernels.hpp) on SYCL.
#include "strata/kernels/verify_kernels.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "mapped_host.hpp"

#include <cuda_runtime.h>
#include <sycl/ext/oneapi/experimental/clock.hpp>

#include <chrono>
#include <stdexcept>
#include <string>
#include <thread>

namespace strata::kernels {

// The A770 has no device-wide clock (sycl_ext_oneapi_clock's device scope; its sub-group counter is per EU and not
// comparable between kernels: two stamps 200 ms apart read as a negative delta).  The stamp is the HOST's clock
// instead: on the first call a host thread starts writing steady_clock nanoseconds into a host USM word, and the
// stamp kernel reads that word past the GPU caches (mapped_host.hpp).  The reading is late by one PCIe read and the
// writer's period, ~1-2 us - fine for the stage profiler's 50-500 us stages.  The thread keeps one core busy, so
// this runs only in the opt-in STRATA_VERIFY_PROFILE.  A GPU with a device clock uses it.
namespace {
unsigned long long* host_clock_word() {
    static unsigned long long* word = [] {
        auto* w = sycl::malloc_host<unsigned long long>(8, sycl_runtime::context_shared());
        if (w == nullptr) throw std::runtime_error("gpu_stamp: no host USM for the host clock");
        const auto now = [] {
            return (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
                std::chrono::steady_clock::now().time_since_epoch()).count();
        };
        __atomic_store_n(w, now(), __ATOMIC_RELEASE);
        std::thread([w, now] {
            for (;;) {
                __atomic_store_n(w, now(), __ATOMIC_RELEASE);
                for (int spin = 0; spin < 8; ++spin) __builtin_ia32_pause();
            }
        }).detach();
        return w;
    }();
    return word;
}
}  // namespace

void gpu_stamp(unsigned long long* buf, int i, void* stream) {
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    const sycl::device dev = q.get_device();
    if (dev.has(sycl::aspect::ext_oneapi_clock_device)) {
        const double ns_per_tick = (double) dev.get_info<sycl::info::device::profiling_timer_resolution>();
        q.single_task([=]() {
            const uint64_t ticks = sycl::ext::oneapi::experimental::clock<sycl::ext::oneapi::experimental::clock_scope::device>();
            buf[i] = (unsigned long long) ((double) ticks * ns_per_tick);
        });
    } else {
        const unsigned long long* word = host_clock_word();
        q.single_task([=]() {
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
            buf[i] = sycl_mapped::load(word);
        });
    }
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string("gpu_stamp: ") + cudaGetErrorString(e));
}

}  // namespace strata::kernels
