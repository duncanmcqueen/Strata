// src/kernels/sycl/gpu_stamp.cpp - the verify window's stage-profiler stamp (verify_kernels.hpp) on SYCL.
#include "strata/kernels/verify_kernels.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "mapped_host.hpp"

#include <cuda_runtime.h>
#include <sycl/ext/oneapi/experimental/clock.hpp>

#include <atomic>
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
//
// The writer must be JOINED, not detached: it never returns, and a detached thread outlives main() to write into a
// host USM word the runtime tears down at exit.  On Battlemage that made `sycl_engine_kernels` segfault in the
// writer during process exit (main in exit(), the thread at the atomic store).  A function-local static is
// constructed on first use - after the SYCL runtime - so its destructor (which stops and joins the thread) runs
// before the runtime's teardown, and the writer is gone before the word is freed.
namespace {
unsigned long long now_ns() {
    return (unsigned long long) std::chrono::duration_cast<std::chrono::nanoseconds>(
        std::chrono::steady_clock::now().time_since_epoch()).count();
}

struct HostClock {
    unsigned long long* word;
    std::atomic<bool> stop{false};
    std::thread writer;

    HostClock() : word(sycl::malloc_host<unsigned long long>(8, sycl_runtime::context_shared())) {
        if (word == nullptr) throw std::runtime_error("gpu_stamp: no host USM for the host clock");
        __atomic_store_n(word, now_ns(), __ATOMIC_RELEASE);
        writer = std::thread([this] {
            while (!stop.load(std::memory_order_relaxed)) {
                __atomic_store_n(word, now_ns(), __ATOMIC_RELEASE);
                for (int spin = 0; spin < 8; ++spin) __builtin_ia32_pause();
            }
        });
    }
    ~HostClock() {
        stop.store(true, std::memory_order_relaxed);
        if (writer.joinable()) writer.join();
    }
};

unsigned long long* host_clock_word() {
    static HostClock clock;   // joined at static destruction, before the SYCL runtime frees the word
    return clock.word;
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
