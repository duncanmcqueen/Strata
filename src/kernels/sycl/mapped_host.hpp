#pragma once
// src/kernels/sycl/mapped_host.hpp - private to the SYCL kernels: reading host USM that the HOST updates while
// kernels run (doorbell flags, the CPU expert pool's results, mapped sampler parameters).
//
// Measured on the A770 (Level Zero, xe driver): a running kernel never saw a host store to host USM through a plain,
// volatile, device- or system-scope atomic load, or after a system-scope acquire fence - 2,000,000 polls each, all
// served from the GPU caches; a spin wait built on them ran until the driver reset the device.  A load with the
// sycl_ext_intel_cache_controls "uncached in L1 and L3" read hint saw the store within the 20 ms the host took to make
// it.  That load is not volatile, so a spin loop must also contain a synchronization operation, or the compiler may
// hoist the load out of a loop it can assume terminates (measured: without the fence the uncached spin hung too).
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/experimental/cache_control_properties.hpp>
#include <sycl/ext/oneapi/experimental/annotated_ptr/annotated_ptr.hpp>

namespace strata::kernels::sycl_mapped {

namespace syclex = sycl::ext::oneapi::experimental;
namespace intelex = sycl::ext::intel::experimental;
using uncached_read = intelex::read_hint_key::value_t<
    intelex::cache_control<intelex::cache_mode::uncached, syclex::cache_level::L1, syclex::cache_level::L3>>;

/// p[i], read past the GPU caches.
template<class T> inline T load(const T* p, size_t i = 0) {
    syclex::annotated_ptr<T, decltype(syclex::properties(uncached_read{}))> a(const_cast<T*>(p));
    return a[i];
}

using uncached_write = intelex::write_hint_key::value_t<
    intelex::cache_control<intelex::cache_mode::uncached, syclex::cache_level::L1, syclex::cache_level::L3>>;

/// p[i] = v, written past the GPU caches (a value the host polls while the stream continues).  Measured on the A770: a
/// plain or volatile store to host USM, even followed by a system-scope fence, reached the host only when the kernel
/// ended (482 ms into a 482 ms kernel); this store reached it at 44 ms, the launch latency.  Within a captured graph
/// the next kernel may be a spin wait on the host's answer, so the kernel end comes too late.
template<class T> inline void store(T* p, T v, size_t i = 0) {
    syclex::annotated_ptr<T, decltype(syclex::properties(uncached_write{}))> a(p);
    a[i] = v;
}

/// Spin until pred(load(flag)) holds; the acquire fence keeps every later read after the observed value and keeps the
/// load inside the loop.
template<class T, class Pred> inline T spin_until(const T* flag, Pred pred) {
    while (true) {
        sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::device);
        const T v = load(flag);
        if (pred(v)) {
            sycl::atomic_fence(sycl::memory_order::acquire, sycl::memory_scope::system);
            return v;
        }
    }
}

}  // namespace strata::kernels::sycl_mapped
