// src/sycl_runtime/memory.cpp - the CUDA memory calls on SYCL USM.
//
// Device memory is sycl::malloc_device against the SHARED context (one context spans
// every Level Zero GPU of the run, so a pointer allocated here is valid on any of
// that context's queues).  Mapped host memory is sycl::malloc_host: host USM is
// device-readable over PCIe and the device pointer IS the host pointer, which is
// what cudaHostGetDevicePointer returns.  Copies are queue memcpy/memset; the
// direction enum is validated but not otherwise needed, because USM copies resolve
// the locations from the pointers.
#include "../sycl_runtime/internal.hpp"

#include <cstring>
#include <map>
#include <unordered_map>

using strata::sycl_runtime::current_device;
using strata::sycl_runtime::fail;
using strata::sycl_runtime::from_exception;
using strata::sycl_runtime::g_allocated_bytes;
using strata::sycl_runtime::registry;
using strata::sycl_runtime::set_last_error;
using strata::sycl_runtime::stream_or_default;

namespace {

bool valid_kind(cudaMemcpyKind kind) {
    return kind >= cudaMemcpyHostToHost && kind <= cudaMemcpyDefault;
}

// Live device allocations (base -> size): cudaFree keeps the memGetInfo fallback honest with them, and
// device_offset_end finds the allocation that holds an address.
std::mutex g_sizes_mutex;
std::map<void*, size_t> g_sizes;

}  // namespace

namespace strata::sycl_runtime {
/// How far into its device allocation the byte range [p, p + bytes) ends; 0 when p is not in one.
size_t device_offset_end(const void* p, size_t bytes) {
    std::lock_guard<std::mutex> lock(g_sizes_mutex);
    auto it = g_sizes.upper_bound(const_cast<void*>(p));
    if (it == g_sizes.begin()) return 0;
    --it;
    const auto base = reinterpret_cast<uintptr_t>(it->first), q = reinterpret_cast<uintptr_t>(p);
    if (q >= base + it->second) return 0;
    return (size_t) (q - base) + bytes;
}
}  // namespace strata::sycl_runtime

cudaError_t cudaMalloc(void** pointer, size_t bytes) {
    if (!pointer) return fail(cudaErrorInvalidValue);
    *pointer = nullptr;
    try {
        void* p = sycl::malloc_device(bytes ? bytes : 1, registry().devices[current_device()], registry().context);
        if (!p) return fail(cudaErrorMemoryAllocation);
        {
            std::lock_guard<std::mutex> lock(g_sizes_mutex);
            g_sizes[p] = bytes;
        }
        g_allocated_bytes.fetch_add(static_cast<int64_t>(bytes), std::memory_order_relaxed);
        *pointer = p;
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaFree(void* pointer) {
    if (!pointer) return set_last_error(cudaSuccess);
    try {
        // USM free is not ordered against in-flight queue work (cudaFree is): the
        // arena users free after synchronizing, so wait the device before releasing.
        cudaDeviceSynchronize();
        sycl::free(pointer, registry().context);
        {
            std::lock_guard<std::mutex> lock(g_sizes_mutex);
            const auto it = g_sizes.find(pointer);
            if (it != g_sizes.end()) {
                g_allocated_bytes.fetch_sub(static_cast<int64_t>(it->second), std::memory_order_relaxed);
                g_sizes.erase(it);
            }
        }
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaMallocHost(void** pointer, size_t bytes) {
    return cudaHostAlloc(pointer, bytes, cudaHostAllocDefault);
}

cudaError_t cudaHostAlloc(void** pointer, size_t bytes, unsigned int flags) {
    if (!pointer) return fail(cudaErrorInvalidValue);
    *pointer = nullptr;
    if (flags & ~(cudaHostAllocPortable | cudaHostAllocMapped | cudaHostAllocWriteCombined))
        return fail(cudaErrorInvalidValue);
    try {
        // Host USM: one allocation the device reads over PCIe.  Mapped vs portable is
        // a CUDA-side distinction; here both flags are accepted and mean the same.
        void* p = sycl::malloc_host(bytes ? bytes : 1, registry().context);
        if (!p) return fail(cudaErrorMemoryAllocation);
        *pointer = p;
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaFreeHost(void* pointer) {
    if (!pointer) return set_last_error(cudaSuccess);
    try {
        cudaDeviceSynchronize();  // same ordering argument as cudaFree
        sycl::free(pointer, registry().context);
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaHostGetDevicePointer(void** devicePointer, void* hostPointer, unsigned int) {
    if (!devicePointer || !hostPointer) return fail(cudaErrorInvalidValue);
    *devicePointer = hostPointer;  // host USM is device-accessible as-is
    return set_last_error(cudaSuccess);
}

cudaError_t cudaHostRegister(void*, size_t, unsigned int) {
    // No SYCL equivalent: the SYCL port of pinned.cu allocates the arena with
    // sycl::malloc_host instead (PageBacking::PinnedBySycl).  Fail loudly so a call
    // site that was missed cannot run unpinned and wonder about the bandwidth.
    return fail(cudaErrorInvalidValue);
}

cudaError_t cudaHostUnregister(void*) {
    return set_last_error(cudaSuccess);  // paired with the refusal above: nothing to undo
}

cudaError_t cudaMemcpy(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind) {
    if (!valid_kind(kind) || (!dst && bytes) || (!src && bytes)) return fail(cudaErrorInvalidValue);
    if (bytes == 0) return set_last_error(cudaSuccess);
    try {
        // The legacy default stream: CUDA's synchronous copy starts after the work already in the
        // device's blocking streams and in the default stream, and returns when it completes.
        strata_cuda_stream* d = stream_or_default(nullptr);
        strata::sycl_runtime::wait_blocking_streams(d->device);
        d->queue.memcpy(dst, src, bytes).wait();
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaMemcpyAsync(void* dst, const void* src, size_t bytes, cudaMemcpyKind kind, cudaStream_t stream) {
    if (!valid_kind(kind) || (!dst && bytes) || (!src && bytes)) return fail(cudaErrorInvalidValue);
    if (bytes == 0) return set_last_error(cudaSuccess);
    try {
        strata_cuda_stream* const sp = stream_or_default(stream);
        strata::sycl_runtime::note_submit(sp, sp->capturing ? "memcpy@c" : "memcpy", __builtin_return_address(0));
        sp->queue.memcpy(dst, src, bytes);
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaMemcpy2DAsync(void* dst, size_t dstPitch, const void* src, size_t srcPitch, size_t width,
                              size_t height, cudaMemcpyKind kind, cudaStream_t stream) {
    if (!valid_kind(kind) || !dst || !src) return fail(cudaErrorInvalidValue);
    try {
        sycl::queue& q = stream_or_default(stream)->queue;
        auto* d = static_cast<char*>(dst);
        const auto* s = static_cast<const char*>(src);
        for (size_t row = 0; row < height; ++row) q.memcpy(d + row * dstPitch, s + row * srcPitch, width);
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaMemcpyPeerAsync(void*, int, const void*, int, size_t, cudaStream_t) {
    // Out of scope for v1 (the porting guide's runtime table); refused rather than
    // silently routed through the host.
    return fail(cudaErrorInvalidValue);
}

cudaError_t cudaMemset(void* pointer, int value, size_t bytes) {
    if (!pointer && bytes) return fail(cudaErrorInvalidValue);
    if (bytes == 0) return set_last_error(cudaSuccess);
    try {
        strata_cuda_stream* d = stream_or_default(nullptr);
        strata::sycl_runtime::wait_blocking_streams(d->device);  // the legacy default stream, as cudaMemcpy
        d->queue.memset(pointer, value, bytes).wait();
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaMemsetAsync(void* pointer, int value, size_t bytes, cudaStream_t stream) {
    if (!pointer && bytes) return fail(cudaErrorInvalidValue);
    if (bytes == 0) return set_last_error(cudaSuccess);
    try {
        stream_or_default(stream)->queue.memset(pointer, value, bytes);
        return set_last_error(cudaSuccess);
    } catch (const sycl::exception& e) {
        return from_exception(e);
    }
}

cudaError_t cudaMemGetInfo(size_t* freeBytes, size_t* totalBytes) {
    if (!freeBytes || !totalBytes) return fail(cudaErrorInvalidValue);
    const sycl::device& d = registry().devices[current_device()];
    const size_t total = d.get_info<sycl::info::device::global_mem_size>();
    *totalBytes = total;
    try {
        // The Level Zero/sysman reading (needs ZES_ENABLE_SYSMAN=1); counts the whole
        // card, other processes included, like cudaMemGetInfo does.
        *freeBytes = d.get_info<sycl::ext::intel::info::device::free_memory>();
    } catch (const sycl::exception&) {
        // Fallback: the card minus what this process is known to hold.  Understates
        // pressure from other processes; still bounds the arena sanely.
        const int64_t held = g_allocated_bytes.load(std::memory_order_relaxed);
        *freeBytes = held >= 0 && static_cast<size_t>(held) < total ? total - static_cast<size_t>(held) : total;
    }
    return set_last_error(cudaSuccess);
}
