// src/core/sycl/device.cpp - the SYCL counterpart of src/core/device.cu (P2.S1: the GPU side of the runtime core).
//
// The host logic is device.cu's, through the CUDA-shaped runtime shim; what differs:
//   * the arena's NaN poison is a SYCL kernel;
//   * device_code_error() asks whether this binary carries a device image the selected card can run
//     (sycl::is_compatible on the poison kernel - every SYCL source is compiled for the same AOT/JIT targets, so it
//     stands for all of them), where CUDA asks cudaFuncGetAttributes;
//   * there is no compute-capability floor: the runtime reports an 8.0 sentinel (docs/INTEL_SYCL.md) and refuses
//     a GPU without 32-wide sub-groups at enumeration, which is the real requirement here.
#include "strata/core/device.hpp"
#include "strata/sycl_runtime/device_profile.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstring>

namespace strata::core {

namespace {

void check(cudaError_t e, const char* what) {
    if (e != cudaSuccess) throw CudaError(std::string(what) + ": " + cudaGetErrorString(e), (int) e);
}

class PoisonKernel;

// a NaN pattern, not zero (see device.cu)
void poison(float* p, uint64_t n_floats) {
    sycl::queue& q = sycl_runtime::queue_from_stream(nullptr);
    q.parallel_for<PoisonKernel>(sycl::range<1>((size_t) n_floats),
                                 [=](sycl::id<1> i) { p[i] = sycl::bit_cast<float>(0x7fc00000u); });
}

}  // namespace

const char* compiled_gpu_archs() { return ""; }

int device_count() {
    int count = 0;
    if (cudaGetDeviceCount(&count) != cudaSuccess) {
        cudaGetLastError();
        return 0;
    }
    return count < 0 ? 0 : count;
}

bool device_summary(int ordinal, std::string& name, std::string& detail) {
    cudaDeviceProp p{};
    if (ordinal < 0 || ordinal >= device_count() || cudaGetDeviceProperties(&p, ordinal) != cudaSuccess) {
        cudaGetLastError();
        return false;
    }
    // Build the immutable per-device profiles here: startup, outside any graph capture, so no device query ever
    // runs mid-capture when a selector is first consulted.
    sycl_runtime::warm_device_profiles();
    const sycl_runtime::DeviceProfile& prof = sycl_runtime::profile_for_device(ordinal);
    char buf[160];
    std::snprintf(buf, sizeof(buf), "Level Zero, %d compute units, %.1f GiB, sub-group 32, arch=%s", p.multiProcessorCount,
                  (double) p.totalGlobalMem / (1024.0 * 1024 * 1024), sycl_runtime::arch_name(prof.arch));
    name = p.name;
    detail = buf;
    return true;
}

std::string gpu_arch_problem(int) { return ""; }   // the runtime refuses a GPU without 32-wide sub-groups

std::string device_code_error() {
    try {
        const sycl::device d = sycl_runtime::device_current();
        if (sycl::is_compatible<PoisonKernel>(d)) return {};
        return "this Strata engine has no device image for " + d.get_info<sycl::info::device::name>() +
               " (rebuild with -DSTRATA_SYCL_AOT_DEVICES naming this card; the spir64 JIT image needs a driver that "
               "accepts it)";
    } catch (const sycl::exception& e) {
        return e.what();
    }
}

DeviceInfo device_info(int ordinal) {
    int count = 0;
    check(cudaGetDeviceCount(&count), "cudaGetDeviceCount");
    if (count == 0)
        throw CudaError("no Intel GPU is present; Strata's SYCL build needs an Arc GPU on Level Zero with 32-wide "
                        "sub-groups (check ONEAPI_DEVICE_SELECTOR and the driver)",
                        -1);
    if (ordinal < 0 || ordinal >= count)
        throw CudaError("device ordinal " + std::to_string(ordinal) + " is out of range (have " + std::to_string(count) + ")", -1);
    DeviceInfo d;
    d.ordinal = ordinal;
    check(cudaSetDevice(ordinal), "cudaSetDevice");
    cudaDeviceProp p{};
    check(cudaGetDeviceProperties(&p, ordinal), "cudaGetDeviceProperties");
    d.name = p.name;
    d.cc_major = p.major;
    d.cc_minor = p.minor;
    d.multi_processor_count = p.multiProcessorCount;
    size_t free_b = 0, total_b = 0;
    check(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    d.free_bytes = free_b;
    d.total_bytes = total_b;
    check(cudaDriverGetVersion(&d.driver_version), "cudaDriverGetVersion");
    check(cudaRuntimeGetVersion(&d.runtime_version), "cudaRuntimeGetVersion");
    return d;
}

DeviceArena::DeviceArena(uint64_t bytes, int ordinal, bool poison_it) : capacity_(bytes), ordinal_(ordinal), poison_(poison_it) {
    if (bytes == 0) throw CudaError("DeviceArena of 0 bytes", -1);
    check(cudaSetDevice(ordinal), "cudaSetDevice");
    check(cudaMalloc(&base_, (size_t) bytes), "cudaMalloc");
    if (poison_) {
        try {
            poison((float*) base_, bytes / sizeof(float));
        } catch (const sycl::exception& e) {
            throw CudaError(std::string("poison_kernel: ") + e.what(), -1);
        }
        check(cudaDeviceSynchronize(), "poison sync");
    }
}

DeviceArena::~DeviceArena() {
    if (base_) cudaFree(base_);
}

void* DeviceArena::alloc(uint64_t bytes, uint64_t align) {
    if (bytes == 0) return nullptr;
    if (align == 0 || (align & (align - 1)) != 0) throw CudaError("DeviceArena::alloc alignment must be a power of two", -1);
    const uint64_t start = (used_ + align - 1) & ~(align - 1);
    if (start + bytes > capacity_) {
        char msg[256];
        std::snprintf(msg, sizeof(msg),
                      "DeviceArena out of memory: asked for %llu B at offset %llu (align %llu) in a %llu B "
                      "region - the plan from P1.S9 did not close",
                      (unsigned long long) bytes, (unsigned long long) start, (unsigned long long) align,
                      (unsigned long long) capacity_);
        throw CudaError(msg, -1);
    }
    used_ = start + bytes;
    return (char*) base_ + start;
}

}  // namespace strata::core
