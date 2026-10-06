// src/kernels/sycl/native_qsa.cpp - SYCL port of src/kernels/cuda/native_qsa.cu; see
// include/strata/kernels/native_qsa.hpp.  Pinned llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d F32 arithmetic
// (MIT License, Copyright (c) 2023-2026 The ggml authors; full text in the CUDA file).
//
// The norm keeps the pinned 256/1024-thread xor reduction and `(scale * x) * gamma`; unlike native_gr_norm.cpp's
// kernel it has no restrict qualifiers, because the header allows output == input exactly.  The gate's sigmoid uses
// `sycl::exp` and an `fdiv_rn` reciprocal (IGC's default float division is not correctly rounded).
#include "strata/kernels/native_qsa.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "fp_exact.hpp"

#include <cuda_runtime.h>
#include <sycl/ext/intel/math.hpp>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

namespace im = strata::kernels::fp_exact;
std::atomic<bool> enabled{false};

inline float warp_sum(const sycl::sub_group& sg, float value) {
#pragma unroll
    for (int offset = 16; offset; offset >>= 1) value += sycl::permute_group_by_xor(sg, value, offset);
    return value;
}

template<int BlockSize>
void norm(sycl::queue& q, const float* input, const float* gamma, float* output, int n_cols, int n_rows, float epsilon) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sums(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_rows * BlockSize, BlockSize),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int tid = (int) it.get_local_id(0);
            const std::size_t row_offset = std::size_t(it.get_group(0)) * n_cols;
            const float* in = input + row_offset;
            float* out = output + row_offset;
            float partial = 0.0f;
            for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) {
                const float x = in[col];
                partial += x * x;
            }
            const sycl::sub_group sg = it.get_sub_group();
            partial = warp_sum(sg, partial);
            const int lane = tid % 32;
            if (lane == 0) sums[tid / 32] = partial;
            // every reduction read of the input precedes this barrier; afterwards each work-item reads and writes
            // only its own elements, which is what makes the exact in-place call safe
            sycl::group_barrier(it.get_group());
            partial = lane < BlockSize / 32 ? sums[lane] : 0.0f;
            partial = warp_sum(sg, partial);
            const float mean = partial / n_cols;
            const float scale = sycl::rsqrt(mean + epsilon);
            for (std::size_t col = tid; col < std::size_t(n_cols); col += BlockSize) out[col] = scale * in[col] * gamma[col];
        });
    });
}

std::size_t elements(int cols, int rows) {
    if (cols <= 0 || rows <= 0 || std::uint64_t(cols) * rows > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native QSA requires positive bounded dimensions");
    return std::size_t(cols) * rows;
}
bool valid(const void* ptr, std::size_t bytes) {
    const auto address = reinterpret_cast<std::uintptr_t>(ptr);
    return ptr && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, std::size_t an, const void* b, std::size_t bn) {
    const auto ap = reinterpret_cast<std::uintptr_t>(a), bp = reinterpret_cast<std::uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
void buffers(const float* input, std::size_t in_bytes, const float* weight, std::size_t weight_bytes, float* output,
             void* stream) {
    if (!stream || !valid(input, in_bytes) || !valid(weight, weight_bytes) || !valid(output, in_bytes) ||
        overlap(input, in_bytes, weight, weight_bytes) || overlap(output, in_bytes, weight, weight_bytes) ||
        (input != output && overlap(input, in_bytes, output, in_bytes)))
        throw std::invalid_argument("native QSA requires a stream, aligned spans, and disjoint buffers or exact input/output alias");
}
void check_launch() {
    const auto result = cudaGetLastError();
    if (result != cudaSuccess) throw std::runtime_error(std::string("native QSA launch: ") + cudaGetErrorString(result));
}
}  // namespace

void native_qsa_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_rms_norm_weighted(const float* input, const float* gamma, float* output, int n_cols, int n_rows,
                                  float epsilon, void* stream) {
    const auto count = elements(n_cols, n_rows);
    if (!std::isfinite(epsilon) || epsilon < 0.0f) throw std::invalid_argument("native QSA requires finite nonnegative epsilon");
    buffers(input, count * 4, gamma, std::size_t(n_cols) * 4, output, stream);
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    if (n_cols < 1024) norm<256>(q, input, gamma, output, n_cols, n_rows, epsilon);
    else norm<1024>(q, input, gamma, output, n_cols, n_rows, epsilon);
    check_launch();
}

void native_qsa_gate_apply(const float* attn, const float* q_full, float* output, int n_head, int head_dim,
                           void* stream) {
    const auto count = elements(head_dim, n_head);
    buffers(attn, count * 4, q_full, count * 8, output, stream);
    sycl_runtime::queue_from_stream(stream).parallel_for(sycl::range<1>(count), [=](sycl::id<1> id) {
        const std::size_t i = id[0], head = i / head_dim, channel = i % head_dim;
        const float raw = q_full[head * 2 * head_dim + head_dim + channel];
        const float sigmoid = im::fdiv_rn(1.0f, 1.0f + sycl::exp(-raw));
        output[i] = attn[i] * sigmoid;
    });
    check_launch();
}

}  // namespace strata::kernels
