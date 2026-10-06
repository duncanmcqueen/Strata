// src/kernels/sycl/native_moe.cpp - SYCL port of src/kernels/cuda/native_moe.cu; see
// include/strata/kernels/native_moe.hpp.  Pinned llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d weighted
// reduction (MIT License, Copyright (c) 2023-2026 The ggml authors; full text in the CUDA file).
//
// The header's rounding contract is written out instead of left to contraction: the first product rounded on its
// own, the following products FMA-accumulated in expert order, the shared row added once.
#include "strata/kernels/native_moe.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "fp_exact.hpp"

#include <cuda_runtime.h>
#include <sycl/ext/intel/math.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>

namespace strata::kernels {
namespace {

namespace im = strata::kernels::fp_exact;
std::atomic<bool> enabled{false};

void combine(sycl::queue& q, const float* parts, const float* weights, const float* shared, float* output,
             int64_t n_embd, int k, int n_tok) {
    q.parallel_for(sycl::range<2>((size_t) n_tok, (size_t) n_embd), [=](sycl::id<2> id) {
        const int64_t tk = (int64_t) id[0], col = (int64_t) id[1];
        const float* p = parts + tk * k * n_embd;
        const float* w = weights + tk * k;
        float sum = im::fmul_rn(p[col], w[0]);
        for (int expert = 1; expert < k; ++expert) sum = sycl::fma(p[int64_t(expert) * n_embd + col], w[expert], sum);
        if (shared) sum = im::fadd_rn(sum, shared[tk * n_embd + col]);
        output[tk * n_embd + col] = sum;
    });
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
bool valid_span(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % alignof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}  // namespace

void native_moe_combine_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_moe_combine_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_moe_combine(const float* parts, const float* weights, const float* shared, float* output, int64_t n_embd,
                        int64_t k, void* stream) {
    if (!stream || n_embd <= 0 || n_embd > std::numeric_limits<int>::max() || k < 1 || k > 15)
        throw std::invalid_argument("native MoE combine requires a stream, positive width and 1..15 experts");
    const size_t row_bytes = size_t(n_embd) * sizeof(float);
    const size_t part_bytes = row_bytes * size_t(k), weight_bytes = size_t(k) * sizeof(float);
    if (!valid_span(parts, part_bytes) || !valid_span(weights, weight_bytes) || !valid_span(output, row_bytes) ||
        (shared && !valid_span(shared, row_bytes)) || overlap(output, row_bytes, parts, part_bytes) ||
        overlap(output, row_bytes, weights, weight_bytes) || (shared && overlap(output, row_bytes, shared, row_bytes)))
        throw std::invalid_argument("native MoE combine requires aligned spans and disjoint output");
    combine(sycl_runtime::queue_from_stream(stream), parts, weights, shared, output, n_embd, int(k), 1);
}

void native_moe_combine_multi(const float* parts, const float* weights, const float* shared, float* output,
                              int64_t n_embd, int64_t k, int n_tok, void* stream) {
    if (!stream || n_embd <= 0 || k < 1 || k > 15 || n_tok < 1)
        throw std::invalid_argument("native MoE combine (multi) requires a stream, width, 1..15 experts, tokens");
    combine(sycl_runtime::queue_from_stream(stream), parts, weights, shared, output, n_embd, int(k), n_tok);
}

}  // namespace strata::kernels
