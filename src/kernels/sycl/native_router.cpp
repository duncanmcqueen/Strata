// src/kernels/sycl/native_router.cpp - SYCL port of src/kernels/cuda/native_router.cu; see
// include/strata/kernels/native_router.hpp.  Pinned llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d topk-moe
// (MIT License, Copyright (c) 2023-2026 The ggml authors; full text in the CUDA file).
//
// The CUDA kernel launches a 32x8 block of which only row 0 works; here one 32-wide sub-group per token does the
// same work in the same order (the seven idle warps had no effect).  `expf` becomes `sycl::exp`; the two
// reciprocals are `fdiv_rn` (IGC's default float division is not correctly rounded); the xor reductions and the
// lowest-index tie rule are unchanged.
#include "strata/kernels/native_router.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "fp_exact.hpp"

#include <cuda_runtime.h>
#include <sycl/ext/intel/math.hpp>

#include <atomic>
#include <cfloat>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {

namespace im = strata::kernels::fp_exact;
std::atomic<bool> enabled{false};

void route(sycl::queue& q, const float* logits, int32_t* ids, float* weights, int n_tok) {
    q.parallel_for(sycl::nd_range<1>((size_t) n_tok * 32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const size_t tok = it.get_group(0);
        const float* lg = logits + tok * 512;
        int32_t* id = ids + tok * 10;
        float* wt = weights + tok * 10;
        const int lane = (int) it.get_local_id(0);
        float values[16];
#pragma unroll
        for (int i = 0; i < 16; ++i) values[i] = lg[lane + i * 32];
        float maximum = -INFINITY;
#pragma unroll
        for (int i = 0; i < 16; ++i) maximum = sycl::fmax(maximum, values[i]);
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) maximum = sycl::fmax(maximum, sycl::permute_group_by_xor(sg, maximum, mask));
        float sum = 0.0f;
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[i] = sycl::exp(values[i] - maximum);
            sum = im::fadd_rn(sum, values[i]);
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1) sum = im::fadd_rn(sum, sycl::permute_group_by_xor(sg, sum, mask));
        const float reciprocal = im::fdiv_rn(1.0f, sum);
#pragma unroll
        for (int i = 0; i < 16; ++i) {
            values[i] = im::fmul_rn(values[i], reciprocal);
            if (sycl::isnan(values[i])) values[i] = -FLT_MAX;
        }
        float selected = 0.0f, selected_sum = 0.0f;
        for (int rank = 0; rank < 10; ++rank) {
            float best = values[0];
            int expert = lane;
#pragma unroll
            for (int i = 1; i < 16; ++i)
                if (values[i] > best) { best = values[i]; expert = lane + i * 32; }
#pragma unroll
            for (int mask = 16; mask; mask >>= 1) {
                const float other = sycl::permute_group_by_xor(sg, best, mask);
                const int other_id = sycl::permute_group_by_xor(sg, expert, mask);
                if (other > best || (other == best && other_id < expert)) { best = other; expert = other_id; }
            }
            if ((expert & 31) == lane) {
                // the static index form of values[expert / 32] = -INFINITY (private arrays stay in registers)
#pragma unroll
                for (int i = 0; i < 16; ++i)
                    if (i == expert / 32) values[i] = -INFINITY;
                id[rank] = expert;
                // deliberately accumulated by WINNING EXPERT lane, in selection order (the CUDA comment)
                selected_sum = im::fadd_rn(selected_sum, best);
            }
            if (rank == lane) selected = best;
        }
#pragma unroll
        for (int mask = 16; mask; mask >>= 1)
            selected_sum = im::fadd_rn(selected_sum, sycl::permute_group_by_xor(sg, selected_sum, mask));
        selected_sum = sycl::fmax(selected_sum, 6.103515625e-5f);
        const float inverse_selected_sum = im::fdiv_rn(1.0f, selected_sum);
        if (lane < 10) wt[lane] = im::fmul_rn(selected, inverse_selected_sum);
    });
}
bool valid(const void* p, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(p);
    return p && address % 4 == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
void launch_check() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}  // namespace

void native_router_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_router_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_router_top10(const float* logits, int32_t* ids, float* weights, void* stream) {
    if (!stream || !valid(logits, 512 * 4) || !valid(ids, 10 * 4) || !valid(weights, 10 * 4) ||
        overlap(logits, 512 * 4, ids, 10 * 4) || overlap(logits, 512 * 4, weights, 10 * 4) ||
        overlap(ids, 10 * 4, weights, 10 * 4))
        throw std::invalid_argument("native router requires a stream, aligned spans, and disjoint outputs");
    route(sycl_runtime::queue_from_stream(stream), logits, ids, weights, 1);
    launch_check();
}

void native_router_top10_multi(const float* logits, int32_t* ids, float* weights, int n_tok, void* stream) {
    if (!stream || n_tok < 1 || !valid(logits, (size_t) n_tok * 512 * 4) || !valid(ids, (size_t) n_tok * 10 * 4) ||
        !valid(weights, (size_t) n_tok * 10 * 4))
        throw std::invalid_argument("native router (multi) requires a stream and aligned [n,512]/[n,10] buffers");
    route(sycl_runtime::queue_from_stream(stream), logits, ids, weights, n_tok);
    launch_check();
}

}  // namespace strata::kernels
