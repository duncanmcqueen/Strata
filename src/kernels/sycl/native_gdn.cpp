// src/kernels/sycl/native_gdn.cpp - SYCL port of src/kernels/cuda/native_gdn.cu; see
// include/strata/kernels/native_gdn.hpp.  Pinned llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d
// gated_delta_net recurrence (MIT License, Copyright (c) 2023-2026 The ggml authors; full text in the CUDA file).
//
// The CUDA grid (h_v, 1, S/4) of 32x4 blocks is flattened to h_v * S/4 work-groups of 128 (head fastest); each
// 32-wide sub-group owns one state column, the CUDA warp's.  `expf` becomes `sycl::exp`.
#include "strata/kernels/native_gdn.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"

#include <cuda_runtime.h>

#include <atomic>
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {

std::atomic<bool> enabled{false};
constexpr int S = 128;

inline float warp_sum(const sycl::sub_group& sg, float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) value += sycl::permute_group_by_xor(sg, value, offset);
    return value;
}

bool valid_span(const void* pointer, size_t bytes) {
    const auto address = reinterpret_cast<uintptr_t>(pointer);
    return pointer && address % sizeof(float) == 0 && bytes <= UINTPTR_MAX - address;
}
bool overlap(const void* a, size_t an, const void* b, size_t bn) {
    const auto ap = reinterpret_cast<uintptr_t>(a), bp = reinterpret_cast<uintptr_t>(b);
    return ap < bp + bn && bp < ap + an;
}
}  // namespace

void native_gdn_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_gdn_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_gdn_step(float* state, const float* q, const float* k, const float* v, const float* gate,
                     const float* beta, float* output, const GdnShapes& shape, void* stream) {
    if (!stream || shape.S != S || shape.h_k <= 0 || shape.h_v <= 0 || shape.h_v > 65535 || shape.h_v % shape.h_k != 0)
        throw std::invalid_argument("native GDN requires a stream, S=128 and positive divisible head counts <=65535");
    const size_t state_bytes = size_t(S) * S * size_t(shape.h_v) * sizeof(float);
    const size_t qk_bytes = size_t(S) * size_t(shape.h_k) * sizeof(float);
    const size_t output_bytes = size_t(S) * size_t(shape.h_v) * sizeof(float);
    const size_t head_bytes = size_t(shape.h_v) * sizeof(float);
    if (!valid_span(state, state_bytes) || !valid_span(output, output_bytes) ||
        overlap(state, state_bytes, output, output_bytes))
        throw std::invalid_argument("native GDN requires aligned, disjoint state and output spans");
    const void* inputs[] = {q, k, v, gate, beta};
    const size_t bytes[] = {qk_bytes, qk_bytes, output_bytes, head_bytes, head_bytes};
    for (int i = 0; i < 5; ++i)
        if (!valid_span(inputs[i], bytes[i]) || overlap(state, state_bytes, inputs[i], bytes[i]) ||
            overlap(output, output_bytes, inputs[i], bytes[i]))
            throw std::invalid_argument("native GDN requires aligned input spans disjoint from state and output");
    const float scale = 1.0f / std::sqrt(float(S));
    const int h_k = int(shape.h_k), h_v = int(shape.h_v);
    sycl_runtime::queue_from_stream(stream).parallel_for(
        sycl::nd_range<1>(size_t(h_v) * (S / 4) * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int head = int(it.get_group(0) % h_v);
            const int z = int(it.get_group(0) / h_v);
            const int lane = int(it.get_local_id(0)) % 32, y = int(it.get_local_id(0)) / 32;
            const int col = z * 4 + y;
            const int q_head = head % h_k;
            float s_shard[4], k_reg[4], q_reg[4];
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                const int i = r * 32 + lane;
                s_shard[r] = state[(size_t(i) * h_v + head) * S + col];
                k_reg[r] = k[q_head * S + i];
                q_reg[r] = q[q_head * S + i];
            }
            const float g_val = sycl::exp(gate[head]);
            float kv_shard = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) kv_shard += s_shard[r] * k_reg[r];
            const float kv_col = warp_sum(sg, kv_shard);
            const float delta_col = (v[head * S + col] - g_val * kv_col) * beta[head];
            float attn_partial = 0.0f;
#pragma unroll
            for (int r = 0; r < 4; ++r) {
                s_shard[r] = g_val * s_shard[r] + k_reg[r] * delta_col;
                attn_partial += s_shard[r] * q_reg[r];
            }
            const float attn_col = warp_sum(sg, attn_partial);
            if (lane == 0) output[head * S + col] = attn_col * scale;
#pragma unroll
            for (int r = 0; r < 4; ++r) state[(size_t(r * 32 + lane) * h_v + head) * S + col] = s_shard[r];
        });
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

}  // namespace strata::kernels
