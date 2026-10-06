// src/kernels/sycl/native_flash_attn.cpp - SYCL port of src/kernels/cuda/native_flash_attn.cu; see
// include/strata/kernels/native_flash_attn.hpp.  Pinned llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d vector
// FlashAttention, D = 256, one column, F16 K/V (MIT License, Copyright (c) 2023-2026 The ggml authors).
//
// The CUDA (32, 4) block is a 128-wide work-group (warp = tid / 32); `__syncwarp` is a sub-group barrier; the
// 8-lane `warp_sum<8>` is xor permutes with offsets 4, 2, 1 (they stay inside each 8-lane group); `expf` is
// `sycl::exp`; the pinned accumulate order (`fma(rescale, old, round(V*w))` for the first column of a tile,
// `fma(V, w, acc)` after) keeps its explicit intrinsics; the final division is `fdiv_rn`.
#include "strata/kernels/native_flash_attn.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "fp_exact.hpp"

#include <cuda_runtime.h>
#include <sycl/ext/intel/math.hpp>

#include <cfloat>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

namespace im = strata::kernels::fp_exact;

struct Span { const void* p; std::size_t n, alignment; };
void validate_spans(const Span* spans, int count) {
    for (int i = 0; i < count; ++i) {
        const auto a = reinterpret_cast<std::uintptr_t>(spans[i].p);
        if (!a || a % spans[i].alignment || spans[i].n > UINTPTR_MAX - a)
            throw std::invalid_argument("native FlashAttention requires nonnull aligned bounded spans");
        for (int j = 0; j < i; ++j) {
            const auto b = reinterpret_cast<std::uintptr_t>(spans[j].p);
            if (a < b + spans[j].n && b < a + spans[i].n)
                throw std::invalid_argument("native FlashAttention requires disjoint buffers");
        }
    }
}

}  // namespace

void native_flash_attn_short_step(const float* q, const uint16_t* k, const uint16_t* v, const int32_t* step,
                                  int64_t capacity, int max_context, const QsaShapes& shapes, float* out,
                                  int32_t* status, const uint16_t* mask, void* stream) {
    if (!stream || shapes.n_head != 24 || shapes.n_head_kv != 2 || shapes.head_dim != 256 || shapes.idx_block != 4 ||
        shapes.idx_top_k < 256 || capacity < 256 ||
        uint64_t(capacity) > std::numeric_limits<std::size_t>::max() / 1024 || max_context < 1 || max_context > 256)
        throw std::invalid_argument("native FlashAttention supports only Q24x256/KV2x256, capacity>=256 and context1..256 on an explicit stream");
    const std::size_t kv_bytes = std::size_t(capacity) * 1024;
    const Span spans[] = {{q, 24 * 256 * 4, 4}, {k, kv_bytes, 2}, {v, kv_bytes, 2}, {step, kStepCount * 4, 4},
                          {out, 24 * 256 * 4, 4}, {status, 4, 4}, {mask, 256 * 2, 2}};
    validate_spans(spans, mask ? 7 : 6);
    constexpr int padded_length = 256;
    constexpr float scale = 0.0625f;
    try {
        sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> tile(sycl::range<1>(4 * 4 * 256), h);
            sycl::local_accessor<float, 1> max_shared(sycl::range<1>(32), h), sum_shared(sycl::range<1>(32), h);
            h.parallel_for(sycl::nd_range<1>(24 * 128, 128), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const int tid = (int) it.get_local_id(0), lane = tid % 32, warp = tid / 32;
                const int head = (int) it.get_group(0), kv = head / 12;
                const int width = step[kStepWidth], nkv = step[kStepNKv];
                const bool valid = width >= 1 && width <= max_context && nkv == width && step[kStepPos] == width - 1 &&
                                   step[kStepNBid] == width / 4;
                if (head == 0 && tid == 0) *status = valid ? kNativeFlashAttnSuccess : kNativeFlashAttnUnsupportedStep;
                if (!valid) {   // uniform over the work-group, before any barrier
                    out[head * 256 + tid] = sycl::bit_cast<float>(0x7fc00000u);
                    out[head * 256 + tid + 128] = sycl::bit_cast<float>(0x7fc00000u);
                    return;
                }
                auto h2f = [](uint16_t b) { return f32_from_f16(b); };
                auto sum8 = [&](float x) {
                    for (int offset = 4; offset; offset >>= 1) x += sycl::permute_group_by_xor(sg, x, offset);
                    return x;
                };
                float qx[16], qy[16], vx[16] = {}, vy[16] = {};
                float maximum = -FLT_MAX / 2.0f, sum = 0.0f;
#pragma unroll
                for (int i0 = 0; i0 < 128; i0 += 32) {
                    const int i = i0 + (lane % 8) * 4;
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        const int d = 2 * (i + j);
                        qx[i0 / 8 + j] = q[head * 256 + d] * scale;
                        qy[i0 / 8 + j] = q[head * 256 + d + 1] * scale;
                    }
                }
                for (int base = 0; base < padded_length; base += 128) {
                    float score = 0.0f, next_max = maximum;
#pragma unroll
                    for (int row = 0; row < 8; ++row) {
                        const int cell = base + warp * 32 + (lane & ~7) + row;
                        float dot = 0.0f;
#pragma unroll
                        for (int i0 = 0; i0 < 128; i0 += 32) {
                            const int i = i0 + (lane % 8) * 4;
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const int d = 2 * (i + j);
                                const float a = cell < width ? h2f(k[(cell * 2 + kv) * 256 + d]) : 0.0f;
                                const float b = cell < width ? h2f(k[(cell * 2 + kv) * 256 + d + 1]) : 0.0f;
                                dot += a * qx[i0 / 8 + j];
                                dot += b * qy[i0 / 8 + j];
                            }
                        }
                        dot = sum8(dot);
                        dot += cell < width ? (mask ? h2f(mask[cell]) : 0.0f) : sycl::bit_cast<float>(0xff800000u);
                        next_max = sycl::fmax(next_max, dot + (3.0f * 0.6931f));
                        if (lane % 8 == row) score = dot;
                    }
#pragma unroll
                    for (int offset = 8; offset < 32; offset <<= 1)
                        next_max = sycl::fmax(next_max, sycl::permute_group_by_xor(sg, next_max, offset));
                    const float rescale = sycl::exp(maximum - next_max);
                    maximum = next_max;
                    score = sycl::exp(score - maximum);
                    sum = sum * rescale + score;
                    tile[tid] = score;
                    sycl::group_barrier(sg);
#pragma unroll
                    for (int k0 = 0; k0 < 32; k0 += 4) {
                        const int local = warp * 32 + k0 + lane / 8, cell = base + local;
                        const float weight = tile[local];
#pragma unroll
                        for (int i0 = 0; i0 < 128; i0 += 32) {
                            const int i = i0 + (lane % 8) * 4;
#pragma unroll
                            for (int j = 0; j < 4; ++j) {
                                const int d = 2 * (i + j);
                                const float a = cell < width ? h2f(v[(cell * 2 + kv) * 256 + d]) : 0.0f;
                                const float b = cell < width ? h2f(v[(cell * 2 + kv) * 256 + d + 1]) : 0.0f;
                                float& ax = vx[i0 / 8 + j];
                                float& ay = vy[i0 / 8 + j];
                                if (k0 == 0) {
                                    ax = im::fmaf_rn(rescale, ax, im::fmul_rn(a, weight));
                                    ay = im::fmaf_rn(rescale, ay, im::fmul_rn(b, weight));
                                } else {
                                    ax = im::fmaf_rn(a, weight, ax);
                                    ay = im::fmaf_rn(b, weight, ay);
                                }
                            }
                        }
                    }
                    sycl::group_barrier(sg);   // the next tile's scores overwrite tile[] read above
                }
                if (warp == 0) { max_shared[lane] = -FLT_MAX / 2.0f; sum_shared[lane] = 0.0f; }
                sycl::group_barrier(it.get_group());
                if (lane == 0) max_shared[warp] = maximum;
                sycl::group_barrier(it.get_group());
                float global_max = max_shared[lane];
                for (int offset = 16; offset; offset >>= 1)
                    global_max = sycl::fmax(global_max, sycl::permute_group_by_xor(sg, global_max, offset));
                const float rescale = sycl::exp(maximum - global_max);
#pragma unroll
                for (int i = 0; i < 16; ++i) {
                    vx[i] = im::fmul_rn(vx[i], rescale);
                    vy[i] = im::fmul_rn(vy[i], rescale);
                }
#pragma unroll
                for (int i0 = 0; i0 < 128; i0 += 32) {
                    const int start = warp * 4 * 256 + (lane / 8) * 256 + 2 * (i0 + (lane % 8) * 4);
#pragma unroll
                    for (int j = 0; j < 4; ++j) {
                        tile[start + 2 * j] = vx[i0 / 8 + j];
                        tile[start + 2 * j + 1] = vy[i0 / 8 + j];
                    }
                }
                sum *= rescale;
                for (int offset = 16; offset; offset >>= 1) sum += sycl::permute_group_by_xor(sg, sum, offset);
                if (lane == 0) sum_shared[warp] = sum;
                sycl::group_barrier(it.get_group());
                sum = sum_shared[lane];
                for (int offset = 16; offset; offset >>= 1) sum += sycl::permute_group_by_xor(sg, sum, offset);
#pragma unroll
                for (int i0 = 0; i0 < 256; i0 += 128) {
                    float result = 0.0f;
#pragma unroll
                    for (int w = 0; w < 4; ++w)
#pragma unroll
                        for (int group = 0; group < 4; ++group) result += tile[w * 4 * 256 + group * 256 + i0 + tid];
                    out[head * 256 + i0 + tid] = im::fdiv_rn(result, sum);
                }
            });
        });
    } catch (const sycl::exception& e) {
        throw std::runtime_error(std::string("native FlashAttention launch: ") + e.what());
    }
    const auto result = cudaGetLastError();
    if (result != cudaSuccess) throw std::runtime_error(std::string("native FlashAttention launch: ") + cudaGetErrorString(result));
}

}  // namespace strata::kernels
