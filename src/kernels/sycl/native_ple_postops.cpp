// src/kernels/sycl/native_ple_postops.cpp - SYCL port of src/kernels/cuda/native_ple_postops.cu; see
// include/strata/kernels/native_ple_postops.hpp.
//
// Arithmetic adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// src/models/qwen4exp.cpp and ggml-cuda/{reduce_rows.cuh,sumrows.cu,unary.cu}.
// MIT License, Copyright (c) 2023-2026 The ggml authors (full text in the CUDA file).
//
// What SYCL forces to change:
//   * the row norms are native_gr_norm.cpp's kernel (sycl_detail::native_rms_norm_rows).  The batch's repeating
//     gamma is that kernel's gamma row period, so a batch row and a single token's row run the SAME kernel - the
//     CUDA file's separate rms_rep_kernel is a copy of the GR norm, and on this backend a copy could contract
//     differently.
//   * the gate kernel serves both paths (H or T*H rows), as in the CUDA file.
//   * the conv tap, the SiLU and the residual are one helper used by both paths; the CUDA `__fmul_rn`/`__fadd_rn`
//     become `fmul_rn`/`fadd_rn`, the SiLU's variable division `fdiv_rn`, `expf` `sycl::exp`, and the gate's
//     `sqrtf` `fsqrt_rn` (IGC's default float division/sqrt are not correctly rounded).
//   * the gate's 512-thread SUM_ROWS reduction keeps its eight partial lanes, the product rounded on its own and
//     the two xor trees; its sums are pinned with `fadd_rn`.
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "fp_exact.hpp"
#include "native_rms_rows.hpp"

#include <cuda_runtime.h>
#include <sycl/ext/intel/math.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

namespace im = strata::kernels::fp_exact;

constexpr int N = 2560, H = 4, D = N * H, HISTORY = 9;
static_assert(N == NG_N_EMBD && H == NG_HC && HISTORY == NG_HIST, "native PLE geometry changed");

inline float warp_sum(const sycl::sub_group& sg, float x) {
    for (int offset = 16; offset; offset >>= 1) x = im::fadd_rn(x, sycl::permute_group_by_xor(sg, x, offset));
    return x;
}

// gate[row] for `rows` rows of N: the CUDA gate_kernel (one 512-thread work-group per row)
void gate_launch(sycl::queue& q, const float* key, const float* query, float* gate, float scale, size_t rows) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partials(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>(rows * 512, 512), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int tid = (int) it.get_local_id(0);
            const size_t row = it.get_group(0);
            float sums[8] = {};
#pragma unroll
            for (int j = 0; j < 8; ++j) {
                const int d = tid + j * 512;
                const float p = d < N ? im::fmul_rn(key[row * N + d], query[row * N + d]) : 0.0f;
                sums[j] = im::fadd_rn(sums[j], p);
            }
            float sum = 0;
#pragma unroll
            for (int j = 0; j < 8; ++j) sum = im::fadd_rn(sum, sums[j]);
            const sycl::sub_group sg = it.get_sub_group();
            sum = warp_sum(sg, sum);
            const int lane = tid % 32;
            if (!lane) partials[tid / 32] = sum;
            sycl::group_barrier(it.get_group());
            sum = lane < 16 ? partials[lane] : 0.0f;
            sum = warp_sum(sg, sum);
            if (tid == 0) {
                const float s = im::fmaf_rn(scale, sum, 0.0f);   // ggml SCALE's zero bias
                const float mag = im::fsqrt_rn(sycl::fmax(sycl::fabs(s), 1e-6f));
                const float sign = float((s > 0.0f) - (s < 0.0f));
                gate[row] = im::fdiv_rn(1.0f, im::fadd_rn(1.0f, sycl::exp(-im::fmul_rn(sign, mag))));
            }
        });
    });
}

// one channel's dilated conv (taps x[0..3], oldest first), SiLU, and the residual
struct ConvOut { float activation, result; };
inline ConvOut conv_residual(const float (&x)[4], const uint16_t* weights, int c, float hidden, float gated) {
    float sum = 0;
#pragma unroll
    for (int k = 0; k < 4; ++k) {
        const float term = im::fmul_rn(x[k], f32_from_f16(weights[c * 4 + k]));
        sum = k == 0 ? term : im::fadd_rn(sum, term);
    }
    const float activation = im::fdiv_rn(sum, im::fadd_rn(1.0f, sycl::exp(-sum)));
    return {activation, im::fadd_rn(hidden, im::fadd_rn(gated, activation))};
}

struct Span { const void* p; size_t bytes; size_t alignment; };
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.bytes && y < x + a.bytes;
}
void validate(Span span) {
    const auto p = reinterpret_cast<uintptr_t>(span.p);
    if (!p || p % span.alignment || p > std::numeric_limits<uintptr_t>::max() - span.bytes)
        throw std::invalid_argument("native PLE postops require nonnull aligned bounded spans");
}
void launch_check() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string("native PLE postops launch: ") + cudaGetErrorString(error));
}
}  // namespace

void native_ple_postops(const float* projected_key, const float* hidden, const float* value, const float* history,
                        const PleWeights& w, const NativePlePostopsBuffers& b, void* stream) {
    if (!stream) throw std::invalid_argument("native PLE postops require an explicit stream");
    const Span inputs[] = {{projected_key, D * 4, 4}, {hidden, D * 4, 4}, {value, N * 4, 4},
                           {history, HISTORY * D * 4, 4}, {w.norm_key, D * 4, 4}, {w.norm_query, D * 4, 4},
                           {w.norm_conv, D * 4, 4}, {w.conv1d_f16, 4 * D * 2, 2}};
    const Span outputs[] = {{b.key, D * 4, 4}, {b.query, D * 4, 4}, {b.gate, H * 4, 4}, {b.gated, D * 4, 4},
                            {b.normalized, D * 4, 4}, {b.conv, D * 4, 4}, {b.result, D * 4, 4}};
    for (const auto& span : inputs) validate(span);
    for (const auto& span : outputs) validate(span);
    for (size_t i = 0; i < 7; ++i) {
        for (size_t j = 0; j < 8; ++j)
            if (!(i == 6 && j == 1 && b.result == hidden) && overlaps(outputs[i], inputs[j]))
                throw std::invalid_argument("native PLE postops output overlaps an input or weight");
        for (size_t j = i + 1; j < 7; ++j)
            if (!(i == 1 && j == 4 && b.query == b.normalized) && overlaps(outputs[i], outputs[j]))
                throw std::invalid_argument("native PLE postops writable spans overlap");
    }
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    sycl_detail::native_rms_norm_rows(projected_key, w.norm_key, b.key, N, H, H, NG_RMS_EPS, stream);
    sycl_detail::native_rms_norm_rows(hidden, w.norm_query, b.query, N, H, H, NG_RMS_EPS, stream);
    gate_launch(q, b.key, b.query, b.gate, 1.0f / std::sqrt(float(N)), H);
    float* gated = b.gated;
    const float* gate = b.gate;
    q.parallel_for(sycl::range<1>(D), [=](sycl::id<1> i) { gated[i] = im::fmul_rn(value[i % N], gate[i / N]); });
    launch_check();
    sycl_detail::native_rms_norm_rows(b.gated, w.norm_conv, b.normalized, N, H, H, NG_RMS_EPS, stream);
    const float* normalized = b.normalized;
    const uint16_t* weights = w.conv1d_f16;
    float* conv = b.conv;
    float* result = b.result;
    q.parallel_for(sycl::range<1>(D), [=](sycl::id<1> id) {
        const int c = (int) id[0];
        float x[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) x[k] = k == 3 ? normalized[c] : history[c * HISTORY + 3 * k];
        const ConvOut o = conv_residual(x, weights, c, hidden[c], gated[c]);
        conv[c] = o.activation;
        result[c] = o.result;   // an exact hidden/result alias is safe: each work-item owns one element
    });
    launch_check();
}

void native_ple_postops_batch(float* key, float* hidden, const float* value, float* history, const PleWeights& w,
                              float* query_norm, float* gated, float* gate, int T, void* stream) {
    if (!stream || T <= 0 || !key || !hidden || !value || !history || !query_norm || !gated || !gate)
        throw std::invalid_argument("native PLE postops batch: null input or empty batch");
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    const int rows = T * H;
    const size_t n = size_t(T) * D;
    sycl_detail::native_rms_norm_rows(key, w.norm_key, key, N, rows, H, NG_RMS_EPS, stream);
    sycl_detail::native_rms_norm_rows(hidden, w.norm_query, query_norm, N, rows, H, NG_RMS_EPS, stream);
    gate_launch(q, key, query_norm, gate, 1.0f / std::sqrt(float(N)), (size_t) rows);
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        const size_t i = id[0], t = i / D, d = i % D;
        gated[i] = im::fmul_rn(value[t * N + d % N], gate[t * H + d / N]);
    });
    sycl_detail::native_rms_norm_rows(gated, w.norm_conv, query_norm, N, rows, H, NG_RMS_EPS, stream);
    const float* normalized = query_norm;
    const uint16_t* weights = w.conv1d_f16;
    // the dilated conv (taps 9, 6, 3 tokens back and this one) and the residual; a tap before the chunk reads
    // the history
    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
        const size_t i = id[0];
        const int t = int(i / D), c = int(i % D);
        float x[4];
#pragma unroll
        for (int k = 0; k < 4; ++k) {
            const int p = t - 9 + 3 * k;
            x[k] = p >= 0 ? normalized[size_t(p) * D + c] : history[size_t(c) * HISTORY + (9 + p)];
        }
        hidden[i] = conv_residual(x, weights, c, hidden[i], gated[i]).result;
    });
    // the history after the chunk: the last nine normalized rows (older ones from the history when T < 9)
    q.parallel_for(sycl::range<1>(D), [=](sycl::id<1> id) {
        const int c = (int) id[0];
        float h[HISTORY];
#pragma unroll
        for (int r = 0; r < HISTORY; ++r) {
            const int p = T - HISTORY + r;
            h[r] = p >= 0 ? normalized[size_t(p) * D + c] : history[size_t(c) * HISTORY + (T + r)];
        }
#pragma unroll
        for (int r = 0; r < HISTORY; ++r) history[size_t(c) * HISTORY + r] = h[r];
    });
    launch_check();
}

}  // namespace strata::kernels
