// src/kernels/sycl/native_gdn_preprocess.cpp - SYCL port of src/kernels/cuda/native_gdn_preprocess.cu; see
// include/strata/kernels/native_gdn_preprocess.hpp.  Pinned llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d
// arithmetic (MIT License, Copyright (c) 2023-2026 The ggml authors; full text in the CUDA file).
//
// The rounding boundaries the CUDA file pins (`__fadd_rn`, `__fmul_rn`, `__fmaf_rn`) are `fadd_rn`/`fmul_rn`/
// `fmaf_rn`; `expf`/`log1pf` are `sycl::exp`/`sycl::log1p`; the 256-thread norm reduction keeps its two xor trees.
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "fp_exact.hpp"

#include <cuda_runtime.h>
#include <sycl/ext/intel/math.hpp>

#include <cmath>
#include <cstddef>
#include <cstdint>
#include <initializer_list>
#include <stdexcept>

namespace strata::kernels {
namespace {

namespace im = strata::kernels::fp_exact;
constexpr int S = 128;

inline float warp_sum(const sycl::sub_group& sg, float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) value += sycl::permute_group_by_xor(sg, value, offset);
    return value;
}
inline float norm_sum(sycl::nd_item<1> it, float value, float* sums) {
    const int tid = (int) it.get_local_id(0), lane = tid % 32;
    const sycl::sub_group sg = it.get_sub_group();
    value = warp_sum(sg, value);
    if (lane == 0) sums[tid / 32] = value;
    sycl::group_barrier(it.get_group());
    value = lane < 8 ? sums[lane] : 0.0f;
    return warp_sum(sg, value);
}
inline float sigmoid(float value) { return 1.0f / (1.0f + sycl::exp(-value)); }

// one 256-wide work-group per row of 128
template<class F>
void rows_launch(void* stream, int64_t rows, F body) {
    sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> sums(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) rows * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            body(it, &sums[0]);
        });
    });
}

struct Span { const void* pointer; size_t bytes; };
void valid(Span span) {
    const auto address = reinterpret_cast<uintptr_t>(span.pointer);
    if (!span.pointer || address % sizeof(float) || span.bytes > UINTPTR_MAX - address)
        throw std::invalid_argument("native GDN preprocessing requires aligned nonnull valid spans");
}
void disjoint(Span a, Span b) {
    const auto ap = reinterpret_cast<uintptr_t>(a.pointer), bp = reinterpret_cast<uintptr_t>(b.pointer);
    if (ap < bp + b.bytes && bp < ap + a.bytes)
        throw std::invalid_argument("native GDN preprocessing requires disjoint writable spans");
}
void count_and_stream(int64_t count, void* stream) {
    if (!stream || count <= 0 || count > 65535)
        throw std::invalid_argument("native GDN preprocessing requires a stream and count in [1,65535]");
}
void norm_geometry(int64_t rows, int64_t cols, float epsilon, void* stream) {
    count_and_stream(rows, stream);
    if (cols != S || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GDN norm requires width 128 and finite nonnegative epsilon");
}
void check_launch() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}
}  // namespace

void native_gdn_conv_silu(float* history, const float* input, const float* weights, float* raw_output,
                          float* silu_output, int64_t channels, int64_t d_conv, void* stream) {
    count_and_stream(channels, stream);
    if (d_conv != 4) throw std::invalid_argument("native GDN convolution requires four taps");
    const size_t bytes = size_t(channels) * sizeof(float);
    const Span writable[] = {{history, 3 * bytes}, {raw_output, bytes}, {silu_output, bytes}};
    const Span inputs[] = {{input, bytes}, {weights, 4 * bytes}};
    for (auto span : writable) valid(span);
    for (auto span : inputs) valid(span);
    for (int i = 0; i < 3; ++i) {
        for (int j = 0; j < i; ++j) disjoint(writable[i], writable[j]);
        for (auto span : inputs) disjoint(writable[i], span);
    }
    sycl_runtime::queue_from_stream(stream).parallel_for(sycl::range<1>((size_t) channels), [=](sycl::id<1> id) {
        const int c = (int) id[0];
        float values[4] = {history[c * 3], history[c * 3 + 1], history[c * 3 + 2], input[c]};
        float sum = 0.0f;
#pragma unroll
        for (int tap = 0; tap < 4; ++tap) sum += values[tap] * weights[c * 4 + tap];
        sum = im::fadd_rn(sum, 0.0f);   // the native SSM kernel's zero bias, added even without a bias input
        raw_output[c] = sum;
        silu_output[c] = sum / (1.0f + sycl::exp(-sum));
#pragma unroll
        for (int tap = 0; tap < 3; ++tap) history[c * 3 + tap] = values[tap + 1];
    });
    check_launch();
}

void native_gdn_l2_norm(float* input, int64_t rows, int64_t cols, float epsilon, void* stream) {
    norm_geometry(rows, cols, epsilon, stream);
    valid({input, size_t(rows) * S * sizeof(float)});
    const float eps = epsilon / S, scale_after = 1.0f / std::sqrt(float(S));
    rows_launch(stream, rows, [=](sycl::nd_item<1> it, float* sums) {
        const int col = (int) it.get_local_id(0);
        float* row = input + size_t(it.get_group(0)) * S;
        const float value = col < S ? row[col] : 0.0f;
        float partial = 0.0f;
        if (col < S) partial += value * value;
        partial = norm_sum(it, partial, sums);
        const float scale = sycl::rsqrt(partial / S + eps);
        if (col < S) {
            // the FP32 store boundary between RMSNorm and ggml_scale
            const float normalized = im::fmul_rn(scale, value);
            row[col] = im::fmaf_rn(normalized, scale_after, 0.0f);
        }
    });
    check_launch();
}

void native_gdn_beta_gate(float* beta, int64_t heads, void* stream) {
    count_and_stream(heads, stream);
    valid({beta, size_t(heads) * sizeof(float)});
    sycl_runtime::queue_from_stream(stream).parallel_for(sycl::range<1>((size_t) heads),
                                                         [=](sycl::id<1> i) { beta[i] = sigmoid(beta[i]); });
    check_launch();
}

void native_gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t heads,
                     void* stream) {
    count_and_stream(heads, stream);
    const size_t bytes = size_t(heads) * sizeof(float);
    const Span output{gate, bytes};
    valid(output);
    for (auto input : {Span{alpha, bytes}, Span{dt, bytes}, Span{ssm_a, bytes}}) {
        valid(input);
        disjoint(output, input);
    }
    sycl_runtime::queue_from_stream(stream).parallel_for(sycl::range<1>((size_t) heads), [=](sycl::id<1> i) {
        const float value = im::fadd_rn(alpha[i], dt[i]);
        const float softplus = value > 20.0f ? value : sycl::log1p(sycl::exp(value));   // 1 + e^v loses e^v below ~1e-7
        gate[i] = softplus * ssm_a[i];
    });
    check_launch();
}

void native_gdn_out_norm(const float* output, const float* z, const float* gamma, float* destination, int64_t heads,
                         int64_t cols, float epsilon, void* stream) {
    norm_geometry(heads, cols, epsilon, stream);
    const size_t bytes = size_t(heads) * S * sizeof(float);
    const Span writable{destination, bytes};
    valid(writable);
    for (auto input : {Span{output, bytes}, Span{z, bytes}, Span{gamma, S * sizeof(float)}}) {
        valid(input);
        disjoint(writable, input);
    }
    rows_launch(stream, heads, [=](sycl::nd_item<1> it, float* sums) {
        const int col = (int) it.get_local_id(0);
        const size_t offset = size_t(it.get_group(0)) * S;
        const float value = col < S ? output[offset + col] : 0.0f;
        float partial = 0.0f;
        if (col < S) partial += value * value;
        partial = norm_sum(it, partial, sums);
        const float scale = sycl::rsqrt(partial / S + epsilon);
        if (col < S) {
            // RMSNorm + gamma is one pinned fused operator, followed by sigmoid * mul
            const float weighted = im::fmul_rn(im::fmul_rn(scale, value), gamma[col]);
            destination[offset + col] = weighted * sigmoid(z[offset + col]);
        }
    });
    check_launch();
}

}  // namespace strata::kernels
