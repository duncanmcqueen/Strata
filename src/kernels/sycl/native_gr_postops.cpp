// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{dsv4-hc.cu,scale.cu,unary.cu,unary.cuh}.
//
// MIT License
// Copyright (c) 2023-2026 The ggml authors
//
// Permission is hereby granted, free of charge, to any person obtaining a copy
// of this software and associated documentation files (the "Software"), to deal
// in the Software without restriction, including without limitation the rights
// to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
// copies of the Software, and to permit persons to whom the Software is
// furnished to do so, subject to the following conditions:
//
// The above copyright notice and this permission notice shall be included in
// all copies or substantial portions of the Software.
//
// THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
// IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
// FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
// AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
// LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
// OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
// SOFTWARE.

#include "strata/kernels/native_gr_postops.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <limits>
#include <stdexcept>
#include <string>
#include <sycl/ext/intel/math.hpp>

namespace strata::kernels {
namespace {
constexpr int THREADS = 256;
inline float sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
// ggml SCALE uses scale*x+bias, including its +0 bias. Retain this operation
// explicitly so compile-time zero does not change signed-zero behavior.
inline float scale_zero_bias(float x, float scale) { return sycl::fma(scale, x, 0.0f); }
void down_silu(sycl::queue &queue, size_t groups, size_t threads, float *lo, int count, float scale) {
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(groups * threads, threads),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                             const std::size_t i =
                                 std::size_t(it.get_group(0)) * it.get_local_range(0) + it.get_local_id(0);
                             if (i >= std::size_t(count))
                                 return;
                             const float x = scale_zero_bias(lo[i], scale);
                             lo[i] = x / (1.0f + sycl::exp(-x));
                         });
    });
}
template <bool Fused>
void pre_gated(sycl::queue &queue, size_t groups, size_t threads, const float *__restrict__ xn,
               float *__restrict__ gate, float *__restrict__ mixed, int n_embd, int hc, float scale) {
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(groups * threads, threads),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                             const std::size_t d =
                                 std::size_t(it.get_group(0)) * it.get_local_range(0) + it.get_local_id(0);
                             if (d >= std::size_t(n_embd))
                                 return;
                             float sum = 0.0f;
                             for (int c = 0; c < hc; ++c) {
                                 const std::size_t i = std::size_t(c) * n_embd + d;
                                 const float x = xn[i], w = sigmoid(gate[i]);
                                 const float product = (x * w);
                                 gate[i] = product;
                                 if constexpr (Fused)
                                     sum = sycl::fma(x, w, sum);
                                 else
                                     sum = c == 0 ? product : (sum + product);
                             }
                             if constexpr (Fused)
                                 mixed[d] = scale * sum;
                             else
                                 mixed[d] = scale_zero_bias(sum, scale);
                         });
    });
}
void post(sycl::queue &queue, size_t groups, size_t threads, const float *residual,
          const float *__restrict__ block_out, const float *__restrict__ inject, float *output, int n_embd,
          int hc, float scale) {
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(groups * threads, threads),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                             const std::size_t i =
                                 std::size_t(it.get_group(0)) * it.get_local_range(0) + it.get_local_id(0);
                             if (i >= std::size_t(n_embd) * hc)
                                 return;
                             const int c = int(i / n_embd), d = int(i % n_embd);
                             const float weight =
                                 scale_zero_bias(sigmoid(scale_zero_bias(inject[c], scale)), 2.0f);
                             // Exact residual/output alias is supported; no other thread reads
                             // residual[i].
                             output[i] = sycl::fma(block_out[d], weight, residual[i]);
                         });
    });
}
void check_pointer(const void *p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float))
        throw std::invalid_argument("native GR postops require non-null four-byte aligned pointers");
}
void check_shape(int n, int hc) {
    if (n <= 0 || hc <= 0 || std::uint64_t(n) * hc > std::uint64_t(std::numeric_limits<int>::max()))
        throw std::invalid_argument("native GR postops require positive bounded dimensions");
}
unsigned blocks(std::size_t n) { return unsigned((n + THREADS - 1) / THREADS); }
void check_launch() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("native GR postops launch: ") + cudaGetErrorString(error));
}
} // namespace

void native_gr_down_silu(float *lo, int hc_lr, int hc, void *stream) {
    check_shape(hc_lr, hc);
    check_pointer(lo);
    down_silu(sycl_runtime::queue_from_stream(stream), blocks(hc_lr), THREADS, lo, hc_lr, 1.0f / float(hc));
    check_launch();
}
void native_gr_pre_gated(const float *xn, float *gate, float *mixed, int n_embd, int hc, bool fused_layer,
                         void *stream) {
    check_shape(n_embd, hc);
    check_pointer(xn);
    check_pointer(gate);
    check_pointer(mixed);
    if (fused_layer)
        pre_gated<true>(sycl_runtime::queue_from_stream(stream), blocks(n_embd), THREADS, xn, gate, mixed,
                        n_embd, hc, 1.0f / float(hc));
    else
        pre_gated<false>(sycl_runtime::queue_from_stream(stream), blocks(n_embd), THREADS, xn, gate, mixed,
                         n_embd, hc, 1.0f / float(hc));
    check_launch();
}
void native_gr_post(const float *residual, const float *block_out, const float *inject, float *output,
                    int n_embd, int hc, void *stream) {
    check_shape(n_embd, hc);
    check_pointer(residual);
    check_pointer(block_out);
    check_pointer(inject);
    check_pointer(output);
    post(sycl_runtime::queue_from_stream(stream), blocks(std::size_t(n_embd) * hc), THREADS, residual,
         block_out, inject, output, n_embd, hc, 1.0f / float(hc));
    check_launch();
}
} // namespace strata::kernels
