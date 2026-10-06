// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{norm.cu,common.cuh}. Scope: contiguous weighted F32
// RMSNorm.
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

#include "strata/kernels/native_gr_norm.hpp"
#include "native_rms_rows.hpp"

#include "strata/sycl_runtime/queue_bridge.hpp"
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <cuda_runtime.h>
#include <stdexcept>
#include <string>
#include <sycl/ext/intel/math.hpp>

namespace strata::kernels {
namespace {

inline float norm_warp_sum(sycl::nd_item<1> it, float value) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += sycl::permute_group_by_xor(it.get_sub_group(), value, offset);
    }
    return value;
}

// gamma_rows: the gamma matrix's row count; row r uses gamma row r % gamma_rows (n_rows for the GR call, the
// stream count for native PLE's token batch, so a batch row is computed by the same kernel as a single token's).
template <int BlockSize>
void weighted_rms_norm(sycl::queue &queue, size_t groups, size_t threads, const float *__restrict__ input,
                       const float *__restrict__ gamma, float *__restrict__ output, int n_cols, int gamma_rows,
                       float epsilon) {
    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> sums_local(32, cgh);
        cgh.parallel_for(sycl::nd_range<1>(groups * threads, threads),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                             const int tid = it.get_local_id(0);
                             const std::size_t row_offset = std::size_t(it.get_group(0)) * n_cols;
                             const float *row_input = input + row_offset;
                             const float *row_gamma =
                                 gamma + std::size_t(it.get_group(0) % gamma_rows) * n_cols;
                             float *row_output = output + row_offset;
                             float partial = 0.0f;
                             for (int col = tid; col < n_cols; col += BlockSize) {
                                 const float value = row_input[col];
                                 partial += value * value;
                             }

                             // Pinned block_reduce<SUM,BlockSize>: every warp repeats the final
                             // XOR reduction. There is no warp-0-only broadcast or
                             // downward-shuffle tree.
                             auto *sums = &sums_local[0];
                             partial = norm_warp_sum(it, partial);
                             const int lane = tid % 32;
                             if (lane == 0)
                                 sums[tid / 32] = partial;
                             it.barrier(sycl::access::fence_space::local_space);
                             partial = 0.0f;
                             if (lane < BlockSize / 32)
                                 partial = sums[lane];
                             partial = norm_warp_sum(it, partial);

                             const float mean = partial / n_cols;
                             const float scale = sycl::rsqrt(mean + epsilon);
                             for (int col = tid; col < n_cols; col += BlockSize) {
                                 row_output[col] = scale * row_input[col] * row_gamma[col];
                             }
                         });
    });
}

void check_pointer(const void *p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % alignof(float) != 0)
        throw std::invalid_argument("native GR RMSNorm requires non-null four-byte aligned pointers");
}

} // namespace

namespace sycl_detail {
void native_rms_norm_rows(const float *input, const float *gamma, float *output, int n_cols, int n_rows,
                          int gamma_rows, float epsilon, void *stream) {
    if (n_cols <= 0 || n_rows <= 0 || gamma_rows <= 0 || !std::isfinite(epsilon) || epsilon < 0.0f)
        throw std::invalid_argument("native GR RMSNorm requires positive "
                                    "dimensions and finite nonnegative epsilon");
    check_pointer(input);
    check_pointer(gamma);
    check_pointer(output);
    if (n_cols < 1024)
        weighted_rms_norm<256>(sycl_runtime::queue_from_stream(stream), unsigned(n_rows), 256, input, gamma,
                               output, n_cols, gamma_rows, epsilon);
    else
        weighted_rms_norm<1024>(sycl_runtime::queue_from_stream(stream), unsigned(n_rows), 1024, input, gamma,
                                output, n_cols, gamma_rows, epsilon);
    const auto error = cudaGetLastError();
    if (error != cudaSuccess)
        throw std::runtime_error(std::string("native GR RMSNorm launch: ") + cudaGetErrorString(error));
}
} // namespace sycl_detail

void native_gr_rms_norm_weighted(const float *input, const float *gamma, float *output, int n_cols,
                                 int n_rows, float epsilon, void *stream) {
    sycl_detail::native_rms_norm_rows(input, gamma, output, n_cols, n_rows, n_rows, epsilon, stream);
}

} // namespace strata::kernels
