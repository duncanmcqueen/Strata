// src/kernels/sycl/s2_gemv.cpp - SYCL port of src/kernels/cuda/s2_gemv.cu,
// P2.S2's naive S2 GEMV.  The CUDA file's header carries the contract: dequantize
// on the fly, FP32 accumulation inside the row, `(code - 1)` in the INTEGER domain
// before the group scale and the activation.  The parity test allows a 1e-5
// relative error precisely because contraction may reorder the roundings, so the
// arithmetic is transcribed as written and no rounding call is pinned.
//
// What SYCL forces to change:
//   * `__half2float(__ushort_as_half(x))` -> `f32_from_f16` (f16_bits.hpp): exact,
//     already the SYCL backend's fp16 decode (elementwise.cpp).
//   * grid of 128-thread blocks, one thread per row -> a flat parallel_for over
//     `n_out` work-items, one work-item per row (same item -> row mapping as
//     dequant_s2.cpp's flat launch).
#include "strata/kernels/s2_gemv.hpp"

#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK = 64;

void s2_gemv_row(long long o, const uint16_t* x, const uint8_t* codes, const float* scales, float* y,
                 long long n_in, long long n_out) {
    if (o >= n_out) return;

    const long long nb = n_in / QK;
    const uint8_t* c = codes + o * nb * (QK / 4);
    const float* s = scales + o * nb;

    float acc = 0.0f;
    for (long long b = 0; b < nb; ++b) {
        const float d = s[b];
        const uint8_t* cb = c + b * (QK / 4);
        const uint16_t* xb = x + b * QK;
        for (int j = 0; j < QK; ++j) {
            // (code - 1) in the INTEGER domain, then the group scale, then the activation - same order as the
            // CUDA kernel and the CPU reference; they differ only by floating-point contraction.
            const int code = (cb[j >> 2] >> ((j & 3) * 2)) & 0x03;
            acc += (float) (code - 1) * d * f32_from_f16(xb[j]);
        }
    }
    y[o] = acc;
}

}  // namespace

void s2_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
             int64_t n_out) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK != 0) {
        std::fprintf(stderr, "s2_gemv: n_in %lld is not a multiple of %d\n", (long long) n_in, QK);
        std::exit(1);
    }
    sycl::queue& q = sycl_runtime::queue_from_stream(nullptr);
    q.parallel_for(sycl::range<1>(static_cast<size_t>(n_out)), [=](sycl::id<1> idx) {
        s2_gemv_row(static_cast<long long>(idx[0]), x, codes, scales, y, n_in, n_out);
    });
    // The CUDA file synchronises with cudaDeviceSynchronize before returning; same contract here.
    q.wait();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "s2_gemv: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
