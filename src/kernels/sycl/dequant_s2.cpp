// src/kernels/sycl/dequant_s2.cpp - SYCL port of src/kernels/cuda/dequant_s2.cu,
// P2.S2's first naive kernel.  The header comment of the CUDA file carries the
// format contract (codes packed 4 per byte LSB-first, one FP32 scale per group of
// 64, `(code - 1) * scale` with the subtraction in the INTEGER domain, which is what
// makes it bit-exact).  One work-item per block, as naive as the original on
// purpose: the parity test (dequant_s2_parity.cpp, unchanged) is what says whether
// any future optimization is still right.
#include "strata/kernels/dequant_s2.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK = 64;             // S2 group = Q2_0 block = 64 elements
constexpr int CODES_PER_BYTE = 4;

void dequant_s2_block(long long b, const uint8_t* codes, const float* scales, float* out, long long n_blocks) {
    if (b >= n_blocks) return;
    const float d = scales[b];
    const uint8_t* c = codes + b * (QK / CODES_PER_BYTE);
    float* y = out + b * QK;
    for (int j = 0; j < QK; ++j) {
        const int code = (c[j / CODES_PER_BYTE] >> ((j % CODES_PER_BYTE) * 2)) & 0x03;
        y[j] = (float)(code - 1) * d;      // the -1 is applied to the CODE, then ONE multiply
    }
}

}  // namespace

void dequant_s2(const uint8_t* codes, const float* scales, float* out, int64_t n_blocks) {
    if (n_blocks <= 0) return;
    sycl::queue& q = sycl_runtime::queue_from_stream(nullptr);
    q.parallel_for(sycl::range<1>(static_cast<size_t>(n_blocks)), [=](sycl::id<1> idx) {
        dequant_s2_block(static_cast<long long>(idx[0]), codes, scales, out, n_blocks);
    });
    // The contract: synchronised before returning, so a caller can read `out` immediately.
    q.wait();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "dequant_s2: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace strata::kernels
