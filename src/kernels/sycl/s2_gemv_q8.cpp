// src/kernels/sycl/s2_gemv_q8.cpp - SYCL port of src/kernels/cuda/s2_gemv_q8.cu (the S2 GEMV over Q8_0
// activations), plus the two sibling S2 kernels `s_gemv_parity` links against: `s2_gemv_quads`
// (cuda/s2_gemv_quads.cu) and `s2_gemv_fast` (cuda/s2_gemv_fast.cu).  They live here rather than in their own
// files so the three-file port is self-contained; all three are declared in `s_gemv.hpp`.
//
// The CUDA files' comments carry the contract (the activation is dequantized in the loop, `xq * dx`, as ggml's
// vec_dot_type conversion; a quad lies inside one 32-element block so the block scale is loaded once; the
// group index is `q >> 4` as a shift).  Only the launch mechanics change:
//
//   * `extern __shared__ float partial[]` -> a local accessor sized threads_per_row floats.
//   * `__shfl`-free block tree reduction -> `sycl::group_barrier` between the same shared-memory steps.
//   * `__constant__ float c_codes[256][4]` (s2_gemv_fast) -> a lazily built USM device buffer, uploaded once
//     per process exactly as `ensure_lut` uploaded once per device.
//   * `__half2float` -> `f32_from_f16` (exact; see f16_bits.hpp).
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int QK_S2 = 64;
constexpr int QK8_0 = 32;

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void sync_checked(sycl::queue& q, const char* what) {
    q.wait();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void s2_gemv_q8_row(sycl::nd_item<1> it, float* partial, const uint8_t* act, const uint8_t* codes,
                    const float* scales, float* y, long long n_in, long long n_out, int threads_per_row) {
    const long long o = it.get_group(0);
    if (o >= n_out) return;
    const int tid = (int) it.get_local_id(0);

    const long long n_quads = n_in / 4;
    const uint8_t* c = codes + o * n_quads;             // one code byte per quad
    const float* s = scales + o * (n_in / QK_S2);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long long q = tid; q < n_quads; q += threads_per_row) {
        const uint8_t byte = c[q];
        const float d = s[q >> 4];                      // (q*4) >> 6, the S2 group index as a shift
        // the activation quad: four int8 in one 32-element block, so one block scale
        const long long ablk = (q * 4) / QK8_0;
        const uint8_t* blk = act + ablk * 34;
        const uint16_t dbits = (uint16_t) (blk[0] | (blk[1] << 8));
        const float dx = f32_from_f16(dbits);
        const int8_t* xq = reinterpret_cast<const int8_t*>(blk + 2);
        const int off = (int) ((q * 4) % QK8_0);

        const float w0 = (float) ((int) (byte & 3) - 1) * d;
        const float w1 = (float) ((int) ((byte >> 2) & 3) - 1) * d;
        const float w2 = (float) ((int) ((byte >> 4) & 3) - 1) * d;
        const float w3 = (float) ((int) ((byte >> 6) & 3) - 1) * d;
        a0 += w0 * ((float) xq[off + 0] * dx);
        a1 += w1 * ((float) xq[off + 1] * dx);
        a2 += w2 * ((float) xq[off + 2] * dx);
        a3 += w3 * ((float) xq[off + 3] * dx);
    }
    partial[tid] = (a0 + a1) + (a2 + a3);
    sycl::group_barrier(it.get_group());
    for (int step = threads_per_row / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        sycl::group_barrier(it.get_group());
    }
    if (tid == 0) y[o] = partial[0];
}

void s2_gemv_quads_row(sycl::nd_item<1> it, float* partial, const uint16_t* x, const uint8_t* codes,
                       const float* scales, float* y, long long n_in, long long n_out, int threads_per_row) {
    const long long o = it.get_group(0);
    if (o >= n_out) return;
    const int tid = (int) it.get_local_id(0);

    const long long n_quads = n_in / 4;
    const uint8_t* c = codes + o * n_quads;             // exactly one code byte per quad
    const float* s = scales + o * (n_in / QK_S2);

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long long q = tid; q < n_quads; q += threads_per_row) {
        const uint8_t byte = c[q];                      // ONE load for four codes
        const float d = s[q >> 4];                      // (q*4) >> 6, the group index as a shift
        // ONE 64-bit load for four halves (two 4-byte words).  `x` is 256-byte aligned and four halves are
        // eight bytes.
        const uint32_t xw0 = *reinterpret_cast<const uint32_t*>(x + q * 4);
        const uint32_t xw1 = *reinterpret_cast<const uint32_t*>(x + q * 4 + 2);
        // the four codes, each carrying the -1 bias in the INTEGER domain; the scale is applied once per
        // element here rather than once per group, which is the same expression the generic kernel uses
        const float w0 = (float) ((int) (byte & 3) - 1) * d;
        const float w1 = (float) ((int) ((byte >> 2) & 3) - 1) * d;
        const float w2 = (float) ((int) ((byte >> 4) & 3) - 1) * d;
        const float w3 = (float) ((int) ((byte >> 6) & 3) - 1) * d;
        a0 += w0 * f32_from_f16((uint16_t) xw0);
        a1 += w1 * f32_from_f16((uint16_t) (xw0 >> 16));
        a2 += w2 * f32_from_f16((uint16_t) xw1);
        a3 += w3 * f32_from_f16((uint16_t) (xw1 >> 16));
    }
    partial[tid] = (a0 + a1) + (a2 + a3);
    sycl::group_barrier(it.get_group());
    for (int step = threads_per_row / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        sycl::group_barrier(it.get_group());
    }
    if (tid == 0) y[o] = partial[0];
}

// ---- s2_gemv_fast: the two-lever experiment (kept for its negative result - see the CUDA file) -----------

constexpr int MAX_SHARED_HALVES = 4096;      // 8 KB of local memory for x; n_embd 2560 fits with room

// The 256x4 code table with the -1 bias applied, in device USM: the `__constant__ c_codes` replacement.
// Built once per process, as `ensure_lut` built it once per device; the bench that uses this entry point is
// never stream-captured, so the first-call allocation cannot violate the capture rules.
float* g_code_lut = nullptr;

void ensure_lut(sycl::queue& q) {
    if (g_code_lut != nullptr) return;
    float host[256][4];
    for (int b = 0; b < 256; ++b)
        for (int k = 0; k < 4; ++k) host[b][k] = (float) (((b >> (2 * k)) & 3) - 1);
    g_code_lut = sycl::malloc_device<float>(256 * 4, q);
    if (g_code_lut == nullptr) {
        std::fprintf(stderr, "s2_gemv_fast: could not allocate the code table\n");
        std::exit(1);
    }
    q.memcpy(g_code_lut, host, sizeof(host)).wait();
}

template <bool STAGE_X>
void s2_gemv_fast_row(sycl::nd_item<1> it, uint16_t* sx, float* partial, const float* lut,
                      const uint16_t* x, const uint8_t* codes, const float* scales, float* y, long long n_in,
                      long long n_out, int threads_per_row) {
    const long long o = it.get_group(0);
    if (o >= n_out) return;
    const int tid = (int) it.get_local_id(0);

    if (STAGE_X) {
        for (long long i = tid; i < n_in; i += threads_per_row) sx[i] = x[i];
        sycl::group_barrier(it.get_group());
    }

    const long long n_quads = n_in / 4;
    const uint8_t* c = codes + o * n_quads;
    const float* s = scales + o * (n_in / QK_S2);
    const uint16_t* xsrc = STAGE_X ? sx : x;

    float a0 = 0.0f, a1 = 0.0f, a2 = 0.0f, a3 = 0.0f;
    for (long long q = tid; q < n_quads; q += threads_per_row) {
        const uint8_t byte = c[q];                          // ONE load for four codes
        const float d = s[q >> 4];                          // (q*4) >> 6
        const float cv0 = lut[byte * 4 + 0];                // one broadcast row of the table
        const float cv1 = lut[byte * 4 + 1];
        const float cv2 = lut[byte * 4 + 2];
        const float cv3 = lut[byte * 4 + 3];
        const uint32_t xw0 = *reinterpret_cast<const uint32_t*>(xsrc + q * 4);
        const uint32_t xw1 = *reinterpret_cast<const uint32_t*>(xsrc + q * 4 + 2);
        a0 += cv0 * d * f32_from_f16((uint16_t) xw0);
        a1 += cv1 * d * f32_from_f16((uint16_t) (xw0 >> 16));
        a2 += cv2 * d * f32_from_f16((uint16_t) xw1);
        a3 += cv3 * d * f32_from_f16((uint16_t) (xw1 >> 16));
    }
    partial[tid] = (a0 + a1) + (a2 + a3);
    sycl::group_barrier(it.get_group());
    for (int step = threads_per_row / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        sycl::group_barrier(it.get_group());
    }
    if (tid == 0) y[o] = partial[0];
}

}  // namespace

void s2_gemv_q8(const uint8_t* act, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                int64_t n_out, int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % QK8_0 != 0 || n_in % QK_S2 != 0) {
        std::fprintf(stderr, "s2_gemv_q8: n_in %lld must be a multiple of %d\n", (long long) n_in, QK_S2);
        std::exit(1);
    }
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    const size_t tpr = (size_t) threads_per_row;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partial(sycl::range<1>(tpr), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n_out * tpr), sycl::range<1>(tpr)),
                       [=](sycl::nd_item<1> it) {
            s2_gemv_q8_row(it, &partial[0], act, codes, scales, y, n_in, n_out, threads_per_row);
        });
    });
    check_launch("s2_gemv_q8");
    if (stream == nullptr) sync_checked(q, "s2_gemv_q8");
}

void s2_gemv_quads(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                   int64_t n_out, int threads_per_row) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0) {
        std::fprintf(stderr, "s2_gemv_quads: n_in %lld is not a multiple of 4\n", (long long) n_in);
        std::exit(1);
    }
    sycl::queue& q = sycl_runtime::queue_from_stream(nullptr);
    const size_t tpr = (size_t) threads_per_row;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partial(sycl::range<1>(tpr), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>((size_t) n_out * tpr), sycl::range<1>(tpr)),
                       [=](sycl::nd_item<1> it) {
            s2_gemv_quads_row(it, &partial[0], x, codes, scales, y, n_in, n_out, threads_per_row);
        });
    });
    check_launch("s2_gemv_quads");
    sync_checked(q, "s2_gemv_quads");
}

void s2_gemv_fast(const uint16_t* x, const uint8_t* codes, const float* scales, float* y, int64_t n_in,
                  int64_t n_out, int threads_per_row, bool stage_x) {
    if (n_in <= 0 || n_out <= 0) return;
    if (n_in % 4 != 0 || n_in % QK_S2 != 0) {
        std::fprintf(stderr, "s2_gemv_fast: n_in %lld must be a multiple of %d\n", (long long) n_in, QK_S2);
        std::exit(1);
    }
    sycl::queue& q = sycl_runtime::queue_from_stream(nullptr);
    ensure_lut(q);
    if (stage_x && n_in > MAX_SHARED_HALVES) {
        std::fprintf(stderr, "s2_gemv_fast: n_in %lld exceeds the %d-half shared staging limit\n",
                     (long long) n_in, MAX_SHARED_HALVES);
        std::exit(1);
    }
    const size_t tpr = (size_t) threads_per_row;
    const float* lut = g_code_lut;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<uint16_t, 1> sx(sycl::range<1>(stage_x ? MAX_SHARED_HALVES : 1), h);
        sycl::local_accessor<float, 1> partial(sycl::range<1>(tpr), h);
        const sycl::nd_range<1> rng(sycl::range<1>((size_t) n_out * tpr), sycl::range<1>(tpr));
        if (stage_x) {
            h.parallel_for(rng, [=](sycl::nd_item<1> it) {
                s2_gemv_fast_row<true>(it, &sx[0], &partial[0], lut, x, codes, scales, y, n_in, n_out,
                                       threads_per_row);
            });
        } else {
            h.parallel_for(rng, [=](sycl::nd_item<1> it) {
                s2_gemv_fast_row<false>(it, &sx[0], &partial[0], lut, x, codes, scales, y, n_in, n_out,
                                        threads_per_row);
            });
        }
    });
    check_launch("s2_gemv_fast");
    sync_checked(q, "s2_gemv_fast");
}

}  // namespace strata::kernels
