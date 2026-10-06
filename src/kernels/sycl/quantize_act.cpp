// src/kernels/sycl/quantize_act.cpp - SYCL port of src/kernels/cuda/quantize_act.cu
// (quantize an activation the way ggml converts src1).  The CUDA file's header and
// per-kernel comments carry the numerical contract - the THREE SUBTLETIES (FP32 d32
// with integer division by d32, FLOAT64 division with rint half-to-even, dequantized
// value using the fp16 scale), the hit path's deliberate CPU rounding rule, ggml's
// nearest_int magic constant - and are not repeated here; the arithmetic below is
// line-for-line the same.  Only the launch mechanics changed (one work-item per
// block, as before).
//
// `__fmul_rn` becomes `strata::kernels::fp_exact::fmul_rn`: the point of the intrinsic
// is that the product is rounded to f32 BEFORE nearest_int's magic-constant add, and
// a compiler-contracted FMA would skip that rounding (see the CUDA file's comment).
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "fp_exact.hpp"

#include <cuda_runtime.h>

#include <sycl/ext/intel/math.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

constexpr int QK8_0 = 32;

void quantize_q8_0_block(long long b, const float* x, uint8_t* blocks, long long n_blocks) {
    if (b >= n_blocks) return;
    const float* xb = x + b * QK8_0;
    uint8_t* out = blocks + b * 34;                 // { fp16 d ; int8 qs[32] }

    float amax = 0.0f;
    for (int i = 0; i < QK8_0; ++i) amax = sycl::fmax(amax, sycl::fabs(xb[i]));
    if (amax == 0.0f) {
        // ggml leaves the block zeroed: d = 0 and every q = 0.
        const uint16_t zb = f16_from_f32(0.0f);
        out[0] = (uint8_t) (zb & 0xFF);
        out[1] = (uint8_t) (zb >> 8);
        for (int i = 0; i < QK8_0; ++i) out[2 + i] = 0;
        return;
    }
    const float d32 = amax / 127.0f;
    const uint16_t d16bits = f16_from_f32(d32);
    out[0] = (uint8_t) (d16bits & 0xFF);
    out[1] = (uint8_t) (d16bits >> 8);

    // double division, matching the reference exactly (subtlety 2)
    for (int i = 0; i < QK8_0; ++i) {
        double q = sycl::rint((double) xb[i] / (double) d32);
        if (q > 127.0) q = 127.0;
        if (q < -128.0) q = -128.0;
        out[2 + i] = (uint8_t) (int8_t) q;
    }
}

/// THE HIT PATH'S QUANTIZER: reproduces `cpu/expert.cpp`'s act_quant_q8_1 exactly
/// (reciprocal multiply, round half AWAY FROM ZERO, parallel fp32 scale array with
/// d32 unrounded).  See the CUDA file for why this differs from quantize_q8_0.
void quantize_q8_0_scaled_block(long long b, const float* x, uint8_t* blocks, float* scales, long long n_blocks) {
    if (b >= n_blocks) return;
    const float* xb = x + b * QK8_0;
    uint8_t* out = blocks + b * 34;

    float amax = 0.0f;
    for (int i = 0; i < QK8_0; ++i) amax = sycl::fmax(amax, sycl::fabs(xb[i]));
    // VERBATIM from `cpu/expert.cpp:144-145`, including the `amax > 0` guard.
    // fdiv_rn for `1.f / s`: the CPU reference's division is IEEE; IGC's default
    // variable-divisor lowering is not (see quantize_q8_K_block).
    const float s = amax > 0.f ? amax / 127.f : 0.f;
    const float inv = s > 0.f ? strata::kernels::fp_exact::fdiv_rn(1.f, s) : 0.f;
    scales[b] = s;

    const uint16_t d16bits = f16_from_f32(s);
    out[0] = (uint8_t) (d16bits & 0xFF);
    out[1] = (uint8_t) (d16bits >> 8);
    for (int i = 0; i < QK8_0; ++i) {
        // VERBATIM from `cpu/expert.cpp:159-162`: reciprocal multiply, then round half away from zero.
        const float t = xb[i] * inv;
        const float r = t + (t >= 0.f ? 0.5f : -0.5f);
        int v = (int) r;
        v = v < -127 ? -127 : (v > 127 ? 127 : v);
        out[2 + i] = (uint8_t) (int8_t) v;
    }
}

void dequant_q8_0_block(long long b, const uint8_t* blocks, float* x, long long n_blocks) {
    if (b >= n_blocks) return;
    const uint8_t* blk = blocks + b * 34;
    const uint16_t dbits = (uint16_t) (blk[0] | (blk[1] << 8));
    const float d = f32_from_f16(dbits);
    float* out = x + b * QK8_0;
    for (int i = 0; i < QK8_0; ++i) out[i] = (float) (int8_t) blk[2 + i] * d;
}

// ===================== Q8_K =====================
//
// `block_q8_K` = { float d ; int8_t qs[256] ; int16_t bsums[16] } = 292 bytes, no padding
// (`static_assert` in ggml-common.h).  QK_K = 256.

constexpr int QK_K = 256;
constexpr int Q8K_BYTES = 292;

/// ggml's `nearest_int` (ggml-quants.c L621), transcribed rather than replaced: the
/// 1.5*2^23 magic constant exists precisely because its tie rule differs from rintf's.
int nearest_int_dev(float fval) {
    const float val = fval + 12582912.0f;
    int i;
    memcpy(&i, &val, 4);
    return (i & 0x007fffff) - 0x00400000;
}

void quantize_q8_K_block(long long b, const float* x, uint8_t* blocks, long long n_blocks) {
    if (b >= n_blocks) return;
    const float* xb = x + b * QK_K;
    uint8_t* out = blocks + b * Q8K_BYTES;
    float* d = (float*) out;
    int8_t* qs = (int8_t*) (out + 4);
    int16_t* bsums = (int16_t*) (out + 4 + QK_K);

    // `max` is the SIGNED value at the largest magnitude position, strictly greater,
    // so a tie keeps the FIRST maximum - what `np.argmax` does in the reference.
    float max = 0.0f, amax = 0.0f;
    for (int j = 0; j < QK_K; ++j) {
        const float ax = sycl::fabs(xb[j]);
        if (ax > amax) {
            amax = ax;
            max = xb[j];
        }
    }
    if (amax == 0.0f) {
        // ggml LEAVES bsums UNWRITTEN here; this zeroes them so a byte comparison is
        // deterministic (the CUDA file records this deliberate divergence).
        *d = 0.0f;
        for (int j = 0; j < QK_K; ++j) qs[j] = 0;
        for (int j = 0; j < QK_K / 16; ++j) bsums[j] = 0;
        return;
    }
    // fdiv_rn, NOT plain `/`: nvcc's `/` is IEEE correctly-rounded, IGC's default
    // lowering is not (measured 1 ulp off on ~28% of divisors), and both `d` and
    // every `qs` byte derive from iscale.
    const float iscale = strata::kernels::fp_exact::fdiv_rn(-127.0f, max);  // -127, NOT -128; see the header
    for (int j = 0; j < QK_K; ++j) {
        // fmul_rn, NOT `iscale * xb[j]`: the product must be rounded to f32 before
        // nearest_int's add, and a contracted FMA would skip that rounding.
        const int v = nearest_int_dev(strata::kernels::fp_exact::fmul_rn(iscale, xb[j]));
        qs[j] = (int8_t) (v > 127 ? 127 : v);      // MIN only - the source has no lower clamp
    }
    for (int j = 0; j < QK_K / 16; ++j) {
        int sum = 0;
        for (int ii = 0; ii < 16; ++ii) sum += qs[j * 16 + ii];
        bsums[j] = (int16_t) sum;
    }
    *d = strata::kernels::fp_exact::fdiv_rn(1.0f, iscale);
}

void dequant_q8_K_block(long long b, const uint8_t* blocks, float* x, long long n_blocks) {
    if (b >= n_blocks) return;
    const uint8_t* blk = blocks + b * Q8K_BYTES;
    float d;
    memcpy(&d, blk, 4);
    const int8_t* qs = (const int8_t*) (blk + 4);
    float* out = x + b * QK_K;
    for (int i = 0; i < QK_K; ++i) out[i] = (float) qs[i] * d;
}

// Shared launch: one work-item per block, the CUDA file's shape; a failed launch
// exits exactly as the CUDA launch-check did.
template <typename F>
void launch_blocks(long long n_blocks, void* stream, const char* what, F&& body) {
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    q.parallel_for(sycl::range<1>(static_cast<size_t>(n_blocks)), [=](sycl::id<1> idx) {
        body(static_cast<long long>(idx[0]));
    });
    if (stream == nullptr) q.wait();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

void quantize_q8_0(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "quantize_q8_0: n %lld is not a multiple of %d\n", (long long) n, QK8_0);
        std::exit(1);
    }
    const long long nb = n / QK8_0;
    launch_blocks(nb, stream, "quantize_q8_0",
                  [=](long long b) { quantize_q8_0_block(b, x, blocks, nb); });
}

/// See `quantize_q8_0_scaled_block`.  Writes the same 34-byte `block_q8_0` layout as `quantize_q8_0`, plus
/// `scales[n/32]` carrying the fp32 `s` the CPU path used.  `scales` must not be null.
void quantize_q8_0_scaled(const float* x, uint8_t* blocks, float* scales, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK8_0 != 0) {
        std::fprintf(stderr, "quantize_q8_0_scaled: n %lld is not a multiple of %d\n", (long long) n, QK8_0);
        std::exit(1);
    }
    if (scales == nullptr) {
        std::fprintf(stderr, "quantize_q8_0_scaled: scales is null\n");
        std::exit(1);
    }
    const long long nb = n / QK8_0;
    launch_blocks(nb, stream, "quantize_q8_0_scaled",
                  [=](long long b) { quantize_q8_0_scaled_block(b, x, blocks, scales, nb); });
}

void dequant_q8_0(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const long long nb = n / QK8_0;
    launch_blocks(nb, stream, "dequant_q8_0",
                  [=](long long b) { dequant_q8_0_block(b, blocks, x, nb); });
}

void quantize_q8_K(const float* x, uint8_t* blocks, int64_t n, void* stream) {
    if (n <= 0) return;
    if (n % QK_K != 0) {
        std::fprintf(stderr, "quantize_q8_K: n %lld is not a multiple of %d\n", (long long) n, QK_K);
        std::exit(1);
    }
    const long long nb = n / QK_K;
    launch_blocks(nb, stream, "quantize_q8_K",
                  [=](long long b) { quantize_q8_K_block(b, x, blocks, nb); });
}

void dequant_q8_K(const uint8_t* blocks, float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    const long long nb = n / QK_K;
    launch_blocks(nb, stream, "dequant_q8_K",
                  [=](long long b) { dequant_q8_K_block(b, blocks, x, nb); });
}

}  // namespace strata::kernels
