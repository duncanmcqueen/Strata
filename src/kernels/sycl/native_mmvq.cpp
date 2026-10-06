// src/kernels/sycl/native_mmvq.cpp - SYCL port of src/kernels/cuda/native_mmvq.cu; see
// include/strata/kernels/native_mmvq.hpp.
//
// Adapted from llama.cpp 3cf03257f219afbe7334045ff7c6a06ac68c627d:
// ggml/src/ggml-cuda/{quantize.cu,vecdotq.cuh,mmvq.cu,common.cuh} and ggml/src/ggml-common.h.
// MIT License, Copyright (c) 2023-2026 The ggml authors (full text in the CUDA file and third_party/ggml/LICENSE).
//
// What SYCL forces to change:
//   * ONE work-group kernel per (format, ncols, layout).  The CUDA file's ncols = 1 kernels are, as its own
//     comment on the multi-column kernel says, that kernel's EXACT layout with one column: the same lane-to-block
//     mapping, blocks per iteration, small-K rule, warp-ascending cross-warp sum and xor tree, with `apply(load())`
//     doing the dot's operations in the dot's order.  Here the single-column call is that kernel with NCOLS = 1,
//     so every column of an exact multi-column call is the single-column result by construction (iq_parity and
//     shared_expert_parity check it).
//   * `__byte_perm`, `__vsubss4` are written out on 32-bit words; the IQ4_NL two-stage table lookup becomes four
//     byte lookups into a constexpr table (the same words: low nibbles in .x, high nibbles in .y).
//   * every float product and sum goes through `fmul_rn`/`fadd_rn`/`fsub_rn` (IGC contracts mul+add differently
//     in different kernels; see iq_kernels.cpp), so the exact and upstream layouts differ only by the cross-warp
//     grouping the header documents.
//   * the q8_1 quantizer is iq_kernels' `quantize_q8_1_rows`: the CUDA file carries a second copy of the same
//     llama.cpp quantize.cu kernel; contiguous columns quantize as one vector, as the CUDA comment notes.
//   * the CUDA "explicit non-null stream" rule is kept as a host check.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/iq_kernels.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "fp_exact.hpp"

#include <cuda_runtime.h>
#include <sycl/ext/intel/math.hpp>

#include <cstddef>
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <limits>
#include <map>
#include <mutex>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

namespace im = strata::kernels::fp_exact;
// every helper of the dot products inlined: one that is not leaves its local arrays (u, d8, sc, m) in private
// memory, and the multi-column kernels ran 27x slower than one column (measured on the A770, Q4_K at 4 columns)
#define STRATA_INLINE __attribute__((always_inline)) inline

constexpr int QK = 256;
constexpr int Q8K = 32;
constexpr int QI = 32;
constexpr int VDR = 2;
constexpr int WARPS = 4;
constexpr int WARP = 32;

inline float fm(float a, float b) { return im::fmul_rn(a, b); }
inline float fa(float a, float b) { return im::fadd_rn(a, b); }
inline float fs(float a, float b) { return im::fsub_rn(a, b); }

// fp16 as raw bits in the block structs (the layouts and offsets are the CUDA file's, asserted below)
struct h16 { uint16_t b; float f() const { return f32_from_f16(b); } };
struct alignas(4) h16x2 { h16 x, y; };

struct Q5KBlock { h16x2 dm; uint8_t scales[12]; uint8_t qh[32]; uint8_t qs[128]; };
struct alignas(4) Q81Block { h16x2 ds; int8_t qs[32]; };
struct Q20Block { h16 d; uint8_t qs[16]; };
struct Q3KBlock { uint8_t hmask[32]; uint8_t qs[64]; uint8_t scales[12]; h16 d; };
struct IQ4XSBlock { h16 d; uint16_t scales_h; uint8_t scales_l[4]; uint8_t qs[128]; };
struct Q4KBlock { h16x2 dm; uint8_t scales[12]; uint8_t qs[128]; };
struct Q6KBlock { uint8_t ql[128]; uint8_t qh[64]; int8_t scales[16]; h16 d; };
struct Q40Block { h16 d; uint8_t qs[16]; };
struct Q50Block { h16 d; uint8_t qh[4]; uint8_t qs[16]; };
struct Q80Block { h16 d; int8_t qs[32]; };
struct IQ4NLBlock { h16 d; uint8_t qs[16]; };
static_assert(sizeof(Q5KBlock) == 176 && alignof(Q5KBlock) == 4);
static_assert(sizeof(Q81Block) == 36 && alignof(Q81Block) == 4);
static_assert(sizeof(Q20Block) == 18 && alignof(Q20Block) == 2 && offsetof(Q20Block, qs) == 2);
static_assert(sizeof(Q3KBlock) == 110 && alignof(Q3KBlock) == 2 && offsetof(Q3KBlock, qs) == 32 &&
              offsetof(Q3KBlock, scales) == 96 && offsetof(Q3KBlock, d) == 108);
static_assert(sizeof(IQ4XSBlock) == 136 && alignof(IQ4XSBlock) == 2 && offsetof(IQ4XSBlock, scales_h) == 2 &&
              offsetof(IQ4XSBlock, scales_l) == 4 && offsetof(IQ4XSBlock, qs) == 8);
static_assert(offsetof(Q5KBlock, scales) == 4 && offsetof(Q5KBlock, qh) == 16 && offsetof(Q5KBlock, qs) == 48 &&
              offsetof(Q81Block, qs) == 4);
static_assert(sizeof(Q4KBlock) == 144 && alignof(Q4KBlock) == 4 && offsetof(Q4KBlock, scales) == 4 &&
              offsetof(Q4KBlock, qs) == 16);
static_assert(sizeof(Q6KBlock) == 210 && alignof(Q6KBlock) == 2 && offsetof(Q6KBlock, qh) == 128 &&
              offsetof(Q6KBlock, scales) == 192 && offsetof(Q6KBlock, d) == 208);
static_assert(sizeof(Q40Block) == 18 && alignof(Q40Block) == 2 && offsetof(Q40Block, qs) == 2);
static_assert(sizeof(Q50Block) == 22 && alignof(Q50Block) == 2 && offsetof(Q50Block, qh) == 2 &&
              offsetof(Q50Block, qs) == 6);
static_assert(sizeof(Q80Block) == 34 && alignof(Q80Block) == 2 && offsetof(Q80Block, qs) == 2);
static_assert(sizeof(IQ4NLBlock) == 18 && alignof(IQ4NLBlock) == 2 && offsetof(IQ4NLBlock, qs) == 2);

struct int2 { int x, y; };

STRATA_INLINE uint32_t byte_perm(uint32_t a, uint32_t b, uint32_t s) {   // CUDA's 3-bit selector
    const uint64_t bytes = ((uint64_t) b << 32) | a;
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) r |= (uint32_t) ((bytes >> (8 * ((s >> (4 * i)) & 7))) & 0xFF) << (8 * i);
    return r;
}
// per-byte signed saturating a - b
STRATA_INLINE int vsubss4(int a, int b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) {
        int d = (int) (int8_t) (a >> (8 * i)) - (int) (int8_t) (b >> (8 * i));
        d = d < -128 ? -128 : d > 127 ? 127 : d;
        r |= (uint32_t) (uint8_t) d << (8 * i);
    }
    return (int) r;
}
STRATA_INLINE int load_int_b2(const void* ptr, int i32) {
    const auto* x = static_cast<const uint16_t*>(ptr);
    int value = x[2 * i32] << 0;
    value |= x[2 * i32 + 1] << 16;
    return value;
}
STRATA_INLINE int q8_word(const Q81Block* b, int i) { return reinterpret_cast<const int*>(b->qs)[i]; }

constexpr int8_t kIq4nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10, 1, 13, 25, 38, 53, 69, 89, 113};
STRATA_INLINE int2 iq4_table_lookup(int q4) {
    uint32_t x = 0, y = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t b = ((uint32_t) q4 >> (8 * i)) & 0xFF;
        x |= (uint32_t) (uint8_t) kIq4nl[b & 0xF] << (8 * i);
        y |= (uint32_t) (uint8_t) kIq4nl[b >> 4] << (8 * i);
    }
    return {(int) x, (int) y};
}

// ---------------------------------------------------------------- the pinned *_impl expressions
STRATA_INLINE float q5_q8_dot_impl(const int* vl, const int* vh, const int* u, const uint8_t* sc, const uint8_t* m,
                            const h16x2& dm5, const float* d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0f0f0f0f;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0f0f0f0f;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = STRATA_DP4A(v0i, u[2 * i], STRATA_DP4A(v1i, u[2 * i + 1], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i], STRATA_DP4A(0x01010101, u[2 * i + 1], 0));
        sumf_d = fa(sumf_d, fm(d8[i], (float) (dot1 * sc[i])));
        sumf_m = fa(sumf_m, fm(d8[i], (float) (dot2 * m[i])));
    }
    return fs(fm(dm5.x.f(), sumf_d), fm(dm5.y.f(), sumf_m));
}
STRATA_INLINE float q4_q8_dot_impl(const int* v, const int* u, const uint8_t* sc, const uint8_t* m, const h16x2& dm4,
                            const float* d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0f0f0f0f;
        const int v1i = (v[1] >> (4 * i)) & 0x0f0f0f0f;
        const int dot1 = STRATA_DP4A(v1i, u[2 * i + 1], STRATA_DP4A(v0i, u[2 * i], 0));
        const int dot2 = STRATA_DP4A(0x01010101, u[2 * i + 1], STRATA_DP4A(0x01010101, u[2 * i], 0));
        sumf_d = fa(sumf_d, fm(d8[i], (float) (dot1 * sc[i])));
        sumf_m = fa(sumf_m, fm(d8[i], (float) (dot2 * m[i])));
    }
    return fs(fm(dm4.x.f(), sumf_d), fm(dm4.y.f(), sumf_m));
}
STRATA_INLINE float q3_q8_dot_impl(int vl, int vh, const int* u, const uint8_t* scales, int scale_offset, float d3,
                            const float* d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 4; ++i) {
        const int isc = scale_offset + 2 * i;
        const int isc_low = isc % 8;
        const int sc_shift_low = 4 * (isc / 8);
        const int sc_low = (scales[isc_low] >> sc_shift_low) & 0xf;
        const int isc_high = isc % 4;
        const int sc_shift_high = 2 * (isc / 4);
        const int sc_high = ((scales[8 + isc_high] >> sc_shift_high) & 3) << 4;
        const int sc = (sc_low | sc_high) - 32;
        const int vil = (vl >> (2 * i)) & 0x03030303;
        const int vih = ((vh >> i) << 2) & 0x04040404;
        const int vi = vsubss4(vil, vih);
        sumf = fa(sumf, fm(d8[i], (float) (STRATA_DP4A(vi, u[i], 0) * sc)));
    }
    return fm(d3, sumf);
}
STRATA_INLINE float q6_q8_dot_impl(int vl, int vh, const int* u, const int8_t* scales, float d, const float* d8) {
    float sumf = 0.0f;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int sc = scales[4 * i];
        const int vil = (vl >> (4 * i)) & 0x0f0f0f0f;
        const int vih = ((vh >> (4 * i)) << 4) & 0x30303030;
        const int vi = vsubss4(vil | vih, 0x20202020);
        sumf = fa(sumf, fm(d8[i], (float) (STRATA_DP4A(vi, u[i], 0) * sc)));
    }
    return fm(d, sumf);
}
STRATA_INLINE void k_scales(const uint8_t* scales8, int bq8_offset, uint16_t aux[2]) {
    const uint16_t* scales = reinterpret_cast<const uint16_t*>(scales8);
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = uint32_t(-int32_t(j >= 2));
    aux[0] = uint16_t(((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = uint16_t(((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}

// the four 32-element formats (the CUDA file's small_q8_dot overloads)
STRATA_INLINE float small_q8_dot(const Q40Block* w, const Q81Block* x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int v = load_int_b2(w->qs, iqs + i);
        const int vi0 = (v >> 0) & 0x0f0f0f0f;
        const int vi1 = (v >> 4) & 0x0f0f0f0f;
        sumi = STRATA_DP4A(vi0, q8_word(x, iqs + i), sumi);
        sumi = STRATA_DP4A(vi1, q8_word(x, iqs + i + 4), sumi);
    }
    return fm(w->d.f(), fs(fm((float) sumi, x->ds.x.f()), fm(4.0f, x->ds.y.f())));
}
STRATA_INLINE float small_q8_dot(const Q50Block* w, const Q81Block* x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int vl = load_int_b2(w->qs, iqs + i);
        const int vh = load_int_b2(w->qh, 0) >> (4 * (iqs + i));
        int vi0 = (vl >> 0) & 0x0f0f0f0f;
        vi0 |= (vh << 4) & 0x00000010;
        vi0 |= (vh << 11) & 0x00001000;
        vi0 |= (vh << 18) & 0x00100000;
        vi0 |= (vh << 25) & 0x10000000;
        sumi = STRATA_DP4A(vi0, q8_word(x, iqs + i), sumi);
        int vi1 = (vl >> 4) & 0x0f0f0f0f;
        vi1 |= (vh >> 12) & 0x00000010;
        vi1 |= (vh >> 5) & 0x00001000;
        vi1 |= (vh << 2) & 0x00100000;
        vi1 |= (vh << 9) & 0x10000000;
        sumi = STRATA_DP4A(vi1, q8_word(x, iqs + i + 4), sumi);
    }
    return fm(w->d.f(), fs(fm((float) sumi, x->ds.x.f()), fm(8.0f, x->ds.y.f())));
}
STRATA_INLINE float small_q8_dot(const Q80Block* w, const Q81Block* x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) sumi = STRATA_DP4A(load_int_b2(w->qs, iqs + i), q8_word(x, iqs + i), sumi);
    return fm(fm(w->d.f(), x->ds.x.f()), float(sumi));
}
STRATA_INLINE float small_q8_dot(const IQ4NLBlock* w, const Q81Block* x, int iqs) {
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < 2; ++i) {
        const int2 v = iq4_table_lookup(load_int_b2(w->qs, iqs + i));
        sumi = STRATA_DP4A(v.x, q8_word(x, iqs + i), sumi);
        sumi = STRATA_DP4A(v.y, q8_word(x, iqs + i + 4), sumi);
    }
    return fm(fm(w->d.f(), x->ds.x.f()), (float) sumi);
}

// ---------------------------------------------------------------- per-format iteration traits (load / apply)
struct Q5KTraits {
    using Block = Q5KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    STRATA_INLINE static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int vl[2], vh[2]; uint16_t aux[2]; h16x2 dm; int bq8_offset; };
    STRATA_INLINE static W load(const Block* bq5, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq5->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        const int* qh = reinterpret_cast<const int*>(bq5->qh + 4 * ((iqs / 2) % 4));
        r.vl[0] = ql[0];
        r.vl[1] = ql[4];
        r.vh[0] = qh[0] >> r.bq8_offset;
        r.vh[1] = qh[4] >> r.bq8_offset;
        k_scales(bq5->scales, r.bq8_offset, r.aux);
        r.dm = bq5->dm;
        return r;
    }
    STRATA_INLINE static float apply(const W& r, const Q81Block* bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = bq8i->ds.x.f();
            u[2 * i] = q8_word(bq8i, (iqs / 2) % 4);
            u[2 * i + 1] = q8_word(bq8i, (iqs / 2) % 4 + 4);
        }
        const uint8_t sc[2] = {(uint8_t) r.aux[0], (uint8_t) (r.aux[0] >> 8)};
        const uint8_t m[2] = {(uint8_t) r.aux[1], (uint8_t) (r.aux[1] >> 8)};
        return q5_q8_dot_impl(r.vl, r.vh, u, sc, m, r.dm, d8);
    }
};
struct Q4KTraits {
    using Block = Q4KBlock;
    static constexpr int DIV = QK, T = QI / VDR, KBY = QK / Q8K, BPI = VDR * WARPS * WARP / QI;
    STRATA_INLINE static int kqs(int tid) { return VDR * (tid % (QI / VDR)); }
    struct W { int v[2]; uint16_t aux[2]; h16x2 dm; int bq8_offset; };
    STRATA_INLINE static W load(const Block* bq4, int iqs) {
        W r;
        r.bq8_offset = 2 * ((iqs / 2) / 4);
        const int* ql = reinterpret_cast<const int*>(bq4->qs + 16 * r.bq8_offset + 4 * ((iqs / 2) % 4));
        r.v[0] = ql[0];
        r.v[1] = ql[4];
        k_scales(bq4->scales, r.bq8_offset, r.aux);
        r.dm = bq4->dm;
        return r;
    }
    STRATA_INLINE static float apply(const W& r, const Q81Block* bq8, int iqs) {
        int u[4];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            const Q81Block* bq8i = bq8 + r.bq8_offset + i;
            d8[i] = bq8i->ds.x.f();
            u[2 * i] = q8_word(bq8i, (iqs / 2) % 4);
            u[2 * i + 1] = q8_word(bq8i, (iqs / 2) % 4 + 4);
        }
        const uint8_t sc[2] = {(uint8_t) r.aux[0], (uint8_t) (r.aux[0] >> 8)};
        const uint8_t m[2] = {(uint8_t) r.aux[1], (uint8_t) (r.aux[1] >> 8)};
        return q4_q8_dot_impl(r.v, u, sc, m, r.dm, d8);
    }
};
struct Q20Traits {
    using Block = Q20Block;
    static constexpr int DIV = 64, T = 2, KBY = 2, BPI = WARPS * WARP / 2;
    STRATA_INLINE static int kqs(int tid) { return tid % 2; }
    struct W { int qx[4], qy[4]; float d2; };
    STRATA_INLINE static W load(const Block* w, int iqs) {
        W r;
        r.d2 = w->d.f();
        const int16_t* qs = reinterpret_cast<const int16_t*>(w->qs) + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = byte_perm(0x020100ff, 0x020100ff, q >> 0);
            const int qo = byte_perm(0x020100ff, 0x020100ff, q >> 2);
            r.qx[j] = byte_perm(qe, qo, 0x5140);
            r.qy[j] = byte_perm(qe, qo, 0x7362);
        }
        return r;
    }
    STRATA_INLINE static float apply(const W& r, const Q81Block* x, int iqs) {
        const Q81Block* chunk = x + iqs;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = STRATA_DP4A(q8_word(chunk, j * 2), r.qx[j], sumi);
            sumi = STRATA_DP4A(q8_word(chunk, j * 2 + 1), r.qy[j], sumi);
        }
        return fm(fm(r.d2, chunk->ds.x.f()), (float) sumi);
    }
};
struct Q3KTraits {
    using Block = Q3KBlock;
    static constexpr int DIV = 256, T = 16, KBY = 8, BPI = WARPS * WARP / 16;
    STRATA_INLINE static int kqs(int tid) { return tid % 16; }
    struct W { int vl, vh; float d; const uint8_t* scales; int scale_offset, bq8_offset; };
    STRATA_INLINE static W load(const Block* w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 8);
        r.scale_offset = iqs - iqs % 8 + (iqs % 8) / 4;
        r.d = w->d.f();
        r.vl = load_int_b2(w->qs, iqs);
        r.vh = ~load_int_b2(w->hmask, iqs % 8) >> r.bq8_offset;
        r.scales = w->scales;
        return r;
    }
    STRATA_INLINE static float apply(const W& r, const Q81Block* x, int iqs) {
        int u[4];
        float d8[4];
#pragma unroll
        for (int i = 0; i < 4; ++i) {
            u[i] = q8_word(x + r.bq8_offset + i, iqs % 8);
            d8[i] = x[r.bq8_offset + i].ds.x.f();
        }
        return q3_q8_dot_impl(r.vl, r.vh, u, r.scales, r.scale_offset, r.d, d8);
    }
};
struct Q6KTraits {
    using Block = Q6KBlock;
    static constexpr int DIV = 256, T = 32, KBY = 8, BPI = WARPS * WARP / 32;
    STRATA_INLINE static int kqs(int tid) { return tid % 32; }
    struct W { int vl, vh; float d; const int8_t* scales; int bq8_offset; };
    STRATA_INLINE static W load(const Block* w, int iqs) {
        W r;
        r.bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
        const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
        const int vh_shift = 2 * ((iqs % 16) / 8);
        r.vl = load_int_b2(w->ql, iqs);
        r.vh = load_int_b2(w->qh, 8 * (iqs / 16) + iqs % 8) >> vh_shift;
        r.scales = w->scales + scale_offset;
        r.d = w->d.f();
        return r;
    }
    STRATA_INLINE static float apply(const W& r, const Q81Block* x, int iqs) {
        int u[2];
        float d8[2];
#pragma unroll
        for (int i = 0; i < 2; ++i) {
            u[i] = q8_word(x + r.bq8_offset + 2 * i, iqs % 8);
            d8[i] = x[r.bq8_offset + 2 * i].ds.x.f();
        }
        return q6_q8_dot_impl(r.vl, r.vh, u, r.scales, r.d, d8);
    }
};
struct IQ4XSTraits {
    using Block = IQ4XSBlock;
    static constexpr int DIV = 256, T = 8, KBY = 8, BPI = 4 * WARPS * WARP / 32;
    STRATA_INLINE static int kqs(int tid) { return 4 * (tid % 8); }
    struct W { int2 v[4]; int ls; float dw; };
    STRATA_INLINE static W load(const Block* w, int iqs) {
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = iq4_table_lookup(reinterpret_cast<const int*>(w->qs)[iqs + j]);
        r.ls = ((w->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0f) | (((w->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = w->d.f();
        return r;
    }
    STRATA_INLINE static float apply(const W& r, const Q81Block* x, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            sumi = STRATA_DP4A(r.v[j].x, q8_word(x + iqs / 4, j), sumi);
            sumi = STRATA_DP4A(r.v[j].y, q8_word(x + iqs / 4, j + 4), sumi);
        }
        sumi *= r.ls - 32;
        return fm(fm(r.dw, x[iqs / 4].ds.x.f()), (float) sumi);
    }
};
template<typename Weight, int Qi>
struct SmallTraits {
    using Block = Weight;
    static constexpr int DIV = 32, T = Qi / 2, KBY = 1, BPI = 2 * WARPS * WARP / Qi;
    STRATA_INLINE static int kqs(int tid) { return 2 * (tid % (Qi / 2)); }
    struct W { const Weight* w; };
    STRATA_INLINE static W load(const Block* w, int) { return W{w}; }
    STRATA_INLINE static float apply(const W& r, const Q81Block* x, int k) { return small_q8_dot(r.w, x, k); }
};

bool g_multi_exact = true;   // as the CUDA file: the upstream layout is not the default until it is measured

// NW sub-groups per work-group, ROWS rows per work-group, NCOLS activation columns; the CUDA multi-column kernel.
template<typename F, int NCOLS, int NW, int ROWS>
void launch_kernel(sycl::queue& q, const typename F::Block* w, const Q81Block* x, float* y, int n_in, int n_out,
                   size_t groups) {
    constexpr int BPI = F::BPI * NW / WARPS;
    constexpr int NP = NW - 1 > 0 ? NW - 1 : 1;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partial(sycl::range<1>(NP * NCOLS * ROWS * WARP), h);
        h.parallel_for(sycl::nd_range<1>(groups * NW * WARP, NW * WARP),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int tid = (int) it.get_local_id(0);
            const int warp = tid / WARP, lane = tid % WARP;
            const int row0 = ROWS * (int) it.get_group(0);
            const int blocks_per_row = n_in / F::DIV;
            const int x_stride = n_in / Q8K;
            float tmp[NCOLS][ROWS] = {};
            for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += BPI) {
                const int kby = kbx * F::KBY;
                const int kqs = F::kqs(tid);
#pragma unroll
                for (int i = 0; i < ROWS; ++i) {
                    if (row0 + i < n_out) {
                        const std::size_t block = std::size_t(row0 + i) * blocks_per_row + kbx;
                        const typename F::W wv = F::load(w + block, kqs);
#pragma unroll
                        for (int j = 0; j < NCOLS; ++j)
                            tmp[j][i] = fa(tmp[j][i], F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs));
                    }
                }
            }
            auto at = [&](int l, int j, int i) -> float& { return partial[((l * NCOLS + j) * ROWS + i) * WARP + lane]; };
            if (warp > 0) {
#pragma unroll
                for (int j = 0; j < NCOLS; ++j)
#pragma unroll
                    for (int i = 0; i < ROWS; ++i) at(warp - 1, j, i) = tmp[j][i];
            }
            sycl::group_barrier(it.get_group());
            if (warp > 0) return;
            const sycl::sub_group sg = it.get_sub_group();
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
#pragma unroll
                for (int i = 0; i < ROWS; ++i) {
#pragma unroll
                    for (int l = 0; l < NW - 1; ++l) tmp[j][i] = fa(tmp[j][i], at(l, j, i));
                    float v = tmp[j][i];
#pragma unroll
                    for (int o = WARP / 2; o > 0; o >>= 1) v = fa(v, sycl::permute_group_by_xor(sg, v, o));
                    if (lane == i && row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = v;
                }
            }
        });
    });
}

// ---- the Intel layout (the exact path, ncols 1..8).  The CUDA layout above gives each work-item the register file of
// a CUDA thread; on the A770 a 32-wide sub-group leaves 32 registers per lane, and the multi-column kernels spilled
// 0.7-13 KB per lane (measured in IGC's assembly; Q4_K at 4 columns ran 27x slower than at 1).  Here a row group is 64
// work-items in 16-wide sub-groups (twice the registers per lane), each lane strides the row's (block, part) calls
// with the format's own `load` / `apply`, the 16 lanes are summed by an xor butterfly and the 4 sub-groups in order
// through local memory.  ncols 1 and ncols > 1 run this same kernel with the same per-column order, so every column of
// a multi-column call is bitwise its single-column call (the header's multi_exact contract) by construction.
constexpr int SG16 = 16, NSG = 4, NT16 = SG16 * NSG;

template<typename F, int NCOLS, int ROWS>
void launch_sg16(sycl::queue& q, const typename F::Block* w, const Q81Block* x, float* y, int n_in, int n_out,
                 std::size_t groups) {
    static_assert(NT16 % F::T == 0, "a format's calls per block must divide the row group");
    constexpr int BPI = NT16 / F::T;   // blocks per iteration of the row group
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(NSG * NCOLS * ROWS), h);
        h.parallel_for(sycl::nd_range<1>(groups * NT16, NT16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
            const int tid = (int) it.get_local_id(0);
            const int sgi = tid / SG16, lane = tid % SG16;
            const int row0 = ROWS * (int) it.get_group(0);
            const int blocks_per_row = n_in / F::DIV;
            const int x_stride = n_in / Q8K;
            const int kqs = F::kqs(tid);
            float tmp[NCOLS][ROWS] = {};
            for (int kbx = tid / F::T; kbx < blocks_per_row; kbx += BPI) {
                const int kby = kbx * F::KBY;
#pragma unroll
                for (int i = 0; i < ROWS; ++i) {
                    if (row0 + i < n_out) {
                        const typename F::W wv = F::load(w + std::size_t(row0 + i) * blocks_per_row + kbx, kqs);
#pragma unroll
                        for (int j = 0; j < NCOLS; ++j)
                            tmp[j][i] = fa(tmp[j][i], F::apply(wv, x + std::size_t(j) * x_stride + kby, kqs));
                    }
                }
            }
            const sycl::sub_group sg = it.get_sub_group();
#pragma unroll
            for (int j = 0; j < NCOLS; ++j)
#pragma unroll
                for (int i = 0; i < ROWS; ++i) {
                    float v = tmp[j][i];
#pragma unroll
                    for (int o = SG16 / 2; o > 0; o >>= 1) v = fa(v, sycl::permute_group_by_xor(sg, v, o));
                    if (lane == 0) part[(sgi * NCOLS + j) * ROWS + i] = v;
                }
            sycl::group_barrier(it.get_group());
            if (tid < NCOLS * ROWS) {
                const int j = tid / ROWS, i = tid % ROWS;
                float sum = part[j * ROWS + i];
#pragma unroll
                for (int l = 1; l < NSG; ++l) sum = fa(sum, part[(l * NCOLS + j) * ROWS + i]);
                if (row0 + i < n_out) y[std::size_t(j) * n_out + row0 + i] = sum;
            }
        });
    });
}

// One 16-wide sub-group per row, NSG rows per work-group: a lane walks the row's (block, part) calls k = lane,
// lane + 16, ... (kbx = k / T, part = kqs(k % T), the calls a CUDA thread tid makes for tid = k), and the row's sum is
// one xor butterfly - no local memory, no barrier, and 4x fewer work-groups than a row group per row.
template<typename F, int NCOLS>
void launch_row_sg(sycl::queue& q, const typename F::Block* w, const Q81Block* x, float* y, int n_in, int n_out) {
    const std::size_t groups = (std::size_t(n_out) + NSG - 1) / NSG;
    q.parallel_for(sycl::nd_range<1>(groups * NT16, NT16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int tid = (int) it.get_local_id(0);
        const int lane = tid % SG16;
        const int row = (int) it.get_group(0) * NSG + tid / SG16;
        if (row >= n_out) return;   // a whole sub-group
        const int blocks_per_row = n_in / F::DIV;
        const int x_stride = n_in / Q8K;
        const typename F::Block* wr = w + std::size_t(row) * blocks_per_row;
        float tmp[NCOLS] = {};
        for (int k = lane; k < blocks_per_row * F::T; k += SG16) {
            const int kbx = k / F::T, kqs = F::kqs(k % F::T);
            const typename F::W wv = F::load(wr + kbx, kqs);
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) tmp[j] = fa(tmp[j], F::apply(wv, x + std::size_t(j) * x_stride + kbx * F::KBY, kqs));
        }
        const sycl::sub_group sg = it.get_sub_group();
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float v = tmp[j];
#pragma unroll
            for (int o = SG16 / 2; o > 0; o >>= 1) v = fa(v, sycl::permute_group_by_xor(sg, v, o));
            if (lane == j % SG16) y[std::size_t(j) * n_out + row] = v;
        }
    });
}

// Q6_K in the reordered layout (native_mmvq.hpp): launch_row_sg's lanes, calls and order with Q6KTraits' load
// reading the four arrays - every word one aligned 32-bit load - and its apply written out per column.
struct Q6KR { const uint8_t* ql; const uint8_t* qh; const int8_t* sc; const uint16_t* d; };
template<int NCOLS>
void launch_q6r(sycl::queue& q, Q6KR w, const Q81Block* x, float* y, int n_in, int n_out) {
    const std::size_t groups = (std::size_t(n_out) + NSG - 1) / NSG;
    q.parallel_for(sycl::nd_range<1>(groups * NT16, NT16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const int tid = (int) it.get_local_id(0);
        const int lane = tid % SG16;
        const int row = (int) it.get_group(0) * NSG + tid / SG16;
        if (row >= n_out) return;   // a whole sub-group
        const int blocks_per_row = n_in / QK;
        const int x_stride = n_in / Q8K;
        float tmp[NCOLS] = {};
        for (int k = lane; k < blocks_per_row * 32; k += SG16) {
            const int kbx = k / 32, iqs = k % 32;
            const std::size_t b = std::size_t(row) * blocks_per_row + kbx;
            const int bq8_offset = 4 * (iqs / 16) + (iqs % 16) / 8;
            const int scale_offset = 8 * (iqs / 16) + (iqs % 16) / 4;
            const int vh_shift = 2 * ((iqs % 16) / 8);
            const int vl = reinterpret_cast<const int*>(w.ql + b * 128)[iqs];
            const int vh = reinterpret_cast<const int*>(w.qh + b * 64)[8 * (iqs / 16) + iqs % 8] >> vh_shift;
            const int8_t* scales = w.sc + b * 16 + scale_offset;
            const float d = f32_from_f16(w.d[b]);
            const Q81Block* xb = x + kbx * 8 + bq8_offset;
#pragma unroll
            for (int j = 0; j < NCOLS; ++j) {
                int u[2];
                float d8[2];
#pragma unroll
                for (int i = 0; i < 2; ++i) {
                    u[i] = q8_word(xb + std::size_t(j) * x_stride + 2 * i, iqs % 8);
                    d8[i] = xb[std::size_t(j) * x_stride + 2 * i].ds.x.f();
                }
                tmp[j] = fa(tmp[j], q6_q8_dot_impl(vl, vh, u, scales, d, d8));
            }
        }
        const sycl::sub_group sg = it.get_sub_group();
#pragma unroll
        for (int j = 0; j < NCOLS; ++j) {
            float v = tmp[j];
#pragma unroll
            for (int o = SG16 / 2; o > 0; o >>= 1) v = fa(v, sycl::permute_group_by_xor(sg, v, o));
            if (lane == j % SG16) y[std::size_t(j) * n_out + row] = v;
        }
    });
}

std::mutex g_q6r_mutex;
std::map<const void*, std::size_t> g_q6r;   // registered reordered matrices -> block count

template<typename F, int NCOLS>
void launch_n(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, sycl::queue& q) {
    const auto* w = static_cast<const typename F::Block*>(weights);
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    if (NCOLS > 1 && !g_multi_exact) {
        constexpr int NW = NCOLS <= 4 ? 4 : 2;
        launch_kernel<F, NCOLS, NW, 2>(q, w, x, y, n_in, n_out, (std::size_t(n_out) + 1) / 2);
        return;
    }
    // a row with fewer calls than the row group has work-items takes 4 rows per group (CUDA's small-K rule)
    static const int layout = [] { const char* v = std::getenv("STRATA_SYCL_MMVQ"); return v ? std::atoi(v) : 0; }();
    if (layout == 0) {
        launch_row_sg<F, NCOLS>(q, w, x, y, n_in, n_out);
    } else if ((n_in / F::DIV) * F::T < NT16)
        launch_sg16<F, NCOLS, 4>(q, w, x, y, n_in, n_out, (std::size_t(n_out) + 3) / 4);
    else
        launch_sg16<F, NCOLS, 1>(q, w, x, y, n_in, n_out, std::size_t(n_out));
}

void validate_shape(int n_in, int ncols, int block_elems = Q8K) {
    if (n_in <= 0 || n_in % block_elems != 0)
        throw std::invalid_argument("native MMVQ requires n_in > 0 and divisible by its block element count");
    if (ncols < 1 || ncols > 8) throw std::invalid_argument("native MMVQ requires 1 <= ncols <= 8");
}
void validate_pointer(const void* p) {
    if (!p || reinterpret_cast<std::uintptr_t>(p) % 4 != 0)
        throw std::invalid_argument("native MMVQ requires non-null 4-byte aligned device pointers");
}
void validate_stream(void* stream) {
    if (!stream) throw std::invalid_argument("native MMVQ requires an explicit non-null stream");
}
void launch_check() {
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(std::string("native MMVQ launch: ") + cudaGetErrorString(error));
}

template<typename F>
void mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    validate_shape(n_in, ncols, F::DIV);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    switch (ncols) {
        case 1: launch_n<F, 1>(weights, x_q8_1, y, n_in, n_out, q); break;
        case 2: launch_n<F, 2>(weights, x_q8_1, y, n_in, n_out, q); break;
        case 3: launch_n<F, 3>(weights, x_q8_1, y, n_in, n_out, q); break;
        case 4: launch_n<F, 4>(weights, x_q8_1, y, n_in, n_out, q); break;
        case 5: launch_n<F, 5>(weights, x_q8_1, y, n_in, n_out, q); break;
        case 6: launch_n<F, 6>(weights, x_q8_1, y, n_in, n_out, q); break;
        case 7: launch_n<F, 7>(weights, x_q8_1, y, n_in, n_out, q); break;
        case 8: launch_n<F, 8>(weights, x_q8_1, y, n_in, n_out, q); break;
    }
    launch_check();
}

template<typename F>
void mmvq_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out, int ncols,
              void* stream) {
    // validate everything before enqueueing the first operation
    validate_shape(n_in, ncols, F::DIV);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    validate_pointer(weights);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    mmvq<F>(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

using Q40Traits = SmallTraits<Q40Block, 4>;
using Q50Traits = SmallTraits<Q50Block, 4>;
using Q80Traits = SmallTraits<Q80Block, 8>;
using IQ4NLTraits = SmallTraits<IQ4NLBlock, 4>;

}  // namespace

void native_mmvq_set_multi_exact(bool exact) { g_multi_exact = exact; }
bool native_mmvq_multi_exact() { return g_multi_exact; }

std::size_t native_q8_1_bytes(int n_in, int ncols) {
    validate_shape(n_in, ncols);
    return std::size_t(ncols) * std::size_t(n_in / Q8K) * sizeof(Q81Block);
}

void native_quantize_q8_1(const float* x, void* x_q8_1, int n_in, int ncols, void* stream) {
    validate_shape(n_in, ncols);
    validate_pointer(x);
    validate_pointer(x_q8_1);
    validate_stream(stream);
    quantize_q8_1_rows(x, ncols, n_in, x_q8_1, stream);
    launch_check();
}

#define STRATA_NATIVE_ENTRY(name, F)                                                                            \
    void native_##name##_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,  \
                              void* stream) {                                                                   \
        mmvq<F>(weights, x_q8_1, y, n_in, n_out, ncols, stream);                                                \
    }                                                                                                           \
    void native_##name##_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in,         \
                             int n_out, int ncols, void* stream) {                                              \
        mmvq_f32<F>(weights, x, scratch_q8_1, y, n_in, n_out, ncols, stream);                                   \
    }
STRATA_NATIVE_ENTRY(q5_k, Q5KTraits)
STRATA_NATIVE_ENTRY(q2_0, Q20Traits)
STRATA_NATIVE_ENTRY(q3_k, Q3KTraits)
STRATA_NATIVE_ENTRY(iq4_xs, IQ4XSTraits)
STRATA_NATIVE_ENTRY(q4_k, Q4KTraits)
STRATA_NATIVE_ENTRY(q4_0, Q40Traits)
STRATA_NATIVE_ENTRY(q5_0, Q50Traits)
STRATA_NATIVE_ENTRY(q8_0, Q80Traits)
STRATA_NATIVE_ENTRY(iq4_nl, IQ4NLTraits)
#undef STRATA_NATIVE_ENTRY

void native_q6_k_reorder(const void* blocks, void* out, std::size_t n_blocks) {
    const auto* in = static_cast<const uint8_t*>(blocks);
    auto* o = static_cast<uint8_t*>(out);
    for (std::size_t i = 0; i < n_blocks; ++i) {
        const uint8_t* b = in + i * sizeof(Q6KBlock);
        std::memcpy(o + i * 128, b, 128);
        std::memcpy(o + n_blocks * 128 + i * 64, b + 128, 64);
        std::memcpy(o + n_blocks * 192 + i * 16, b + 192, 16);
        std::memcpy(o + n_blocks * 208 + i * 2, b + 208, 2);
    }
}
void native_q6_k_register_reordered(const void* weights, std::size_t n_blocks) {
    validate_pointer(weights);
    const std::lock_guard<std::mutex> lock(g_q6r_mutex);
    g_q6r[weights] = n_blocks;
}
void native_q6_k_unregister(const void* weights) {
    const std::lock_guard<std::mutex> lock(g_q6r_mutex);
    g_q6r.erase(weights);
}
std::size_t native_q6_k_reordered_blocks(const void* weights) {
    const std::lock_guard<std::mutex> lock(g_q6r_mutex);
    const auto found = g_q6r.find(weights);
    return found == g_q6r.end() ? 0 : found->second;
}

void native_q6_k_mmvq(const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                      void* stream) {
    const std::size_t nb = native_q6_k_reordered_blocks(weights);
    if (nb == 0) { mmvq<Q6KTraits>(weights, x_q8_1, y, n_in, n_out, ncols, stream); return; }
    validate_shape(n_in, ncols, QK);
    if (n_out <= 0 || std::size_t(n_in / QK) * std::size_t(n_out) != nb)
        throw std::invalid_argument("native MMVQ: a reordered Q6_K matrix of another shape");
    validate_pointer(x_q8_1);
    validate_pointer(y);
    validate_stream(stream);
    const auto* b = static_cast<const uint8_t*>(weights);
    const Q6KR w{b, b + nb * 128, reinterpret_cast<const int8_t*>(b + nb * 192),
                 reinterpret_cast<const uint16_t*>(b + nb * 208)};
    const auto* x = static_cast<const Q81Block*>(x_q8_1);
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    switch (ncols) {
        case 1: launch_q6r<1>(q, w, x, y, n_in, n_out); break;
        case 2: launch_q6r<2>(q, w, x, y, n_in, n_out); break;
        case 3: launch_q6r<3>(q, w, x, y, n_in, n_out); break;
        case 4: launch_q6r<4>(q, w, x, y, n_in, n_out); break;
        case 5: launch_q6r<5>(q, w, x, y, n_in, n_out); break;
        case 6: launch_q6r<6>(q, w, x, y, n_in, n_out); break;
        case 7: launch_q6r<7>(q, w, x, y, n_in, n_out); break;
        case 8: launch_q6r<8>(q, w, x, y, n_in, n_out); break;
    }
    launch_check();
}
void native_q6_k_f32(const void* weights, const float* x, void* scratch_q8_1, float* y, int n_in, int n_out,
                     int ncols, void* stream) {
    validate_shape(n_in, ncols, QK);
    validate_pointer(x);
    validate_pointer(scratch_q8_1);
    validate_stream(stream);
    native_quantize_q8_1(x, scratch_q8_1, n_in, ncols, stream);
    native_q6_k_mmvq(weights, scratch_q8_1, y, n_in, n_out, ncols, stream);
}

bool native_mmvq_supported(int ggml_type) noexcept {
    return ggml_type == 2 || ggml_type == 6 || ggml_type == 7 || ggml_type == 8 || ggml_type == 11 ||
           ggml_type == 12 || ggml_type == 13 || ggml_type == 14 || ggml_type == 20 ||
           ggml_type == 23 || ggml_type == 42 || ggml_type == 16 || ggml_type == 17 || ggml_type == 18 ||
           ggml_type == 21 || ggml_type == 22 || ggml_type == 29;
}

std::size_t native_mmvq_weight_bytes(int ggml_type, int n_in, int n_out) {
    int block_elems, block_bytes;
    switch (ggml_type) {
    case 2: block_elems = 32; block_bytes = 18; break;
    case 6: block_elems = 32; block_bytes = 22; break;
    case 7: block_elems = 32; block_bytes = 24; break;
    case 8: block_elems = 32; block_bytes = 34; break;
    case 20: block_elems = 32; block_bytes = 18; break;
    case 11: block_elems = 256; block_bytes = 110; break;
    case 12: block_elems = 256; block_bytes = 144; break;
    case 13: block_elems = 256; block_bytes = 176; break;
    case 14: block_elems = 256; block_bytes = 210; break;
    case 23: block_elems = 256; block_bytes = 136; break;
    case 42: block_elems = 64; block_bytes = 18; break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        block_elems = 256; block_bytes = (int) iq_row_bytes(ggml_type, 256); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
    validate_shape(n_in, 1, block_elems);
    if (n_out <= 0) throw std::invalid_argument("native MMVQ requires n_out > 0");
    const std::size_t row_bytes = std::size_t(n_in / block_elems) * block_bytes;
    if (row_bytes > std::numeric_limits<std::size_t>::max() / std::size_t(n_out))
        throw std::length_error("native MMVQ weight byte count overflows size_t");
    return row_bytes * std::size_t(n_out);
}

void native_mmvq(int ggml_type, const void* weights, const void* x_q8_1, float* y, int n_in, int n_out, int ncols,
                 void* stream) {
    switch (ggml_type) {
    case 2: native_q4_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 6: native_q5_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 7: iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 8: native_q8_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 20: native_iq4_nl_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 11: native_q3_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 12: native_q4_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 13: native_q5_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 14: native_q6_k_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 23: native_iq4_xs_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 42: native_q2_0_mmvq(weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    case 16: case 17: case 18: case 21: case 22: case 29:
        iq_mmvq(ggml_type, weights, x_q8_1, y, n_in, n_out, ncols, stream); break;
    default: throw std::invalid_argument("unsupported native MMVQ GGML type");
    }
}

}  // namespace strata::kernels
