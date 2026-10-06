// src/kernels/sycl/iq_kernels.cpp - SYCL port of src/kernels/cuda/iq_kernels.cu; see
// include/strata/kernels/iq_kernels.hpp.  The CUDA file's comments carry the contracts (the Q5_1 min term,
// the decode-once Split traits, the group stride, the fused SwiGLU + q8_1 pass) and are not repeated here.
//
// The dot products, dequantizers and q8_1 quantizer are llama.cpp's (ggml-cuda/vecdotq.cuh, dequantize.cuh,
// quantize.cu at the commit in third_party/ggml/VERSION.txt; MIT, third_party/ggml/LICENSE).  The block structs
// and codebook grids come from ggml-common.h, included unchanged through its SYCL declarations.
//
// What SYCL forces to change:
//   * `__byte_perm`, `__vcmpne4`, `__vsub4` and `__popc` are written out on 32-bit words (`byte_perm` uses
//     CUDA's 3-bit selector, which Q2_0 relies on).  The integer results are the CUDA instructions' bit for bit.
//   * the CUDA casts of codebook tables to wider words (`(const uint2*) iq2xxs_grid`, the 16-entry IQ4_NL table
//     read as four `uint32_t`) become shifts of the 64-bit entries and four byte lookups: SYCL gives the host
//     `static const` tables no alignment guarantee beyond their element type.
//   * every float product and sum of the dots, the lane reductions and the q8_1 block sum goes through
//     `fmul_rn`/`fadd_rn`/`fsub_rn`: IGC contracts mul+add differently in different kernels (measured on the
//     grouped 2-bit port), and the per-column/decode-once/grouped kernels are compared BITWISE by
//     iq_multi_parity and native_grouped_parity.  Variable-divisor divisions use `fdiv_rn` (IGC's default
//     float division is not correctly rounded).
//   * `__expf` in SwiGLU becomes `sycl::exp`; the plain and fused SwiGLU kernels share one helper, so the
//     v1 and fused passes stay bitwise equal on this backend.
//   * warps are 32-wide sub-groups of 1D work-groups; CUDA's 2D grids are flattened (x fastest).
//   * the row dots (MMVQ and the grouped experts) run on 16-wide sub-groups, W work-items per row (warp_sum_w):
//     each work-item carries 32 / W of the CUDA warp's lanes with their own accumulators and replays the CUDA
//     butterfly, so the sums are the 32-lane ones bit for bit.  At 32 lanes the A770 has 32 registers per lane and
//     the decode-once kernels spilled 2-5.5 KB per thread (IGC's dumps); grouped experts at the Coder's geometry went
//     from 644 to 399 us per layer call (IQ2_S/IQ4_NL).  W = 16 is the default (8 and 4 measured no faster).
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/dp4a.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "fp_exact.hpp"

#include <cuda_runtime.h>
#include <sycl/ext/intel/math.hpp>

#define GGML_COMMON_DECL_SYCL
#define GGML_COMMON_IMPL_SYCL
#include "ggml-common.h"

#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {

namespace im = strata::kernels::fp_exact;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

inline float fm(float a, float b) { return im::fmul_rn(a, b); }
inline float fa(float a, float b) { return im::fadd_rn(a, b); }
inline float fs(float a, float b) { return im::fsub_rn(a, b); }
inline float lo(const ggml_half2& h) { return (float) h[0]; }
inline float hi(const ggml_half2& h) { return (float) h[1]; }

struct int2 { int x, y; };
struct uint2 { uint32_t x, y; };

// ---------------------------------------------------------------- CUDA's byte instructions, written out
inline uint32_t byte_perm(uint32_t a, uint32_t b, uint32_t s) {
    const uint64_t bytes = ((uint64_t) b << 32) | a;
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) r |= (uint32_t) ((bytes >> (8 * ((s >> (4 * i)) & 7))) & 0xFF) << (8 * i);
    return r;
}
inline int vcmpne4(uint32_t a, uint32_t b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i)
        if (((a >> (8 * i)) & 0xFF) != ((b >> (8 * i)) & 0xFF)) r |= 0xFFu << (8 * i);
    return (int) r;
}
inline int vsub4(uint32_t a, uint32_t b) {
    uint32_t r = 0;
    for (int i = 0; i < 4; ++i) r |= (((a >> (8 * i)) - (b >> (8 * i))) & 0xFF) << (8 * i);
    return (int) r;
}
inline uint2 grid64(uint64_t g) { return {(uint32_t) g, (uint32_t) (g >> 32)}; }
inline uint8_t byte_of(uint64_t g, int j) { return (uint8_t) (g >> (8 * j)); }
inline uint8_t byte_of(uint32_t g, int j) { return (uint8_t) (g >> (8 * j)); }

// ---------------------------------------------------------------- llama.cpp helpers (vecdotq.cuh)
inline int get_int_b2(const void* x, int i32) {
    const uint16_t* x16 = (const uint16_t*) x;
    int x32 = x16[2 * i32 + 0] << 0;
    x32 |= x16[2 * i32 + 1] << 16;
    return x32;
}
inline int get_int_b4(const void* x, int i32) { return ((const int*) x)[i32]; }
inline uint32_t unpack_ksigns(const uint8_t v) {
    const uint32_t p = sycl::popcount((uint32_t) v) & 1;
    const uint32_t s = v ^ p << 7;
    return s * 0x01010101;
}
// the same words as the byte_perm form: x = table[low nibble of each byte], y = table[high nibble]
inline int2 get_int_from_table_16(const int& q4, const int8_t* table) {
    uint32_t x = 0, y = 0;
    for (int i = 0; i < 4; ++i) {
        const uint32_t b = ((uint32_t) q4 >> (8 * i)) & 0xFF;
        x |= (uint32_t) (uint8_t) table[b & 0xF] << (8 * i);
        y |= (uint32_t) (uint8_t) table[b >> 4] << (8 * i);
    }
    return {(int) x, (int) y};
}
#define ggml_cuda_dp4a(a, b, c) STRATA_DP4A((a), (b), (c))

// ---------------------------------------------------------------- the dot products (vecdotq.cuh)
inline float vec_dot_q2_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                               const int& iqs) {
    const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
    const float d2 = bq2_0->d;
    const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
    const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int q = qs[j];
        const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
        const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
        const int qe = byte_perm(0x020100FF, 0x020100FF, q >> 0);
        const int qo = byte_perm(0x020100FF, 0x020100FF, q >> 2);
        const int qx = byte_perm(qe, qo, 0x5140);
        const int qy = byte_perm(qe, qo, 0x7362);
        sumi = ggml_cuda_dp4a(u, qx, sumi);
        sumi = ggml_cuda_dp4a(v, qy, sumi);
    }
    const float d8 = lo(bq8_1_chunk->ds);
    return fm(fm(d2, d8), (float) sumi);
}

inline float vec_dot_iq2_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                                  const int& iqs) {
    const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
    const uint32_t q2 = get_int_b2(bq2->qs, iqs);
    const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
    int sumi = 0;
#pragma unroll
    for (int k0 = 0; k0 < 8; k0 += 2) {
        const uint2 grid_pos = grid64(iq2xxs_grid[byte_of(q2, k0 / 2)]);
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
        const int signs0 = vcmpne4(signs & 0x08040201, 0);
        const int grid0 = vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 0);
        sumi = ggml_cuda_dp4a(grid0, u0, sumi);
        const int signs1 = vcmpne4(signs & 0x80402010, 0);
        const int grid1 = vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, k0 + 1);
        sumi = ggml_cuda_dp4a(grid1, u1, sumi);
    }
    const int ls = aux32 >> 27 | 1;
    sumi = sumi * ls / 8;
    const float d = fm((float) bq2->d, lo(bq8_1[iqs / 2].ds));
    return fm(d, (float) sumi);
}

inline float vec_dot_iq2_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                                 const int& iqs) {
    const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
    const uint32_t w0 = get_int_b2(bq2->qs, iqs + 0), w1 = get_int_b2(bq2->qs, iqs + 1);
    const uint16_t q2[4] = {(uint16_t) w0, (uint16_t) (w0 >> 16), (uint16_t) w1, (uint16_t) (w1 >> 16)};
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = grid64(iq2xs_grid[q2[l0 / 2] & 0x1FF]);
        const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
        const int signs0 = vcmpne4(signs & 0x08040201, 0);
        const int grid_l = vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = vcmpne4(signs & 0x80402010, 0);
        const int grid_h = vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = fm((float) bq2->d, lo(bq8_1[iqs / 2].ds));
    return fm(d, (float) sumi);
}

inline void iq2_s_signs(uint8_t sp, int& signs0, int& signs1) {
    signs0 = vcmpne4(((sp & 0x03) << 7) | ((sp & 0x0C) << 21), 0x00000000);
    signs1 = vcmpne4(((sp & 0x30) << 3) | ((sp & 0xC0) << 17), 0x00000000);
}

inline float vec_dot_iq2_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                                const int& iqs) {
    const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
    const uint32_t qs_packed = get_int_b2(bq2->qs, iqs / 2);
    const int qh = bq2->qh[iqs / 2];
    const uint32_t signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
    const int ls0 = bq2->scales[iqs / 2] & 0x0F;
    const int ls1 = bq2->scales[iqs / 2] >> 4;
    int sumi0 = 0, sumi1 = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint2 grid_pos = grid64(iq2s_grid[byte_of(qs_packed, l0 / 2) | ((qh << (8 - l0)) & 0x300)]);
        int signs0, signs1;
        iq2_s_signs(byte_of(signs_packed_32, l0 / 2), signs0, signs1);
        const int grid_l = vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        if (l0 < 4) {
            sumi0 = ggml_cuda_dp4a(grid_l, u0, sumi0);
            sumi0 = ggml_cuda_dp4a(grid_h, u1, sumi0);
        } else {
            sumi1 = ggml_cuda_dp4a(grid_l, u0, sumi1);
            sumi1 = ggml_cuda_dp4a(grid_h, u1, sumi1);
        }
    }
    const int sumi = (sumi0 * ls0 + sumi1 * ls1 + (sumi0 + sumi1) / 2) / 4;
    const float d = fm((float) bq2->d, lo(bq8_1[iqs / 2].ds));
    return fm(d, (float) sumi);
}

inline float vec_dot_iq3_xxs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                                  const int& iqs) {
    const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
    const uint32_t w0 = get_int_b2(bq3->qs, iqs), w1 = get_int_b2(bq3->qs, iqs + 1);
    const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t w = l0 < 4 ? w0 : w1;
        const uint2 grid_pos = {iq3xxs_grid[byte_of(w, (l0 + 0) % 4)], iq3xxs_grid[byte_of(w, (l0 + 1) % 4)]};
        const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
        const int signs0 = vcmpne4(signs & 0x08040201, 0);
        const int grid_l = vsub4(grid_pos.x ^ signs0, signs0);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int signs1 = vcmpne4(signs & 0x80402010, 0);
        const int grid_h = vsub4(grid_pos.y ^ signs1, signs1);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    const int ls = aux32 >> 28;
    sumi = (ls * sumi + sumi / 2) / 2;
    const float d = fm((float) bq3->d, lo(bq8_1[iqs / 2].ds));
    return fm(d, (float) sumi);
}

inline float vec_dot_iq3_s_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                                const int& iqs) {
    const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
    const uint32_t w0 = get_int_b2(bq3->qs, iqs + 0), w1 = get_int_b2(bq3->qs, iqs + 1);
    const int qh = bq3->qh[iqs / 2];
    const uint32_t signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
    int sumi = 0;
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const uint32_t w = l0 < 4 ? w0 : w1;
        const uint2 grid_pos = {iq3s_grid[byte_of(w, (l0 + 0) % 4) | ((qh << (8 - l0)) & 0x100)],
                                iq3s_grid[byte_of(w, (l0 + 1) % 4) | ((qh << (7 - l0)) & 0x100)]};
        int signs0, signs1;
        iq2_s_signs(byte_of(signs_packed_32, l0 / 2), signs0, signs1);
        const int grid_l = vsub4(grid_pos.x ^ signs0, signs0);
        const int grid_h = vsub4(grid_pos.y ^ signs1, signs1);
        const int u0 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 2].qs, l0 + 1);
        sumi = ggml_cuda_dp4a(grid_l, u0, sumi);
        sumi = ggml_cuda_dp4a(grid_h, u1, sumi);
    }
    sumi *= 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
    const float d = fm((float) bq3->d, lo(bq8_1[iqs / 2].ds));
    return fm(d, (float) sumi);
}

// the IQ1_M fp16 super-block scale, assembled from the top nibbles of the four scale words
inline float iq1m_scale(const block_iq1_m* bq1) {
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    const uint16_t u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00F0) | ((sc[2] >> 4) & 0x0F00) | (sc[3] & 0xF000);
    return f32_from_f16(u16);
}
// exact in float: (qhl & 8) * (2 * IQ1M_DELTA / 8) is 0 or 0.25
inline float iq1m_delta(int qhl) { return -1.0f + IQ1M_DELTA - (qhl & 0x08) * (2.0f * IQ1M_DELTA / 0x08); }

inline float vec_dot_iq1_m_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                                const int& iqs) {
    const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
    const uint32_t qs_packed = get_int_b4(bq1->qs, iqs);
    int sumi[2] = {0, 0};
    float sumf[2] = {0.0f, 0.0f};
#pragma unroll
    for (int l0 = 0; l0 < 8; l0 += 2) {
        const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
        const int grid = iq1s_grid_gpu[byte_of(qs_packed, l0 / 2) | ((qhl & 0x07) << 8)];
        const int grid0 = (grid >> 0) & 0x0F0F0F0F;
        const int grid1 = (grid >> 4) & 0x0F0F0F0F;
        const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
        const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid0, u0, sumi[l0 / 4]);
        sumi[l0 / 4] = ggml_cuda_dp4a(grid1, u1, sumi[l0 / 4]);
        const float delta = iq1m_delta(qhl);
        int sumy = 0;
        sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
        sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
        sumf[l0 / 4] = fa(sumf[l0 / 4], fm(delta, (float) sumy));
    }
    const uint16_t* sc = (const uint16_t*) bq1->scales;
    const float d = fm(iq1m_scale(bq1), lo(bq8_1[iqs].ds));
    const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
    const int sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
    const int sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
    return fm(d, fa(fm(fa((float) sumi[0], sumf[0]), (float) sc0), fm(fa((float) sumi[1], sumf[1]), (float) sc1)));
}

inline float vec_dot_iq4_nl_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                                 const int& iqs) {
    const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
    const int* q8 = (const int*) bq8_1->qs + iqs;
    int sumi = 0;
#pragma unroll
    for (int l = 0; l < 2; ++l) {
        const int aux_q4 = get_int_b2(bq4->qs, iqs + l);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        sumi = ggml_cuda_dp4a(v.x, q8[l + 0], sumi);
        sumi = ggml_cuda_dp4a(v.y, q8[l + 4], sumi);
    }
    const float d = fm((float) bq4->d, lo(bq8_1->ds));
    return fm(d, (float) sumi);
}

inline float vec_dot_iq4_xs_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                                 const int& iqs) {
    const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        const int aux_q4 = get_int_b4(bq4->qs, iqs + j);
        const int2 v = get_int_from_table_16(aux_q4, kvalues_iq4nl);
        const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
        const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
        sumi = ggml_cuda_dp4a(v.x, u0, sumi);
        sumi = ggml_cuda_dp4a(v.y, u1, sumi);
    }
    const int ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
    sumi *= ls - 32;
    const float d = fm((float) bq4->d, lo(bq8_1[iqs / 4].ds));
    return fm(d, (float) sumi);
}

// ---------------------------------------------------------------- Unsloth's UD-Q4_K_XL experts
constexpr int VDR_Q4_K = 2, VDR_Q5_K = 2, VDR_Q5_1 = 2, VDR_Q5_0 = 2, VDR_Q8_0 = 2;

inline float vec_dot_q4_K_q8_1_impl_vmmq(const int* __restrict__ v, const int* __restrict__ u,
                                         const uint8_t* __restrict__ sc, const uint8_t* __restrict__ m,
                                         const ggml_half2& dm4, const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const int v0i = (v[0] >> (4 * i)) & 0x0F0F0F0F;
        const int v1i = (v[1] >> (4 * i)) & 0x0F0F0F0F;
        const int dot1 = ggml_cuda_dp4a(v1i, u[2 * i + 1], ggml_cuda_dp4a(v0i, u[2 * i + 0], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 1], ggml_cuda_dp4a(0x01010101, u[2 * i + 0], 0));
        sumf_d = fa(sumf_d, fm(d8[i], (float) (dot1 * sc[i])));
        sumf_m = fa(sumf_m, fm(d8[i], (float) (dot2 * m[i])));   // the min times the sum of the QUANTIZED activations
    }
    return fs(fm(lo(dm4), sumf_d), fm(hi(dm4), sumf_m));
}
inline float vec_dot_q5_K_q8_1_impl_vmmq(const int* __restrict__ vl, const int* __restrict__ vh,
                                         const int* __restrict__ u, const uint8_t* __restrict__ sc,
                                         const uint8_t* __restrict__ m, const ggml_half2& dm5,
                                         const float* __restrict__ d8) {
    float sumf_d = 0.0f;
    float sumf_m = 0.0f;
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const int vl0i = (vl[0] >> (4 * i)) & 0x0F0F0F0F;
        const int vl1i = (vl[1] >> (4 * i)) & 0x0F0F0F0F;
        const int vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        const int vh1i = ((vh[1] >> i) << 4) & 0x10101010;
        const int v0i = vl0i | vh0i;
        const int v1i = vl1i | vh1i;
        const int dot1 = ggml_cuda_dp4a(v0i, u[2 * i + 0], ggml_cuda_dp4a(v1i, u[2 * i + 1], 0));
        const int dot2 = ggml_cuda_dp4a(0x01010101, u[2 * i + 0], ggml_cuda_dp4a(0x01010101, u[2 * i + 1], 0));
        sumf_d = fa(sumf_d, fm(d8[i], (float) (dot1 * sc[i])));
        sumf_m = fa(sumf_m, fm(d8[i], (float) (dot2 * m[i])));
    }
    return fs(fm(lo(dm5), sumf_d), fm(hi(dm5), sumf_m));
}
inline void k_scale_min(const uint8_t* scales8, int bq8_offset, uint16_t aux[2]) {
    const uint16_t* scales = (const uint16_t*) scales8;
    const int j = bq8_offset / 2;
    const int jm = j & 1;
    const uint32_t s0 = scales[jm + 0];
    const uint32_t s2 = scales[jm + 2];
    const uint32_t s4 = scales[jm + 4];
    const uint32_t hi = (uint32_t) -(int32_t) (j >= 2);
    aux[0] = (uint16_t) (((s0 & 0x3f3f) & ~hi) | ((((s4 >> 0) & 0x0f0f) | ((s0 & 0xc0c0) >> 2)) & hi));
    aux[1] = (uint16_t) (((s2 & 0x3f3f) & ~hi) | ((((s4 >> 4) & 0x0f0f) | ((s2 & 0xc0c0) >> 2)) & hi));
}
inline float vec_dot_q4_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                               const int& iqs) {
    const block_q4_K* bq4_K = (const block_q4_K*) vbq + kbx;
    int v[2];
    int u[2 * QR4_K];
    float d8[QR4_K];
    const int bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    const int* q4 = (const int*) (bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    v[0] = q4[0];
    v[1] = q4[4];
    uint16_t aux[2];
    k_scale_min(bq4_K->scales, bq8_offset, aux);
    const uint8_t sc[2] = {(uint8_t) aux[0], (uint8_t) (aux[0] >> 8)};
    const uint8_t m[2] = {(uint8_t) aux[1], (uint8_t) (aux[1] >> 8)};
#pragma unroll
    for (int i = 0; i < QR4_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = lo(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, bq4_K->dm, d8);
}
inline float vec_dot_q5_K_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                               const int& iqs) {
    const block_q5_K* bq5_K = (const block_q5_K*) vbq + kbx;
    int vl[2];
    int vh[2];
    int u[2 * QR5_K];
    float d8[QR5_K];
    const int bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
    const int* ql = (const int*) (bq5_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    const int* qh = (const int*) (bq5_K->qh + 4 * ((iqs / 2) % 4));
    vl[0] = ql[0];
    vl[1] = ql[4];
    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;
    uint16_t aux[2];
    k_scale_min(bq5_K->scales, bq8_offset, aux);
    const uint8_t sc[2] = {(uint8_t) aux[0], (uint8_t) (aux[0] >> 8)};
    const uint8_t m[2] = {(uint8_t) aux[1], (uint8_t) (aux[1] >> 8)};
#pragma unroll
    for (int i = 0; i < QR5_K; ++i) {
        const block_q8_1* bq8i = bq8_1 + bq8_offset + i;
        d8[i] = lo(bq8i->ds);
        const int* q8 = (const int*) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0] = q8[0];
        u[2 * i + 1] = q8[4];
    }
    return vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);
}
inline int q5_lo_word(int vl, int vh) {
    int vi0 = (vl >> 0) & 0x0F0F0F0F;
    vi0 |= (vh << 4) & 0x00000010;
    vi0 |= (vh << 11) & 0x00001000;
    vi0 |= (vh << 18) & 0x00100000;
    vi0 |= (vh << 25) & 0x10000000;
    return vi0;
}
inline int q5_hi_word(int vl, int vh) {
    int vi1 = (vl >> 4) & 0x0F0F0F0F;
    vi1 |= (vh >> 12) & 0x00000010;
    vi1 |= (vh >> 5) & 0x00001000;
    vi1 |= (vh << 2) & 0x00100000;
    vi1 |= (vh << 9) & 0x10000000;
    return vi1;
}
inline float vec_dot_q5_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                               const int& iqs) {
    const block_q5_0* bq5_0 = (const block_q5_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_0; ++i) {
        const int vl = get_int_b2(bq5_0->qs, iqs + i);
        const int vh = get_int_b2(bq5_0->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_0);
        sumi = ggml_cuda_dp4a(q5_lo_word(vl, vh), u0, sumi);
        sumi = ggml_cuda_dp4a(q5_hi_word(vl, vh), u1, sumi);
    }
    const float d5 = (float) bq5_0->d;
    const float d8 = lo(bq8_1->ds);
    const float s8 = hi(bq8_1->ds);
    return fm(d5, fs(fm((float) sumi, d8), fm(16.0f * VDR_Q5_0 / QI5_0, s8)));
}
inline float vec_dot_q5_1_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                               const int& iqs) {
    const block_q5_1* bq5_1 = (const block_q5_1*) vbq + kbx;
    int sumi = 0, sumu = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q5_1; ++i) {
        const int vl = get_int_b4(bq5_1->qs, iqs + i);
        const int vh = get_int_b4(bq5_1->qh, 0) >> (4 * (iqs + i));
        const int u0 = get_int_b4(bq8_1->qs, iqs + i), u1 = get_int_b4(bq8_1->qs, iqs + i + QI5_1);
        sumi = ggml_cuda_dp4a(q5_lo_word(vl, vh), u0, sumi);
        sumi = ggml_cuda_dp4a(q5_hi_word(vl, vh), u1, sumi);
        sumu = ggml_cuda_dp4a(0x01010101, u1, ggml_cuda_dp4a(0x01010101, u0, sumu));
    }
    const float d8 = lo(bq8_1->ds);
    return fa(fm((float) sumi, fm(lo(bq5_1->dm), d8)), fm((float) sumu, fm(hi(bq5_1->dm), d8)));
}
inline float vec_dot_q8_0_q8_1(const void* __restrict__ vbq, const block_q8_1* __restrict__ bq8_1, const int& kbx,
                               const int& iqs) {
    const block_q8_0* bq8_0 = (const block_q8_0*) vbq + kbx;
    int sumi = 0;
#pragma unroll
    for (int i = 0; i < VDR_Q8_0; ++i)
        sumi = ggml_cuda_dp4a(get_int_b2(bq8_0->qs, iqs + i), get_int_b4(bq8_1->qs, iqs + i), sumi);
    return fm(fm((float) bq8_0->d, lo(bq8_1->ds)), (float) sumi);
}

// ---------------------------------------------------------------- the formats
// qk = values per block, ipb = dot calls per block (qi / vdr), step = the iqs stride between calls.
template<int TY> struct Fmt;
template<> struct Fmt<16> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<17> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<18> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_xxs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<20> { static constexpr int qk = 32, ipb = 2, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_nl_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<21> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq3_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<23> { static constexpr int qk = 256, ipb = 8, step = 4;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq4_xs_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<22> { static constexpr int qk = 256, ipb = 8, step = 2;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq2_s_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<29> { static constexpr int qk = 256, ipb = 8, step = 1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_iq1_m_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<42> { static constexpr int qk = 64, ipb = 2, step = 1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q2_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<12> { static constexpr int qk = 256, ipb = QI4_K / VDR_Q4_K, step = VDR_Q4_K;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q4_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<13> { static constexpr int qk = 256, ipb = QI5_K / VDR_Q5_K, step = VDR_Q5_K;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_K_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<7> { static constexpr int qk = 32, ipb = QI5_1 / VDR_Q5_1, step = VDR_Q5_1;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_1_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<6> { static constexpr int qk = 32, ipb = QI5_0 / VDR_Q5_0, step = VDR_Q5_0;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q5_0_q8_1(v, y, kbx, iqs); } };
template<> struct Fmt<8> { static constexpr int qk = 32, ipb = QI8_0 / VDR_Q8_0, step = VDR_Q8_0;
    static float dot(const void* v, const block_q8_1* y, int kbx, int iqs) { return vec_dot_q8_0_q8_1(v, y, kbx, iqs); } };

#define STRATA_GU_FMTS(X) X(16) X(17) X(18) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(6) X(8)
#define STRATA_D_FMTS(X) X(20) X(23) X(42) X(7) X(6) X(8)
#define STRATA_MMVQ_FMTS(X) X(16) X(17) X(18) X(20) X(21) X(22) X(23) X(29) X(42) X(12) X(13) X(7) X(6) X(8)

// The row dots run W work-items per row (W = 16, 8 or 4) on 16-wide sub-groups, 16 / W rows per sub-group.  Work-item
// l of a row is the CUDA warp's lanes l + j W (j < 32 / W), each with its own accumulator over its own k in the CUDA
// order (k = lane, lane + 32, ...).  The CUDA butterfly (offsets 16..1) is then replayed: its steps of offset >= W pair
// lanes that one work-item holds (t[j] with t[j + h], in the lane's operand order), the rest are permutes within the
// row's W work-items.  Every step adds what the 32-lane step added, so the sums are the CUDA kernel's bit for bit.
template<int W>
inline float warp_sum_w(const sycl::sub_group& sg, float (&t)[32 / W]) {
#pragma unroll
    for (int h = 16 / W; h >= 1; h >>= 1)
#pragma unroll
        for (int j = 0; j < h; ++j) t[j] = fa(t[j], t[j + h]);
    float v = t[0];
#pragma unroll
    for (int o = W / 2; o > 0; o >>= 1) v = fa(v, sycl::permute_group_by_xor(sg, v, o));
    return v;
}

template<int TY, int W>
inline float row_dot_w(const sycl::sub_group& sg, const uint8_t* row, const block_q8_1* x, int nb, int lr) {
    using F = Fmt<TY>;
    constexpr int V = 32 / W;
    float t[V];
#pragma unroll
    for (int j = 0; j < V; ++j) t[j] = 0.0f;
    const int nk = nb * F::ipb;
    for (int k0 = lr; k0 < nk; k0 += 32) {
#pragma unroll
        for (int j = 0; j < V; ++j) {
            const int k = k0 + j * W;
            if (k < nk) t[j] = fa(t[j], F::dot(row, x + (k / F::ipb) * (F::qk / 32), k / F::ipb, F::step * (k % F::ipb)));
        }
    }
    return warp_sum_w<W>(sg, t);
}

// ---------------------------------------------------------------- decode once, apply to every column
template<int TY> struct Split;
template<int TY> inline constexpr bool kSplit = false;

template<> inline constexpr bool kSplit<16> = true;
template<> struct Split<16> {   // IQ2_XXS
    struct W { int g[8]; int ls; float dw; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq2_xxs* bq2 = (const block_iq2_xxs*) vbq + kbx;
        const uint32_t q2 = get_int_b2(bq2->qs, iqs);
        const uint32_t aux32 = get_int_b2(bq2->qs, iqs + 1);
        W r;
#pragma unroll
        for (int k0 = 0; k0 < 8; k0 += 2) {
            const uint2 grid_pos = grid64(iq2xxs_grid[byte_of(q2, k0 / 2)]);
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * k0 / 2));
            const int signs0 = vcmpne4(signs & 0x08040201, 0);
            r.g[k0 + 0] = vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = vcmpne4(signs & 0x80402010, 0);
            r.g[k0 + 1] = vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = aux32 >> 27 | 1;
        r.dw = (float) bq2->d;
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = sumi * r.ls / 8;
        const float d = fm(r.dw, lo(bq8_1[iqs / 2].ds));
        return fm(d, (float) sumi);
    }
};
struct SplitLs2 {
    struct W { int g[8]; int ls0, ls1; float dw; };
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi0 = 0, sumi1 = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) sumi0 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi0);
#pragma unroll
        for (int j = 4; j < 8; ++j) sumi1 = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi1);
        const int sumi = (sumi0 * r.ls0 + sumi1 * r.ls1 + (sumi0 + sumi1) / 2) / 4;
        const float d = fm(r.dw, lo(bq8_1[iqs / 2].ds));
        return fm(d, (float) sumi);
    }
};
template<> inline constexpr bool kSplit<17> = true;
template<> struct Split<17> : SplitLs2 {   // IQ2_XS
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq2_xs* bq2 = (const block_iq2_xs*) vbq + kbx;
        const uint32_t w0 = get_int_b2(bq2->qs, iqs + 0), w1 = get_int_b2(bq2->qs, iqs + 1);
        const uint16_t q2[4] = {(uint16_t) w0, (uint16_t) (w0 >> 16), (uint16_t) w1, (uint16_t) (w1 >> 16)};
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 grid_pos = grid64(iq2xs_grid[q2[l0 / 2] & 0x1FF]);
            const uint32_t signs = unpack_ksigns(q2[l0 / 2] >> 9);
            const int signs0 = vcmpne4(signs & 0x08040201, 0);
            r.g[l0 + 0] = vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = vcmpne4(signs & 0x80402010, 0);
            r.g[l0 + 1] = vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.dw = (float) bq2->d;
        return r;
    }
};
template<> inline constexpr bool kSplit<22> = true;
template<> struct Split<22> : SplitLs2 {   // IQ2_S
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq2_s* bq2 = (const block_iq2_s*) vbq + kbx;
        const uint32_t qs_packed = get_int_b2(bq2->qs, iqs / 2);
        const int qh = bq2->qh[iqs / 2];
        const uint32_t signs_packed_32 = get_int_b2(bq2->qs, QK_K / 32 + iqs / 2);
        W r;
        r.ls0 = bq2->scales[iqs / 2] & 0x0F;
        r.ls1 = bq2->scales[iqs / 2] >> 4;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint2 grid_pos = grid64(iq2s_grid[byte_of(qs_packed, l0 / 2) | ((qh << (8 - l0)) & 0x300)]);
            int signs0, signs1;
            iq2_s_signs(byte_of(signs_packed_32, l0 / 2), signs0, signs1);
            r.g[l0 + 0] = vsub4(grid_pos.x ^ signs0, signs0);
            r.g[l0 + 1] = vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.dw = (float) bq2->d;
        return r;
    }
};
template<> inline constexpr bool kSplit<18> = true;
template<> struct Split<18> {   // IQ3_XXS
    struct W { int g[8]; int ls; float dw; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq3_xxs* bq3 = (const block_iq3_xxs*) vbq + kbx;
        const uint32_t w0 = get_int_b2(bq3->qs, iqs), w1 = get_int_b2(bq3->qs, iqs + 1);
        const uint32_t aux32 = get_int_b2(bq3->qs, QK_K / 16 + iqs / 2);
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t w = l0 < 4 ? w0 : w1;
            const uint2 grid_pos = {iq3xxs_grid[byte_of(w, (l0 + 0) % 4)], iq3xxs_grid[byte_of(w, (l0 + 1) % 4)]};
            const uint32_t signs = unpack_ksigns(aux32 >> (7 * l0 / 2));
            const int signs0 = vcmpne4(signs & 0x08040201, 0);
            r.g[l0 + 0] = vsub4(grid_pos.x ^ signs0, signs0);
            const int signs1 = vcmpne4(signs & 0x80402010, 0);
            r.g[l0 + 1] = vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = aux32 >> 28;
        r.dw = (float) bq3->d;
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi = (r.ls * sumi + sumi / 2) / 2;
        const float d = fm(r.dw, lo(bq8_1[iqs / 2].ds));
        return fm(d, (float) sumi);
    }
};
template<> inline constexpr bool kSplit<21> = true;
template<> struct Split<21> {   // IQ3_S
    struct W { int g[8]; int ls; float dw; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq3_s* bq3 = (const block_iq3_s*) vbq + kbx;
        const uint32_t w0 = get_int_b2(bq3->qs, iqs + 0), w1 = get_int_b2(bq3->qs, iqs + 1);
        const int qh = bq3->qh[iqs / 2];
        const uint32_t signs_packed_32 = get_int_b2(bq3->signs, iqs / 2);
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const uint32_t w = l0 < 4 ? w0 : w1;
            const uint2 grid_pos = {iq3s_grid[byte_of(w, (l0 + 0) % 4) | ((qh << (8 - l0)) & 0x100)],
                                    iq3s_grid[byte_of(w, (l0 + 1) % 4) | ((qh << (7 - l0)) & 0x100)]};
            int signs0, signs1;
            iq2_s_signs(byte_of(signs_packed_32, l0 / 2), signs0, signs1);
            r.g[l0 + 0] = vsub4(grid_pos.x ^ signs0, signs0);
            r.g[l0 + 1] = vsub4(grid_pos.y ^ signs1, signs1);
        }
        r.ls = 1 + 2 * ((bq3->scales[iqs / 4] >> ((iqs << 1) & 0x04)) & 0x0F);
        r.dw = (float) bq3->d;
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 8; ++j) sumi = ggml_cuda_dp4a(r.g[j], get_int_b4(bq8_1[iqs / 2].qs, j), sumi);
        sumi *= r.ls;
        const float d = fm(r.dw, lo(bq8_1[iqs / 2].ds));
        return fm(d, (float) sumi);
    }
};
template<> inline constexpr bool kSplit<29> = true;
template<> struct Split<29> {   // IQ1_M
    struct W { int g[8]; float delta[4]; int sc0, sc1; float dw; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq1_m* bq1 = (const block_iq1_m*) vbq + kbx;
        const uint32_t qs_packed = get_int_b4(bq1->qs, iqs);
        W r;
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int qhl = bq1->qh[2 * iqs + l0 / 4] >> (4 * ((l0 / 2) % 2));
            const int grid = iq1s_grid_gpu[byte_of(qs_packed, l0 / 2) | ((qhl & 0x07) << 8)];
            r.g[l0 + 0] = (grid >> 0) & 0x0F0F0F0F;
            r.g[l0 + 1] = (grid >> 4) & 0x0F0F0F0F;
            r.delta[l0 / 2] = iq1m_delta(qhl);
        }
        const uint16_t* sc = (const uint16_t*) bq1->scales;
        r.dw = iq1m_scale(bq1);
        const int tmp = sc[iqs / 2] >> (6 * (iqs % 2));
        r.sc0 = 2 * ((tmp >> 0) & 0x07) + 1;
        r.sc1 = 2 * ((tmp >> 3) & 0x07) + 1;
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi[2] = {0, 0};
        float sumf[2] = {0.0f, 0.0f};
#pragma unroll
        for (int l0 = 0; l0 < 8; l0 += 2) {
            const int u0 = get_int_b4(bq8_1[iqs].qs, l0 + 0);
            const int u1 = get_int_b4(bq8_1[iqs].qs, l0 + 1);
            sumi[l0 / 4] = ggml_cuda_dp4a(r.g[l0 + 0], u0, sumi[l0 / 4]);
            sumi[l0 / 4] = ggml_cuda_dp4a(r.g[l0 + 1], u1, sumi[l0 / 4]);
            int sumy = 0;
            sumy = ggml_cuda_dp4a(u0, 0x01010101, sumy);
            sumy = ggml_cuda_dp4a(u1, 0x01010101, sumy);
            sumf[l0 / 4] = fa(sumf[l0 / 4], fm(r.delta[l0 / 2], (float) sumy));
        }
        const float d = fm(r.dw, lo(bq8_1[iqs].ds));
        return fm(d, fa(fm(fa((float) sumi[0], sumf[0]), (float) r.sc0), fm(fa((float) sumi[1], sumf[1]), (float) r.sc1)));
    }
};
template<> inline constexpr bool kSplit<20> = true;
template<> struct Split<20> {   // IQ4_NL
    struct W { int2 v[2]; float dw; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq4_nl* bq4 = (const block_iq4_nl*) vbq + kbx;
        W r;
#pragma unroll
        for (int l = 0; l < 2; ++l) r.v[l] = get_int_from_table_16(get_int_b2(bq4->qs, iqs + l), kvalues_iq4nl);
        r.dw = (float) bq4->d;
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const int* q8 = (const int*) bq8_1->qs + iqs;
        int sumi = 0;
#pragma unroll
        for (int l = 0; l < 2; ++l) {
            sumi = ggml_cuda_dp4a(r.v[l].x, q8[l + 0], sumi);
            sumi = ggml_cuda_dp4a(r.v[l].y, q8[l + 4], sumi);
        }
        const float d = fm(r.dw, lo(bq8_1->ds));
        return fm(d, (float) sumi);
    }
};
template<> inline constexpr bool kSplit<23> = true;
template<> struct Split<23> {   // IQ4_XS
    struct W { int2 v[4]; int ls; float dw; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_iq4_xs* bq4 = (const block_iq4_xs*) vbq + kbx;
        W r;
#pragma unroll
        for (int j = 0; j < 4; ++j) r.v[j] = get_int_from_table_16(get_int_b4(bq4->qs, iqs + j), kvalues_iq4nl);
        r.ls = ((bq4->scales_l[iqs / 8] >> (iqs & 0x04)) & 0x0F) | (((bq4->scales_h >> (iqs / 2)) & 0x03) << 4);
        r.dw = (float) bq4->d;
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u0 = get_int_b4(bq8_1[iqs / 4].qs, j + 0);
            const int u1 = get_int_b4(bq8_1[iqs / 4].qs, j + 4);
            sumi = ggml_cuda_dp4a(r.v[j].x, u0, sumi);
            sumi = ggml_cuda_dp4a(r.v[j].y, u1, sumi);
        }
        sumi *= r.ls - 32;
        const float d = fm(r.dw, lo(bq8_1[iqs / 4].ds));
        return fm(d, (float) sumi);
    }
};
template<> inline constexpr bool kSplit<42> = true;
template<> struct Split<42> {   // Q2_0
    struct W { int qx[4], qy[4]; float d2; };
    static W load(const void* __restrict__ vbq, int kbx, int iqs) {
        const block_q2_0* bq2_0 = (const block_q2_0*) vbq + kbx;
        W r;
        r.d2 = bq2_0->d;
        const int16_t* qs = (const int16_t*) bq2_0->qs + iqs * 4;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int q = qs[j];
            const int qe = byte_perm(0x020100FF, 0x020100FF, q >> 0);
            const int qo = byte_perm(0x020100FF, 0x020100FF, q >> 2);
            r.qx[j] = byte_perm(qe, qo, 0x5140);
            r.qy[j] = byte_perm(qe, qo, 0x7362);
        }
        return r;
    }
    static float apply(const W& r, const block_q8_1* __restrict__ bq8_1, int iqs) {
        const block_q8_1* bq8_1_chunk = bq8_1 + iqs;
        int sumi = 0;
#pragma unroll
        for (int j = 0; j < 4; ++j) {
            const int u = get_int_b4(bq8_1_chunk->qs, j * 2 + 0);
            const int v = get_int_b4(bq8_1_chunk->qs, j * 2 + 1);
            sumi = ggml_cuda_dp4a(u, r.qx[j], sumi);
            sumi = ggml_cuda_dp4a(v, r.qy[j], sumi);
        }
        const float d8 = lo(bq8_1_chunk->ds);
        return fm(fm(r.d2, d8), (float) sumi);
    }
};

// row_dot_w for up to NC columns, each weight part decoded once
template<int TY, int NC, int W>
inline void row_dot_multi_w(const sycl::sub_group& sg, const uint8_t* row, const block_q8_1* x, const int (&off)[NC],
                            int n, int nb, int lr, float (&s)[NC]) {
    using F = Fmt<TY>;
    using S = Split<TY>;
    constexpr int V = 32 / W;
    float t[NC][V];
#pragma unroll
    for (int c = 0; c < NC; ++c)
#pragma unroll
        for (int j = 0; j < V; ++j) t[c][j] = 0.0f;
    const int nk = nb * F::ipb;
    for (int k0 = lr; k0 < nk; k0 += 32) {
#pragma unroll
        for (int j = 0; j < V; ++j) {
            const int k = k0 + j * W;
            if (k >= nk) continue;
            const int kbx = k / F::ipb, iqs = F::step * (k % F::ipb);
            const typename S::W w = S::load(row, kbx, iqs);
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < n) t[c][j] = fa(t[c][j], S::apply(w, x + off[c] + kbx * (F::qk / 32), iqs));
        }
    }
#pragma unroll
    for (int c = 0; c < NC; ++c) s[c] = c < n ? warp_sum_w<W>(sg, t[c]) : 0.0f;
}

// ---------------------------------------------------------------- launches
// `groups` work-groups of `wg` work-items (a multiple of 32), each 32 a sub-group; body(it) as a CUDA block.
template<class F>
void launch(sycl::queue& q, size_t groups, size_t wg, F body) {
    if (groups == 0) return;
    q.parallel_for(sycl::nd_range<1>(groups * wg, wg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        body(it);
    });
}

// the row-dot kernels: 16-wide sub-groups, W work-items per row (row_dot_w), 128-work-item groups of 128 / W rows
template<class F>
void launch16(sycl::queue& q, size_t groups, size_t wg, F body) {
    if (groups == 0) return;
    q.parallel_for(sycl::nd_range<1>(groups * wg, wg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        body(it);
    });
}
#ifndef STRATA_MMVQ_W
#define STRATA_MMVQ_W 16
#endif
constexpr int MMVQ_W = STRATA_MMVQ_W;

// A row past the end still runs (on the last row, its result dropped): the sub-group's permutes need every work-item.
template<int TY>
void mmvq_kernel(sycl::nd_item<1> it, const uint8_t* __restrict__ w, size_t row_bytes,
                 const block_q8_1* __restrict__ x, float* __restrict__ y, int n_in, int n_out, int ncols) {
    constexpr int W = MMVQ_W;
    const int tid = (int) it.get_local_id(0);
    const int row = (int) it.get_group(0) * (128 / W) + tid / W, lr = tid % W;
    const bool live = row < n_out;
    const int nb = n_in / Fmt<TY>::qk;
    const uint8_t* wr = w + (size_t) (live ? row : n_out - 1) * row_bytes;
    for (int c = 0; c < ncols; ++c) {
        const float s = row_dot_w<TY, W>(it.get_sub_group(), wr, x + (size_t) c * (n_in / 32), nb, lr);
        if (live && lr == 0) y[(size_t) c * n_out + row] = s;
    }
}

template<int TY, int NC>
void mmvq_multi_kernel(sycl::nd_item<1> it, const uint8_t* __restrict__ w, size_t row_bytes,
                       const block_q8_1* __restrict__ x, float* __restrict__ y, int n_in, int n_out, int ncols) {
    constexpr int W = MMVQ_W;
    const int tid = (int) it.get_local_id(0);
    const int row = (int) it.get_group(0) * (128 / W) + tid / W, lr = tid % W;
    const bool live = row < n_out;
    const int nb = n_in / Fmt<TY>::qk, xb = n_in / 32;
    const uint8_t* wr = w + (size_t) (live ? row : n_out - 1) * row_bytes;
    for (int c0 = 0; c0 < ncols; c0 += NC) {
        const int n = sycl::min(NC, ncols - c0);
        int off[NC];
#pragma unroll
        for (int c = 0; c < NC; ++c) off[c] = (c0 + sycl::min(c, n - 1)) * xb;
        float s[NC];
        row_dot_multi_w<TY, NC, W>(it.get_sub_group(), wr, x, off, n, nb, lr, s);
        if (live && lr == 0)
#pragma unroll
            for (int c = 0; c < NC; ++c)
                if (c < n) y[(size_t) (c0 + c) * n_out + row] = s[c];
    }
}

// ---------------------------------------------------------------- grouped native experts
#ifndef STRATA_GU_W
#define STRATA_GU_W 16
#endif
#ifndef STRATA_D_W
#define STRATA_D_W 16
#endif
constexpr int GU_W = STRATA_GU_W, D_W = STRATA_D_W;    // work-items per row (row_dot_w)
constexpr int GU_ROWS = 128 / GU_W, D_ROWS = 128 / D_W;  // rows per 128-work-item group
constexpr int GRP_NC = 4;

// The CUDA (gx, gy) grid flattened: work-group id = by * gx + bx.
struct Grid2 { size_t gx, gy; };

template<int TG, bool MULTI>
void native_gu_kernel(sycl::nd_item<1> it, Grid2 G, const unsigned long long* __restrict__ grp_ptr,
                      const int32_t* __restrict__ grp_start, const int32_t* __restrict__ n_groups,
                      const int32_t* __restrict__ ent_tok, const block_q8_1* __restrict__ xq, NativeExpertLayout L,
                      float* __restrict__ gate, float* __restrict__ up) {
    const int tid = (int) it.get_local_id(0);
    const int lane = tid % GU_W;
    const size_t bid = it.get_group(0);
    const int bx = (int) (bid % G.gx), by = (int) (bid / G.gx);
    const int row_ = bx * GU_ROWS + tid / GU_W;      // 0 .. 2*n_ff
    const bool live = row_ < 2 * L.n_ff;             // a row past the end runs the last row and drops it
    const int row = live ? row_ : (int) (2 * L.n_ff - 1);
    const bool is_up = row >= L.n_ff;
    const int r = is_up ? row - (int) L.n_ff : row;
    const size_t w_off = (is_up ? L.up_off : 0) + (size_t) r * L.gu_row;
    const int nb = (int) (L.n_embd / Fmt<TG>::qk), xb = (int) (L.n_embd / 32);
    float* dst = is_up ? up : gate;
    const sycl::sub_group sg = it.get_sub_group();
    const int ng = *n_groups;
    for (int g = by; g < ng; g += (int) G.gy) {
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        if constexpr (!MULTI) {
            for (int e = e0; e < e1; ++e) {
                const float s = row_dot_w<TG, GU_W>(sg, wr, xq + (size_t) ent_tok[e] * xb, nb, lane);
                if (live && lane == 0) dst[(size_t) e * L.n_ff + r] = s;
            }
        } else {
            for (int e = e0; e < e1; e += GRP_NC) {
                const int n = sycl::min(GRP_NC, e1 - e);
                int off[GRP_NC];
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c) off[c] = ent_tok[e + sycl::min(c, n - 1)] * xb;
                float s[GRP_NC];
                row_dot_multi_w<TG, GRP_NC, GU_W>(sg, wr, xq, off, n, nb, lane, s);
                if (live && lane == 0)
#pragma unroll
                    for (int c = 0; c < GRP_NC; ++c)
                        if (c < n) dst[(size_t) (e + c) * L.n_ff + r] = s[c];
            }
        }
    }
}

template<int TD, bool MULTI>
void native_down_kernel(sycl::nd_item<1> it, Grid2 G, const unsigned long long* __restrict__ grp_ptr,
                        const int32_t* __restrict__ grp_start, const int32_t* __restrict__ n_groups,
                        const int32_t* __restrict__ ent_dst, const block_q8_1* __restrict__ hq, NativeExpertLayout L,
                        float* __restrict__ out) {
    const int tid = (int) it.get_local_id(0);
    const int lane = tid % D_W;
    const size_t bid = it.get_group(0);
    const int bx = (int) (bid % G.gx), by = (int) (bid / G.gx);
    const int r_ = bx * D_ROWS + tid / D_W;
    const bool live = r_ < L.n_embd;                 // a row past the end runs the last row and drops it
    const int r = live ? r_ : (int) (L.n_embd - 1);
    const size_t w_off = L.down_off + (size_t) r * L.d_row;
    const int nb = (int) (L.n_ff / Fmt<TD>::qk), hb = (int) (L.n_ff / 32);
    const sycl::sub_group sg = it.get_sub_group();
    const int ng = *n_groups;
    for (int g = by; g < ng; g += (int) G.gy) {
        const uint8_t* wr = (const uint8_t*) grp_ptr[g] + w_off;
        const int e0 = grp_start[g], e1 = grp_start[g + 1];
        if constexpr (!MULTI) {
            for (int e = e0; e < e1; ++e) {
                const float s = row_dot_w<TD, D_W>(sg, wr, hq + (size_t) e * hb, nb, lane);
                if (live && lane == 0) out[(size_t) ent_dst[e] * L.n_embd + r] = s;
            }
        } else {
            for (int e = e0; e < e1; e += GRP_NC) {
                const int n = sycl::min(GRP_NC, e1 - e);
                int off[GRP_NC];
#pragma unroll
                for (int c = 0; c < GRP_NC; ++c) off[c] = (e + sycl::min(c, n - 1)) * hb;
                float s[GRP_NC];
                row_dot_multi_w<TD, GRP_NC, D_W>(sg, wr, hq, off, n, nb, lane, s);
                if (live && lane == 0)
#pragma unroll
                    for (int c = 0; c < GRP_NC; ++c)
                        if (c < n) out[(size_t) ent_dst[e + c] * L.n_embd + r] = s[c];
            }
        }
    }
}

// SwiGLU, the product rounded on its own (CUDA: __fmul_rn), shared by the v1 and the fused pass
inline float swiglu(float g, float u) { return fm(im::fdiv_rn(g, fa(1.0f, sycl::exp(-g))), u); }

// ---------------------------------------------------------------- q8_1 (quantize.cu)
// value i of a q8_1 row set (the sub-group holds block i / 32, lane = i % 32)
inline void q8_1_store(const sycl::sub_group& sg, const float xi, block_q8_1* __restrict__ y, const long long i) {
    float amax = sycl::fabs(xi), sum = xi;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        amax = sycl::fmax(amax, sycl::permute_group_by_xor(sg, amax, o));
        sum = fa(sum, sycl::permute_group_by_xor(sg, sum, o));
    }
    const float d = im::fdiv_rn(amax, 127.0f);
    const int8_t q = amax == 0.0f ? 0 : (int8_t) sycl::round(im::fdiv_rn(xi, d));
    const long long ib = i / 32, iqs = i % 32;
    y[ib].qs[iqs] = q;
    if (iqs == 0) y[ib].ds = ggml_half2(d, sum);
}

// ---------------------------------------------------------------- dequant (dequantize.cuh)
// fp16 results are raw bits (f16_from_f32: round to nearest even, as __float2half)
template<typename dst_t> inline dst_t cvt(float v);
template<> inline float cvt<float>(float v) { return v; }
template<> inline uint16_t cvt<uint16_t>(float v) { return f16_from_f32(v); }

template<typename dst_t>
void dq_iq2_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xxs* x = (const block_iq2_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint8_t aux8 = il % 2 ? (uint8_t) (q2[il / 2] >> 8) : (uint8_t) q2[il / 2];
    const uint64_t grid = iq2xxs_grid[aux8];
    const uint32_t aux32 = q2[2] | (q2[3] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * byte_of(grid, j) * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
void dq_iq2_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_xs* x = (const block_iq2_xs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* q2 = x[ibs].qs + 4 * ib;
    const uint64_t grid = iq2xs_grid[q2[il] & 511];
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * byte_of(grid, j) * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
void dq_iq2_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq2_s* x = (const block_iq2_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint64_t grid = iq2s_grid[x[ibs].qs[4 * ib + il] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 0x300)];
    const float d = (float) x[ibs].d * (0.5f + ((x[ibs].scales[ib] >> 4 * (il / 2)) & 0xf)) * 0.25f;
    const uint8_t signs = x[ibs].qs[QK_K / 8 + 4 * ib + il];
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * byte_of(grid, j) * (signs & kmask_iq2xs[j] ? -1.f : 1.f));
}
template<typename dst_t>
void dq_iq3_xxs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_xxs* x = (const block_iq3_xxs*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* q3 = x[ibs].qs + 8 * ib;
    const uint16_t* gas = (const uint16_t*) (x[ibs].qs + QK_K / 4) + 2 * ib;
    const uint32_t grid1 = iq3xxs_grid[q3[2 * il + 0]];
    const uint32_t grid2 = iq3xxs_grid[q3[2 * il + 1]];
    const uint32_t aux32 = gas[0] | (gas[1] << 16);
    const float d = (float) x[ibs].d * (0.5f + (aux32 >> 28)) * 0.5f;
    const uint8_t signs = ksigns_iq2xs[(aux32 >> 7 * il) & 127];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * byte_of(grid1, j) * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * byte_of(grid2, j) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
void dq_iq3_s(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq3_s* x = (const block_iq3_s*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint8_t* qs = x[ibs].qs + 8 * ib;
    const uint32_t grid1 = iq3s_grid[qs[2 * il + 0] | ((x[ibs].qh[ib] << (8 - 2 * il)) & 256)];
    const uint32_t grid2 = iq3s_grid[qs[2 * il + 1] | ((x[ibs].qh[ib] << (7 - 2 * il)) & 256)];
    const float d = (float) x[ibs].d * (1 + 2 * ((x[ibs].scales[ib / 2] >> 4 * (ib % 2)) & 0xf));
    const uint8_t signs = x[ibs].signs[4 * ib + il];
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * byte_of(grid1, j) * (signs & kmask_iq2xs[j + 0] ? -1.f : 1.f));
        y[j + 4] = cvt<dst_t>(d * byte_of(grid2, j) * (signs & kmask_iq2xs[j + 4] ? -1.f : 1.f));
    }
}
template<typename dst_t>
void dq_iq1_m(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq1_m* x = (const block_iq1_m*) vx;
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 8 * il;
    const uint16_t* sc = (const uint16_t*) x[ibs].scales;
    const int64_t ib16 = 2 * ib + il / 2;
    const float d = iq1m_scale(&x[ibs]) * (2 * ((sc[ib16 / 4] >> 3 * (ib16 % 4)) & 0x7) + 1);
    const float delta = x[ibs].qh[2 * ib + il / 2] & (0x08 << 4 * (il % 2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t g = iq1s_grid_gpu[x[ibs].qs[4 * ib + il] | (((x[ibs].qh[2 * ib + il / 2] >> 4 * (il % 2)) & 7) << 8)];
    const uint32_t grid32[2] = {g & 0x0f0f0f0f, (g >> 4) & 0x0f0f0f0f};
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(d * ((int8_t) byte_of(grid32[j / 4], j % 4) + delta));
}
template<typename dst_t>
void dq_iq4_nl(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_nl* x = (const block_iq4_nl*) vx + ibs * (QK_K / QK4_NL);
    const int64_t il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x[ib].qs + 4 * il;
    const float d = (float) x[ib].d;
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
template<typename dst_t>
void dq_q3_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q3_K* x = (const block_q3_K*) vx + ibs;
    for (int tt = tid; tt < 64; tt += 32) {
        const int r = tt / 4, t2 = r / 2, is0 = r % 2;
        const int l0 = 16 * is0 + 4 * (tt % 4);
        const int n = t2 / 4, j = t2 - 4 * n;
        const uint8_t m = (uint8_t) (1 << (4 * n + j));
        const int is = 8 * n + 2 * j + is0;
        const int shift = 2 * j;
        const int8_t us = is < 4  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 8] >> 0) & 3) << 4)) :
                          is < 8  ? (int8_t) ((x->scales[is - 0] & 0xF) | (((x->scales[is + 4] >> 2) & 3) << 4)) :
                          is < 12 ? (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is + 0] >> 4) & 3) << 4)) :
                                    (int8_t) ((x->scales[is - 8] >> 4) | (((x->scales[is - 4] >> 6) & 3) << 4));
        const float dl = (float) x->d * (us - 32);
        dst_t* y = yy + 128 * n + 32 * j;
        const uint8_t* q = x->qs + 32 * n;
        const uint8_t* hm = x->hmask;
        for (int l = l0; l < l0 + 4; ++l) y[l] = cvt<dst_t>(dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4)));
    }
}
template<typename dst_t>
void dq_iq4_xs(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_iq4_xs* x = (const block_iq4_xs*) vx + ibs;
    const int il = tid / 8, ib = tid % 8;
    dst_t* y = yy + 32 * ib + 4 * il;
    const uint8_t* q4 = x->qs + 16 * ib + 4 * il;
    const float d = (float) x->d * ((((x->scales_l[ib / 2] >> 4 * (ib % 2)) & 0xf) | (((x->scales_h >> 2 * ib) & 3) << 4)) - 32);
    for (int j = 0; j < 4; ++j) {
        y[j + 0] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] & 0xf]);
        y[j + 16] = cvt<dst_t>(d * kvalues_iq4nl[q4[j] >> 4]);
    }
}
template<typename dst_t>
void dq_q2_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q2_0* x = (const block_q2_0*) vx + ibs * 4;
    const int b = tid / 8, part = tid % 8;
    const float d = (float) x[b].d;
    for (int j = 0; j < 8; ++j) {
        const int i = part * 8 + j;
        const int code = (x[b].qs[i / 4] >> ((i % 4) * 2)) & 3;
        yy[b * 64 + i] = cvt<dst_t>(d * (float) (code - 1));
    }
}
inline void get_scale_min_k4(int j, const uint8_t* q, uint8_t& d, uint8_t& m) {
    if (j < 4) {
        d = q[j] & 63; m = q[j + 4] & 63;
    } else {
        d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}
template<typename dst_t>
void dq_q4_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q4_K* x = (const block_q4_K*) vx;
    const int64_t il = tid / 8, ir = tid % 8, is = 2 * il;
    const int n = 4;
    dst_t* y = yy + 64 * il + n * ir;
    const float dall = lo(x[ibs].dm);
    const float dmin = hi(x[ibs].dm);
    const uint8_t* q = x[ibs].qs + 32 * il + n * ir;
    uint8_t sc, m;
    get_scale_min_k4((int) is + 0, x[ibs].scales, sc, m);
    const float d1 = dall * sc, m1 = dmin * m;
    get_scale_min_k4((int) is + 1, x[ibs].scales, sc, m);
    const float d2 = dall * sc, m2 = dmin * m;
    for (int l = 0; l < n; ++l) {
        y[l + 0] = cvt<dst_t>(d1 * (q[l] & 0xF) - m1);
        y[l + 32] = cvt<dst_t>(d2 * (q[l] >> 4) - m2);
    }
}
template<typename dst_t>
void dq_q5_k(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_K* x = (const block_q5_K*) vx;
    for (int tt = tid; tt < 64; tt += 32) {
        const int il = tt / 16, ir = tt % 16, is = 2 * il;
        dst_t* y = yy + 64 * il + 2 * ir;
        const float dall = lo(x[ibs].dm);
        const float dmin = hi(x[ibs].dm);
        const uint8_t* ql = x[ibs].qs + 32 * il + 2 * ir;
        const uint8_t* qh = x[ibs].qh + 2 * ir;
        uint8_t sc, m;
        get_scale_min_k4(is + 0, x[ibs].scales, sc, m);
        const float d1 = dall * sc, m1 = dmin * m;
        get_scale_min_k4(is + 1, x[ibs].scales, sc, m);
        const float d2 = dall * sc, m2 = dmin * m;
        uint8_t hm = (uint8_t) (1 << (2 * il));
        y[0] = cvt<dst_t>(d1 * ((ql[0] & 0xF) + (qh[0] & hm ? 16 : 0)) - m1);
        y[1] = cvt<dst_t>(d1 * ((ql[1] & 0xF) + (qh[1] & hm ? 16 : 0)) - m1);
        hm <<= 1;
        y[32] = cvt<dst_t>(d2 * ((ql[0] >> 4) + (qh[0] & hm ? 16 : 0)) - m2);
        y[33] = cvt<dst_t>(d2 * ((ql[1] >> 4) + (qh[1] & hm ? 16 : 0)) - m2);
    }
}
template<typename dst_t>
void dq_q5_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_0* x = (const block_q5_0*) vx + ibs * (QK_K / QK5_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = (float) x[ib].d;
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        y[iqs] = cvt<dst_t>(((float) ((x[ib].qs[iqs] & 0xf) | xh_0) - 16.0f) * d);
        y[iqs + 16] = cvt<dst_t>(((float) ((x[ib].qs[iqs] >> 4) | xh_1) - 16.0f) * d);
    }
}
template<typename dst_t>
void dq_q5_1(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q5_1* x = (const block_q5_1*) vx + ibs * (QK_K / QK5_1);
    const int ib = tid % 8, il = tid / 8;
    const float dmx = lo(x[ib].dm), dmy = hi(x[ib].dm);
    uint32_t qh;
    memcpy(&qh, x[ib].qh, sizeof(qh));
    dst_t* y = yy + 32 * ib;
    for (int j = 0; j < 4; ++j) {
        const int iqs = 4 * il + j;
        const int xh_0 = ((qh >> (iqs + 0)) << 4) & 0x10;
        const int xh_1 = ((qh >> (iqs + 12))) & 0x10;
        y[iqs] = cvt<dst_t>((float) ((x[ib].qs[iqs] & 0xf) | xh_0) * dmx + dmy);
        y[iqs + 16] = cvt<dst_t>((float) ((x[ib].qs[iqs] >> 4) | xh_1) * dmx + dmy);
    }
}
template<typename dst_t>
void dq_q8_0(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const block_q8_0* x = (const block_q8_0*) vx + ibs * (QK_K / QK8_0);
    const int ib = tid % 8, il = tid / 8;
    const float d = (float) x[ib].d;
    dst_t* y = yy + 32 * ib + 8 * il;
    for (int j = 0; j < 8; ++j) y[j] = cvt<dst_t>(fm((float) x[ib].qs[8 * il + j], d));
}
template<typename dst_t>
void dq_bf16(const void* vx, int64_t ibs, dst_t* yy, int tid) {
    const uint16_t* x = (const uint16_t*) vx + ibs * 256 + tid * 8;
    for (int j = 0; j < 8; ++j) yy[tid * 8 + j] = cvt<dst_t>(sycl::bit_cast<float>((uint32_t) x[j] << 16));
}

template<typename dst_t>
inline void dq_dispatch(int ty, const void* vx, int64_t ibs, dst_t* y, int tid) {
    switch (ty) {
        case 16: dq_iq2_xxs(vx, ibs, y, tid); break;
        case 17: dq_iq2_xs(vx, ibs, y, tid); break;
        case 18: dq_iq3_xxs(vx, ibs, y, tid); break;
        case 20: dq_iq4_nl(vx, ibs, y, tid); break;
        case 21: dq_iq3_s(vx, ibs, y, tid); break;
        case 22: dq_iq2_s(vx, ibs, y, tid); break;
        case 29: dq_iq1_m(vx, ibs, y, tid); break;
        case 23: dq_iq4_xs(vx, ibs, y, tid); break;
        case 11: dq_q3_k(vx, ibs, y, tid); break;
        case 42: dq_q2_0(vx, ibs, y, tid); break;
        case 12: dq_q4_k(vx, ibs, y, tid); break;
        case 13: dq_q5_k(vx, ibs, y, tid); break;
        case 7: dq_q5_1(vx, ibs, y, tid); break;
        case 6: dq_q5_0(vx, ibs, y, tid); break;
        case 8: dq_q8_0(vx, ibs, y, tid); break;
        case 30: dq_bf16(vx, ibs, y, tid); break;
        default: break;
    }
}

// the 32 work-items of one 256-value superblock per work-group (no sub-group operations)
template<class F>
void launch_dq(sycl::queue& q, size_t blocks, F body) {
    if (blocks == 0) return;
    q.parallel_for(sycl::nd_range<1>(blocks * 32, 32),
                   [=](sycl::nd_item<1> it) { body((int64_t) it.get_group(0), (int) it.get_local_id(0)); });
}

template<typename dst_t>
void dequant_flat(sycl::queue& q, int ty, const void* vx, int64_t n, dst_t* y) {
    launch_dq(q, (size_t) (n / 256), [=](int64_t i, int tid) { dq_dispatch<dst_t>(ty, vx, i, y + i * QK_K, tid); });
}

bool is_iq(int t) {
    return t == 16 || t == 17 || t == 18 || t == 20 || t == 21 || t == 22 || t == 23 || t == 29 || t == 42 || t == 11 ||
           t == 12 || t == 13 || t == 7 || t == 6 || t == 8;
}
int gu_qk(int t) {
    switch (t) {
#define STRATA_QK(T) case T: return Fmt<T>::qk;
        STRATA_GU_FMTS(STRATA_QK)
#undef STRATA_QK
        default: return 0;
    }
}
int d_qk(int t) {
    switch (t) {
#define STRATA_QK(T) case T: return Fmt<T>::qk;
        STRATA_D_FMTS(STRATA_QK)
#undef STRATA_QK
        default: return 0;
    }
}

bool env_on(const char* name) {
    const char* v = std::getenv(name);
    return v != nullptr && v[0] != '\0' && v[0] != '0';
}
bool g_old_kernels = env_on("STRATA_OLD_IQ_MMVQ");

template<int TY>
void launch_mmvq(sycl::queue& q, const uint8_t* W, size_t rb, const block_q8_1* X, float* y, int n_in, int n_out,
                 int ncols) {
    const size_t grid = (size_t) ((n_out + 128 / MMVQ_W - 1) / (128 / MMVQ_W));
#define STRATA_MMVQ_PLAIN \
    launch16(q, grid, 128, [=](sycl::nd_item<1> it) { mmvq_kernel<TY>(it, W, rb, X, y, n_in, n_out, ncols); })
#define STRATA_MMVQ_MULTI(NC) \
    launch16(q, grid, 128, [=](sycl::nd_item<1> it) { mmvq_multi_kernel<TY, NC>(it, W, rb, X, y, n_in, n_out, ncols); })
    if constexpr (!kSplit<TY>) STRATA_MMVQ_PLAIN;
    else if (g_old_kernels) STRATA_MMVQ_PLAIN;
    else if (ncols <= 1) STRATA_MMVQ_MULTI(1);
    else if (ncols == 2) STRATA_MMVQ_MULTI(2);
    else if (ncols <= 4) STRATA_MMVQ_MULTI(4);
    else STRATA_MMVQ_MULTI(8);   // 8 at a time past 8
#undef STRATA_MMVQ_PLAIN
#undef STRATA_MMVQ_MULTI
}

template<int TG>
void launch_gu(sycl::queue& q, Grid2 G, const unsigned long long* grp_ptr, const int32_t* grp_start,
               const int32_t* n_groups, const int32_t* ent_tok, const block_q8_1* X, const NativeExpertLayout& L,
               float* gate, float* up) {
    const NativeExpertLayout l = L;
    auto run = [&](auto multi) {
        constexpr bool M = decltype(multi)::value;
        launch16(q, G.gx * G.gy, 128, [=](sycl::nd_item<1> it) {
            native_gu_kernel<TG, M>(it, G, grp_ptr, grp_start, n_groups, ent_tok, X, l, gate, up);
        });
    };
    if constexpr (!kSplit<TG>) run(std::false_type{});
    else if (g_old_kernels) run(std::false_type{});
    else run(std::true_type{});
}

template<int TD>
void launch_down(sycl::queue& q, Grid2 G, const unsigned long long* grp_ptr, const int32_t* grp_start,
                 const int32_t* n_groups, const int32_t* ent_dst, const block_q8_1* hq, const NativeExpertLayout& L,
                 float* out) {
    const NativeExpertLayout l = L;
    auto run = [&](auto multi) {
        constexpr bool M = decltype(multi)::value;
        launch16(q, G.gx * G.gy, 128, [=](sycl::nd_item<1> it) {
            native_down_kernel<TD, M>(it, G, grp_ptr, grp_start, n_groups, ent_dst, hq, l, out);
        });
    };
    if constexpr (!kSplit<TD>) run(std::false_type{});
    else if (g_old_kernels) run(std::false_type{});
    else run(std::true_type{});
}

}  // namespace

void iq_set_old_kernels(bool old) { g_old_kernels = old; }
bool iq_old_kernels() { return g_old_kernels; }

bool iq_supported(int t) noexcept { return is_iq(t); }
bool embed_type_supported(int t) noexcept { return is_iq(t) || t == 30; }

size_t iq_row_bytes(int t, int64_t n) noexcept {
    switch (t) {
        case 16: return (size_t) (n / 256) * sizeof(block_iq2_xxs);
        case 17: return (size_t) (n / 256) * sizeof(block_iq2_xs);
        case 18: return (size_t) (n / 256) * sizeof(block_iq3_xxs);
        case 20: return (size_t) (n / 32) * sizeof(block_iq4_nl);
        case 21: return (size_t) (n / 256) * sizeof(block_iq3_s);
        case 22: return (size_t) (n / 256) * sizeof(block_iq2_s);
        case 29: return (size_t) (n / 256) * sizeof(block_iq1_m);
        case 23: return (size_t) (n / 256) * sizeof(block_iq4_xs);
        case 11: return (size_t) (n / 256) * sizeof(block_q3_K);
        case 42: return (size_t) (n / 64) * sizeof(block_q2_0);
        case 12: return (size_t) (n / 256) * sizeof(block_q4_K);
        case 13: return (size_t) (n / 256) * sizeof(block_q5_K);
        case 7: return (size_t) (n / 32) * sizeof(block_q5_1);
        case 6: return (size_t) (n / 32) * sizeof(block_q5_0);
        case 8: return (size_t) (n / 32) * sizeof(block_q8_0);
        case 30: return (size_t) n * 2;
        default: return 0;
    }
}

void quantize_q8_1_rows(const float* x, int64_t n_rows, int64_t n_cols, void* y, void* stream) {
    const long long n = (long long) n_rows * n_cols;
    if (n <= 0) return;
    block_q8_1* Y = (block_q8_1*) y;
    launch(sycl_runtime::queue_from_stream(stream), (size_t) ((n + 255) / 256), 256, [=](sycl::nd_item<1> it) {
        const long long i = (long long) it.get_global_id(0);
        if (i >= n) return;   // n is a multiple of 32: whole sub-groups leave together
        q8_1_store(it.get_sub_group(), x[i], Y, i);
    });
    check("quantize_q8_1_rows");
}

void iq_mmvq(int t, const void* w, const void* x_q8_1, float* y, int n_in, int n_out, int ncols, void* stream) {
    const size_t rb = iq_row_bytes(t, n_in);
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    const auto* W = (const uint8_t*) w;
    const auto* X = (const block_q8_1*) x_q8_1;
    switch (t) {
#define STRATA_MMVQ(T) case T: launch_mmvq<T>(q, W, rb, X, y, n_in, n_out, ncols); break;
        STRATA_MMVQ_FMTS(STRATA_MMVQ)
#undef STRATA_MMVQ
        default: std::fprintf(stderr, "iq_mmvq: type %d is not supported\n", t); std::exit(1);
    }
    check("iq_mmvq");
}

void iq_dequant_f16(int t, const void* src, int64_t n, uint16_t* dst, void* stream) {
    if (n % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_f16: bad arguments\n"); std::exit(1); }
    dequant_flat<uint16_t>(sycl_runtime::queue_from_stream(stream), t, src, n, dst);
    check("iq_dequant_f16");
}

void iq_embed_rows(int t, const void* table, size_t row_bytes, const int32_t* tokens, int64_t n_tok, int64_t n_embd,
                   float* out, void* stream) {
    if (n_tok <= 0) return;
    if (n_embd % 256 != 0 || !embed_type_supported(t)) { std::fprintf(stderr, "iq_embed_rows: bad arguments\n"); std::exit(1); }
    const int64_t per_row = n_embd / 256;
    const uint8_t* tab = (const uint8_t*) table;
    launch_dq(sycl_runtime::queue_from_stream(stream), (size_t) (per_row * n_tok), [=](int64_t i, int tid) {
        const int64_t tk = i / per_row, b = i % per_row;
        const uint8_t* row = tab + (size_t) tokens[tk] * row_bytes;
        dq_dispatch<float>(t, row, b, out + (size_t) tk * n_embd + b * QK_K, tid);
    });
    check("iq_embed_rows");
}

void iq_dequant_f32(int t, const void* src, int64_t n, float* dst, void* stream) {
    if (n % 256 != 0 || !embed_type_supported(t)) { std::fprintf(stderr, "iq_dequant_f32: bad arguments\n"); std::exit(1); }
    dequant_flat<float>(sycl_runtime::queue_from_stream(stream), t, src, n, dst);
    check("iq_dequant_f32");
}

void iq_dequant_gu_f16(int t, const void* gate, const void* up, int64_t n_ff, int64_t n_embd, uint16_t* dst, void* stream) {
    if (n_embd % 256 != 0 || !is_iq(t)) { std::fprintf(stderr, "iq_dequant_gu_f16: type %d / %lld\n", t, (long long) n_embd); std::exit(1); }
    const int64_t per_row = n_embd / 256, per_mat = n_ff * per_row;
    launch_dq(sycl_runtime::queue_from_stream(stream), (size_t) (2 * per_mat), [=](int64_t bid, int tid) {
        const int parity = (int) (bid / per_mat);
        const int64_t i = bid % per_mat;
        const int64_t r = i / per_row, c = i % per_row;
        dq_dispatch<uint16_t>(t, parity ? up : gate, i, dst + ((2 * r + parity) * per_row + c) * QK_K, tid);
    });
    check("iq_dequant_gu_f16");
}

bool native_expert_supported(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) noexcept {
    const int qg = gu_qk(gu_type), qd = d_qk(d_type);
    return qg > 0 && qd > 0 && is_iq(gu_type) && is_iq(d_type) && n_embd % qg == 0 && n_ff % qd == 0 &&
           n_embd % 256 == 0 && (n_ff * n_embd) % 256 == 0;
}

NativeExpertLayout native_expert_layout(int gu_type, int d_type, int64_t n_embd, int64_t n_ff) {
    NativeExpertLayout L;
    L.gu_type = gu_type;
    L.d_type = d_type;
    L.n_embd = n_embd;
    L.n_ff = n_ff;
    L.gu_row = iq_row_bytes(gu_type, n_embd);
    L.d_row = iq_row_bytes(d_type, n_ff);
    L.up_off = (size_t) n_ff * L.gu_row;
    L.down_off = 2 * L.up_off;
    L.bytes = L.down_off + (size_t) n_embd * L.d_row;
    return L;
}

size_t native_expert_scratch_bytes(int64_t cap, int64_t n_ff) {
    const size_t f = (size_t) cap * (size_t) n_ff * sizeof(float);
    return 3 * ((f + 255) & ~(size_t) 255) + (((size_t) cap * (size_t) (n_ff / 32) * sizeof(block_q8_1) + 255) & ~(size_t) 255);
}

namespace {
bool g_grouped_v1 = env_on("STRATA_GROUPED_V1");
}  // namespace

void native_grouped_set_v1(bool v1) { g_grouped_v1 = v1; }

void native_expert_grouped(const NativeExpertLayout& L, const unsigned long long* grp_ptr, const int32_t* grp_start,
                           const int32_t* n_groups, const int32_t* ent_dst, const int32_t* ent_tok, int64_t cap_groups,
                           int64_t cap_entries, const void* x_q8_1, void* scratch, float* out, void* stream,
                           int64_t grid_groups) {
    if (cap_groups <= 0 || cap_entries <= 0) return;
    if (L.n_ff % 32 != 0) { std::fprintf(stderr, "native_expert_grouped: n_ff %lld\n", (long long) L.n_ff); std::exit(1); }
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    const size_t f = (size_t) cap_entries * (size_t) L.n_ff * sizeof(float), fa_ = (f + 255) & ~(size_t) 255;
    float* gate = (float*) scratch;
    float* up = (float*) ((uint8_t*) scratch + fa_);
    float* h = (float*) ((uint8_t*) scratch + 2 * fa_);
    block_q8_1* hq = (block_q8_1*) ((uint8_t*) scratch + 3 * fa_);
    const auto* X = (const block_q8_1*) x_q8_1;
    const bool v1 = g_grouped_v1;
    const int64_t gy = (v1 || grid_groups <= 0 || grid_groups > cap_groups) ? cap_groups : grid_groups;
    const Grid2 ggu{(size_t) ((2 * L.n_ff + GU_ROWS - 1) / GU_ROWS), (size_t) gy};
    switch (L.gu_type) {
#define STRATA_GU(T) case T: launch_gu<T>(q, ggu, grp_ptr, grp_start, n_groups, ent_tok, X, L, gate, up); break;
        STRATA_GU_FMTS(STRATA_GU)
#undef STRATA_GU
        default: std::fprintf(stderr, "native_expert_grouped: gate/up type %d\n", L.gu_type); std::exit(1);
    }
    check("native_expert_grouped/gu");
    const long long nh = (long long) cap_entries * L.n_ff;
    const size_t nblk = (size_t) ((nh + 255) / 256);
    if (v1) {
        launch(q, nblk, 256, [=](sycl::nd_item<1> it) {
            const long long i = (long long) it.get_global_id(0);
            if (i < nh) h[i] = swiglu(gate[i], up[i]);
        });
        launch(q, nblk, 256, [=](sycl::nd_item<1> it) {
            const long long i = (long long) it.get_global_id(0);
            if (i >= nh) return;
            q8_1_store(it.get_sub_group(), h[i], hq, i);
        });
    } else {
        // over the call's own entries only, grid-strided; n_ff % 32 == 0 keeps the bounds sub-group-uniform
        const int n_ff = (int) L.n_ff;
        launch(q, nblk, 256, [=](sycl::nd_item<1> it) {
            const long long lo_ = (long long) grp_start[0] * n_ff, hi_ = (long long) grp_start[*n_groups] * n_ff;
            const long long stride = (long long) it.get_global_range(0);
            for (long long i = lo_ + (long long) it.get_global_id(0); i < hi_; i += stride)
                q8_1_store(it.get_sub_group(), swiglu(gate[i], up[i]), hq, i);
        });
    }
    check("native_expert_grouped/swiglu");
    const Grid2 gd{(size_t) ((L.n_embd + D_ROWS - 1) / D_ROWS), (size_t) gy};
    switch (L.d_type) {
#define STRATA_DOWN(T) case T: launch_down<T>(q, gd, grp_ptr, grp_start, n_groups, ent_dst, hq, L, out); break;
        STRATA_D_FMTS(STRATA_DOWN)
#undef STRATA_DOWN
        default: std::fprintf(stderr, "native_expert_grouped: down type %d\n", L.d_type); std::exit(1);
    }
    check("native_expert_grouped/down");
}

}  // namespace strata::kernels
