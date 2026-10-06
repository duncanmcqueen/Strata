// src/kernels/cuda/fused_gr.cu - see include/strata/kernels/fused_gr.hpp.

#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/bf16_bits.hpp"

#include "strata/sycl_runtime/queue_bridge.hpp"
#include <cuda_runtime.h>
#include <stdexcept>

#include <atomic>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <string>
#include <vector>

namespace strata::kernels {
namespace {

struct uint4 {
    uint32_t x, y, z, w;
};
struct float4 {
    float x, y, z, w;
};
float4 make_float4(float x, float y, float z, float w) { return {x, y, z, w}; }

constexpr int N = 2560;   // n_embd
constexpr int HC = 4;     // streams
constexpr int D = N * HC; // 10240
constexpr int LR = 320;   // hc_lr
constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
constexpr int DOWN_BLOCKS = LR / WARPS; // 40 blocks of 8 rows; one more for the inject rows
constexpr int UP_COLS = 32;             // columns d per `up` block (x 4 streams = 128 rows)
constexpr int UP_BLOCKS = N / UP_COLS;  // 80

inline float warp_sum(sycl::nd_item<1> it, float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1)
        v += sycl::permute_group_by_xor(it.get_sub_group(), v, o);
    return v;
}
inline float sigmoidf_(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

// 8 bf16 packed in a uint4 against 8 floats.
inline float dot8(const uint4 w, const float *x) {
    float acc = 0.0f;
    const uint32_t v[4] = {w.x, w.y, w.z, w.w};
#pragma unroll
    for (int j = 0; j < 4; ++j) {
        acc = sycl::fma(sycl::bit_cast<float>(v[j] << 16), x[2 * j], acc);
        acc = sycl::fma(sycl::bit_cast<float>(v[j] & 0xffff0000u), x[2 * j + 1], acc);
    }
    return acc;
}

void gr_down_kernel(sycl::queue &queue, size_t groups, FusedGrArgs a) {
    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> xn_local(D, cgh), part_local(WARPS * HC, cgh), rs_local(HC, cgh);
        cgh.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                             auto *xn = &xn_local[0];
                             auto *part = &part_local[0];
                             auto *s_rs = &rs_local[0];
                             const int t = it.get_local_id(0), lane = t & 31, warp = t >> 5;
                             float gw[HC];
#pragma unroll
                             for (int c = 0; c < HC; ++c)
                                 gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float)HC) : 0.0f;
                             // 1. R' * w_norm into shared memory, and the per-stream sums of
                             // squares of R'.
                             float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
                             for (int i = t * 4; i < D; i += THREADS * 4) {
                                 const int c = i / N, d = i - c * N;
                                 float4 r = *reinterpret_cast<const float4 *>(a.R + i);
                                 if (a.apply) {
                                     const float4 b = *reinterpret_cast<const float4 *>(a.bo_prev + d);
                                     r.x = sycl::fma(b.x, gw[c], r.x);
                                     r.y = sycl::fma(b.y, gw[c], r.y);
                                     r.z = sycl::fma(b.z, gw[c], r.z);
                                     r.w = sycl::fma(b.w, gw[c], r.w);
                                 }
                                 const float4 g = *reinterpret_cast<const float4 *>(a.w_norm + i);
                                 float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
                                 for (int cc = 0; cc < HC; ++cc)
                                     if (cc == c)
                                         ss[cc] += sq;
                                 *reinterpret_cast<float4 *>(xn + i) =
                                     make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
                             }
#pragma unroll
                             for (int c = 0; c < HC; ++c) {
                                 const float v = warp_sum(it, ss[c]);
                                 if (lane == 0)
                                     part[warp * HC + c] = v;
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             if (t < HC) {
                                 float s = 0.0f;
                                 for (int w = 0; w < WARPS; ++w)
                                     s += part[w * HC + t];
                                 s_rs[t] = sycl::rsqrt(s / (float)N + a.eps);
                                 if (it.get_group(0) == 0)
                                     a.rs[t] = s_rs[t];
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             for (int i = t; i < D; i += THREADS)
                                 xn[i] *= s_rs[i / N];
                             it.barrier(sycl::access::fence_space::local_space);
                             // 2. one warp per output row: 10240 bf16 = 1280 chunks of 8, 40 per
                             // lane.
                             const bool inject_block = it.get_group(0) == DOWN_BLOCKS;
                             const int row = inject_block ? warp : it.get_group(0) * WARPS + warp;
                             if (inject_block && (a.w_inject == nullptr || warp >= HC))
                                 return;
                             const uint16_t *wrow = (inject_block ? a.w_inject : a.w_down) + (size_t)row * D;
                             const uint4 *w4 = reinterpret_cast<const uint4 *>(wrow);
                             float acc = 0.0f;
#pragma unroll 4
                             for (int j = lane; j < D / 8; j += 32)
                                 acc += dot8(*(w4 + j), xn + j * 8);
                             acc = warp_sum(it, acc);
                             if (lane != 0)
                                 return;
                             if (inject_block) {
                                 a.inject_out[row] = acc;
                             } else {
                                 const float x = acc / (float)HC;
                                 a.lo[row] = x / (1.0f + sycl::exp(-x));
                             }
                         });
    });
}

void gr_up_kernel(sycl::queue &queue, size_t groups, FusedGrArgs a) {
    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> lo_local(LR, cgh), g_local(HC * UP_COLS, cgh);
        cgh.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                             auto *lo = &lo_local[0];
                             auto *g = &g_local[0];
                             const int t = it.get_local_id(0), lane = t & 31, warp = t >> 5;
                             const int d0 = it.get_group(0) * UP_COLS;
                             for (int k = t; k < LR; k += THREADS)
                                 lo[k] = a.lo[k];
                             it.barrier(sycl::access::fence_space::local_space);
                             // 128 rows (4 streams x 32 columns), 16 per warp: 320 bf16 = 40
                             // chunks of 8.
                             for (int r = warp; r < HC * UP_COLS; r += WARPS) {
                                 const int c = r / UP_COLS, dd = r - c * UP_COLS, i = c * N + d0 + dd;
                                 const uint4 *w4 = reinterpret_cast<const uint4 *>(a.w_up + (size_t)i * LR);
                                 float acc = dot8(*(w4 + lane), lo + lane * 8);
                                 if (lane < LR / 8 - 32)
                                     acc += dot8(*(w4 + 32 + lane), lo + (32 + lane) * 8);
                                 acc = warp_sum(it, acc);
                                 if (lane == 0) {
                                     float rv = a.R[i];
                                     if (a.apply) {
                                         rv = sycl::fma(a.bo_prev[d0 + dd],
                                                        2.0f * sigmoidf_(a.inj_prev[c] / (float)HC), rv);
                                         a.R_out[i] = rv; // this block owns column d0+dd of every stream
                                     }
                                     const float x = rv * a.w_norm[i] * a.rs[c];
                                     g[c * UP_COLS + dd] = x * sigmoidf_(acc);
                                 }
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             if (t < UP_COLS) {
                                 float s = 0.0f;
#pragma unroll
                                 for (int c = 0; c < HC; ++c)
                                     s += g[c * UP_COLS + t];
                                 a.mixed[d0 + t] = s / (float)HC;
                             }
                         });
    });
}

struct GrMulti {
    FusedGrArgs a[kFusedGrMaxT];
    float *xn;
    int T;
};

// Step 1 of `gr_down_kernel`, one block per token, same threads and reduction order: rs[t] and xn[t] to
// global.
void gr_norm_multi_kernel(sycl::queue &queue, size_t groups, GrMulti m) {
    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> part_local(WARPS * HC, cgh), rs_local(HC, cgh);
        cgh.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                             auto *part = &part_local[0];
                             auto *s_rs = &rs_local[0];
                             const FusedGrArgs &a = m.a[it.get_group(0)];
                             float *xn = m.xn + (size_t)it.get_group(0) * D;
                             const int t = it.get_local_id(0), lane = t & 31, warp = t >> 5;
                             float gw[HC];
#pragma unroll
                             for (int c = 0; c < HC; ++c)
                                 gw[c] = a.apply ? 2.0f * sigmoidf_(a.inj_prev[c] / (float)HC) : 0.0f;
                             float ss[HC] = {0.0f, 0.0f, 0.0f, 0.0f};
                             for (int i = t * 4; i < D; i += THREADS * 4) {
                                 const int c = i / N, d = i - c * N;
                                 float4 r = *reinterpret_cast<const float4 *>(a.R + i);
                                 if (a.apply) {
                                     const float4 b = *reinterpret_cast<const float4 *>(a.bo_prev + d);
                                     r.x = sycl::fma(b.x, gw[c], r.x);
                                     r.y = sycl::fma(b.y, gw[c], r.y);
                                     r.z = sycl::fma(b.z, gw[c], r.z);
                                     r.w = sycl::fma(b.w, gw[c], r.w);
                                 }
                                 const float4 g = *reinterpret_cast<const float4 *>(a.w_norm + i);
                                 float sq = r.x * r.x + r.y * r.y + r.z * r.z + r.w * r.w;
#pragma unroll
                                 for (int cc = 0; cc < HC; ++cc)
                                     if (cc == c)
                                         ss[cc] += sq;
                                 *reinterpret_cast<float4 *>(xn + i) =
                                     make_float4(r.x * g.x, r.y * g.y, r.z * g.z, r.w * g.w);
                             }
#pragma unroll
                             for (int c = 0; c < HC; ++c) {
                                 const float v = warp_sum(it, ss[c]);
                                 if (lane == 0)
                                     part[warp * HC + c] = v;
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             if (t < HC) {
                                 float s = 0.0f;
                                 for (int w = 0; w < WARPS; ++w)
                                     s += part[w * HC + t];
                                 s_rs[t] = sycl::rsqrt(s / (float)N + a.eps);
                                 a.rs[t] = s_rs[t];
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             for (int i = t; i < D; i += THREADS)
                                 xn[i] *= s_rs[i / N];
                         });
    });
}

// Step 2 of `gr_down_kernel` for T tokens.  One warp per row (so each lane accumulates the same chunks in the
// same order as the single-token kernel); per tile the lane's weight chunks are loaded BEFORE the activation
// tile is staged, so the DRAM and L2 traffic are in flight together.
//
// The multi-token kernels run on 16-wide sub-groups: at 32 lanes the A770 has 32 registers per lane and these
// kernels spilled (1,632 bytes per thread in down, 3,584 in up; IGC's dumps).  Work-item l of a 16-wide sub-group is
// the CUDA warp's lanes l and l + 16, each with its own accumulator in the CUDA order; warp_sum16 replays the 32-lane
// butterfly (its offset-16 step is exactly that pair, the others stay within a half), so every sum - and so every
// output - is the 32-lane kernel's bit for bit (gr_parity: multi-token against single-token, bitwise).
constexpr int MT = 128;   // work-items per group: 8 virtual 32-lane warps

inline float warp_sum16(const sycl::sub_group &sg, float a, float b) {
    float v = a + b;
#pragma unroll
    for (int o = 8; o > 0; o >>= 1)
        v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}

// The 1280-float tile preserves ascending chunk order and fits eight tokens in
// 40 KiB of local memory. Split/staged CUDA variants are not used on SYCL.
template <int TILEV, int MAX_T = kFusedGrMaxT>
void gr_down_multi_kernel(sycl::queue &queue, size_t groups, GrMulti m) {
    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> tile_local(m.T * TILEV, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(groups * MT, MT),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                constexpr int TQ = TILEV / 8 / 32; // uint4 weight chunks per CUDA lane per tile
                auto *tile = &tile_local[0];       // [T][TILEV]
                const int t = it.get_local_id(0), l16 = t & 15, warp = t >> 4;
                const int T = m.T;
                const bool inject_block = it.get_group(0) == DOWN_BLOCKS;
                const int row = inject_block ? warp : it.get_group(0) * WARPS + warp;
                const bool active = !(inject_block && (m.a[0].w_inject == nullptr || warp >= HC));
                const uint16_t *wrow = (inject_block && m.a[0].w_inject ? m.a[0].w_inject : m.a[0].w_down) +
                                       (size_t)(active ? row : 0) * D;
                const uint4 *w4 = reinterpret_cast<const uint4 *>(wrow);
                float acc_a[MAX_T], acc_b[MAX_T];   // CUDA lanes l16 and l16 + 16
#pragma unroll
                for (int k = 0; k < MAX_T; ++k)
                    acc_a[k] = acc_b[k] = 0.0f;
                for (int base = 0; base < D; base += TILEV) {
                    uint4 wa[TQ];
                    if (active) {
#pragma unroll
                        for (int q = 0; q < TQ; ++q)
                            wa[q] = *(w4 + base / 8 + l16 + 32 * q);
                    }
                    it.barrier(sycl::access::fence_space::local_space); // the previous tile is consumed
                    const float4 *src4 = reinterpret_cast<const float4 *>(m.xn);
                    float4 *tile4 = reinterpret_cast<float4 *>(tile);
                    for (int i = t; i < T * (TILEV / 4); i += MT) {
                        const int k = i / (TILEV / 4), off = i - k * (TILEV / 4);
                        tile4[i] = src4[((size_t)k * D + base) / 4 + off];
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    if (!active)
                        continue;
                    // lane l16's chunks, then lane l16 + 16's (each accumulator keeps the CUDA order; one lane's
                    // weights live at a time)
#pragma unroll
                    for (int q = 0; q < TQ; ++q) {
                        const int ja = l16 + 32 * q;
#pragma unroll
                        for (int k = 0; k < MAX_T; ++k)
                            if (k < T)
                                acc_a[k] += dot8(wa[q], tile + k * TILEV + ja * 8);
                    }
#pragma unroll
                    for (int q = 0; q < TQ; ++q) {
                        const int jb = l16 + 16 + 32 * q;
                        const uint4 wb = *(w4 + base / 8 + jb);
#pragma unroll
                        for (int k = 0; k < MAX_T; ++k)
                            if (k < T)
                                acc_b[k] += dot8(wb, tile + k * TILEV + jb * 8);
                    }
                }
                if (!active)
                    return;
                const sycl::sub_group sg = it.get_sub_group();
                float s[MAX_T];
#pragma unroll
                for (int k = 0; k < MAX_T; ++k)
                    s[k] = k < T ? warp_sum16(sg, acc_a[k], acc_b[k]) : 0.0f;
                // CUDA lane k writes token k (k < 8: work-item k)
#pragma unroll
                for (int k = 0; k < MAX_T; ++k) {
                    if (k >= T || l16 != k)
                        continue;
                    if (inject_block) {
                        m.a[k].inject_out[row] = s[k];
                    } else {
                        const float x = s[k] / (float)HC;
                        m.a[k].lo[row] = x / (1.0f + sycl::exp(-x));
                    }
                }
            });
    });
}

#ifndef STRATA_UPM_COLS
#define STRATA_UPM_COLS 16
#endif
constexpr int UPM_COLS = STRATA_UPM_COLS; // columns per block (x 4 streams = 64 rows, 8 per warp)
constexpr int UPM_BLOCKS = N / UPM_COLS;

// `gr_up_kernel` for T tokens: each row of w_up read once; the T dots reduced by the butterfly so every lane holds
// every sum, and CUDA lane k runs token k's epilogue - the T epilogues in parallel instead of one after another.
void gr_up_multi_kernel(sycl::queue &queue, size_t groups, GrMulti m) {
    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> lo_local(kFusedGrMaxT * LR, cgh),
            g_local(kFusedGrMaxT * HC * UPM_COLS, cgh);
        cgh.parallel_for(sycl::nd_range<1>(groups * MT, MT),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
                             auto *lo = &lo_local[0];
                             auto *g = &g_local[0];
                             const int t = it.get_local_id(0), l16 = t & 15, warp = t >> 4;
                             const sycl::sub_group sg = it.get_sub_group();
                             const int T = m.T;
                             const int d0 = it.get_group(0) * UPM_COLS;
                             for (int i = t; i < T * LR; i += MT)
                                 lo[i] = m.a[i / LR].lo[i % LR];
                             it.barrier(sycl::access::fence_space::local_space);
                             for (int r = warp; r < HC * UPM_COLS; r += WARPS) {
                                 const int c = r / UPM_COLS, dd = r - c * UPM_COLS, i = c * N + d0 + dd;
                                 const uint4 *w4 =
                                     reinterpret_cast<const uint4 *>(m.a[0].w_up + (size_t)i * LR);
                                 // CUDA lane l16: chunks l16 and (l16 < 8) 32 + l16; CUDA lane l16 + 16: chunk l16 + 16
                                 const uint4 wa = *(w4 + l16);
                                 const uint4 wb = l16 < LR / 8 - 32 ? *(w4 + 32 + l16) : uint4{0, 0, 0, 0};
                                 const uint4 wc = *(w4 + 16 + l16);
                                 // the epilogue inputs of this lane's token, fetched while the dots run
                                 float rv = 0.0f, wn = 0.0f, rsc = 0.0f, bo = 0.0f, ip = 0.0f;
                                 bool apply = false;
                                 if (l16 < T) {
                                     const FusedGrArgs &a = m.a[l16];
                                     rv = a.R[i];
                                     wn = a.w_norm[i];
                                     rsc = a.rs[c];
                                     apply = a.apply;
                                     if (apply) {
                                         bo = a.bo_prev[d0 + dd];
                                         ip = a.inj_prev[c];
                                     }
                                 }
                                 float mine = 0.0f;
#pragma unroll
                                 for (int k = 0; k < kFusedGrMaxT; ++k) {
                                     if (k >= T)
                                         break;
                                     float acc = dot8(wa, (lo + k * LR) + l16 * 8);
                                     if (l16 < LR / 8 - 32)
                                         acc += dot8(wb, (lo + k * LR) + (32 + l16) * 8);
                                     const float acc_hi = dot8(wc, (lo + k * LR) + (16 + l16) * 8);
                                     acc = warp_sum16(sg, acc, acc_hi);
                                     if (l16 == k)
                                         mine = acc;
                                 }
                                 if (l16 < T) {
                                     if (apply) {
                                         rv = sycl::fma(bo, 2.0f * sigmoidf_(ip / (float)HC), rv);
                                         m.a[l16].R_out[i] = rv;
                                     }
                                     const float x = rv * wn * rsc;
                                     g[(l16 * HC + c) * UPM_COLS + dd] = x * sigmoidf_(mine);
                                 }
                             }
                             it.barrier(sycl::access::fence_space::local_space);
                             for (int i = t; i < T * UPM_COLS; i += MT) {
                                 const int k = i / UPM_COLS, col = i - k * UPM_COLS;
                                 float s = 0.0f;
#pragma unroll
                                 for (int c = 0; c < HC; ++c)
                                     s += g[(k * HC + c) * UPM_COLS + col];
                                 m.a[k].mixed[d0 + col] = s / (float)HC;
                             }
                         });
    });
}

} // namespace
bool fused_gr_supported(int64_t n, int64_t hc, int64_t lr) { return n == N && hc == HC && lr == LR; }
void fused_gr_read(const FusedGrArgs &a, void *stream) {
    if (!a.R || !a.w_norm || !a.w_down || !a.w_up || !a.lo || !a.rs || !a.mixed ||
        (a.w_inject && !a.inject_out) || (a.apply && (!a.bo_prev || !a.inj_prev || !a.R_out)) ||
        (a.apply && a.inj_prev == a.inject_out))
        throw std::invalid_argument("fused_gr_read: invalid arguments");
    auto &queue = sycl_runtime::queue_from_stream(stream);
    gr_down_kernel(queue, DOWN_BLOCKS + 1, a);
    gr_up_kernel(queue, UP_BLOCKS, a);
}
// Plain weight-sharing path: 1280 activation floats per token, at most 40 KiB local memory.
void fused_gr_read_multi(const FusedGrArgs *a, int n_tok, float *xn_scratch, void *stream,
                         unsigned long long *stamp_buf, int stamp_i0) {
    if (!a || n_tok < 1 || n_tok > kFusedGrMaxT || !xn_scratch)
        throw std::invalid_argument("fused_gr_read_multi: invalid arguments");
    GrMulti m{};
    m.T = n_tok;
    m.xn = xn_scratch;
    for (int t = 0; t < n_tok; ++t) {
        const auto &x = a[t];
        if (!x.R || !x.w_norm || !x.w_down || !x.w_up || !x.lo || !x.rs || !x.mixed ||
            (x.w_inject && !x.inject_out) || (x.apply && (!x.bo_prev || !x.inj_prev || !x.R_out)) ||
            (x.apply && x.inj_prev == x.inject_out) || x.w_down != a[0].w_down || x.w_up != a[0].w_up ||
            x.w_inject != a[0].w_inject || x.w_norm != a[0].w_norm)
            throw std::invalid_argument("fused_gr_read_multi: invalid token arguments or weights");
        m.a[t] = x;
    }
    auto &queue = sycl_runtime::queue_from_stream(stream);
    gr_norm_multi_kernel(queue, n_tok, m);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0, stream);   // the profile's stamps: after the norm and the down
    gr_down_multi_kernel<1280>(queue, DOWN_BLOCKS + 1, m);
    if (stamp_buf) gpu_stamp(stamp_buf, stamp_i0 + 1, stream);
    gr_up_multi_kernel(queue, UPM_BLOCKS, m);
}
int fused_gr_variant() { return 1; }
void fused_gr_check() {
    // Only the plain implementation is available; no device-specific variant
    // selection.
}
} // namespace strata::kernels
