// SYCL port of the CUDA implementation; layouts and reduction order are retained.
// src/kernels/cuda/qsa_select.cu - see include/strata/kernels/qsa_select.hpp.

#include "strata/kernels/qsa_select.hpp"
#include <cstdlib>
#include <cstring>

#include "kv_helpers.hpp"
#include <cuda_runtime.h>

#include <cfloat>
#include <cstdio>
#include <cstdlib>
#include <cstring>

namespace strata::kernels {
namespace {
using namespace sycl_kv;

constexpr int IDX_DIM = 128, IDX_HEADS = 4, R = 4;
constexpr int SCORE_WARPS = 8;
constexpr int TOPK_T = 256;

inline uint32_t order_key(float s) {
    const float v = s + 0.0f;
    if (!(v == v))
        return 0u;
    const uint32_t b = sycl::bit_cast<uint32_t>(v);
    return (b & 0x80000000u) ? ~b : (b | 0x80000000u);
}

void block_scores_kernel_impl(sycl::nd_item<1> it, Grid grid, Grid block, const float *__restrict__ pooled,
                              const float *__restrict__ dead, const float *__restrict__ q_idx,
                              const int32_t *__restrict__ steps, int64_t max_blocks,
                              float *__restrict__ out) {

    const int64_t qi = group_y(it, grid);
    const int32_t *st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid];
    const int64_t b = (int64_t)group_x(it, grid) * SCORE_WARPS + (local_x(it, block) >> 5);
    if (b > n_bid || b >= max_blocks)
        return;
    const int lane = local_x(it, block) & 31;
    const float *key = (b == n_bid) ? dead : pooled + b * IDX_DIM;
    const float4 k4 = *reinterpret_cast<const float4 *>(key + lane * 4);
    const float *q = q_idx + qi * IDX_HEADS * IDX_DIM + lane * 4;
    float score = 0.0f;
#pragma unroll
    for (int h = 0; h < IDX_HEADS; ++h) {
        const float4 q4 = *reinterpret_cast<const float4 *>(q + h * IDX_DIM);
        float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
        for (int o = 16; o > 0; o >>= 1)
            d += xor_lane(it, d, o);
        score += d > 0.0f ? d : 0.0f;
    }
    if (lane == 0) {
        if (b == n_bid && n_kv % R != 0)
            score += 1e9f;
        out[qi * max_blocks + b] = score;
    }
}

void block_scores_kernel(Grid grid, Grid block, void *stream, const float *__restrict__ pooled,
                         const float *__restrict__ dead, const float *__restrict__ q_idx,
                         const int32_t *__restrict__ steps, int64_t max_blocks, float *__restrict__ out) {
    strata::sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(grid.size() * block.size(), block.size()),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           block_scores_kernel_impl(it, grid, block, pooled, dead, q_idx, steps, max_blocks,
                                                    out);
                       });
    });
}

void block_topk_kernel_impl(sycl::nd_item<1> it, Grid grid, Grid block, const float *__restrict__ scores,
                            const int32_t *__restrict__ steps, int64_t max_blocks, int64_t cap,
                            int32_t *__restrict__ ids, int *ptr_hist, int *ptr_s_a, int *ptr_s_b,
                            int &s_digit, int &s_above) {
    int *hist = ptr_hist;
    int *s_a = ptr_s_a;
    int *s_b = ptr_s_b;

    const int64_t qi = group_x(it, grid);
    const int32_t *st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t *out = ids + qi * cap;
    const int t = local_x(it, block);
    if (n_kv <= width) { // everything is selected: the identity, ascending
        for (int64_t j = t; j < n_kv; j += TOPK_T)
            out[j] = (int32_t)j;
        return;
    }
    const float *sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1; // blocks 0..n_bid, the last possibly empty
    const int64_t per = (nb + TOPK_T - 1) / TOPK_T;
    const int64_t b0 = (int64_t)t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int)(n_kv - n_bid * R); };
    // ---- radix select: the largest key thr with (cells with key >= thr) >= width, 8 bits at a time
    uint32_t prefix = 0;
    int above = 0; // cells strictly above the digits fixed so far
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = t; i < 256; i += TOPK_T)
            hist[i] = 0;
        it.barrier(sycl::access::fence_space::local_space);
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
        for (int64_t b = b0; b < b1; ++b) {
            const int w = weight(b);
            if (w == 0)
                continue;
            const uint32_t k = order_key(sc[b]);
            if ((k & hi_mask) == (prefix & hi_mask))
                atomic_add(&hist[(k >> shift) & 255], w);
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[d] >= width)
                    break;
                cum += hist[d];
            }
            s_digit = d;
            s_above = cum;
        }
        it.barrier(sycl::access::fence_space::local_space);
        prefix |= (uint32_t)s_digit << shift;
        above = s_above;
        it.barrier(sycl::access::fence_space::local_space);
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above; // cells equal to thr that fit, lowest index first
    // ---- per-thread counts of cells above and at the threshold, then their exclusive prefixes
    int gt = 0, eq = 0;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0)
            continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr)
            gt += w;
        else if (k == thr)
            eq += w;
    }
    s_a[t] = gt;
    s_b[t] = eq;
    it.barrier(sycl::access::fence_space::local_space);
    if (t == 0) {
        int ag = 0, ae = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int g = s_a[i], e = s_b[i];
            s_a[i] = ag;
            s_b[i] = ae;
            ag += g;
            ae += e;
        }
    }
    it.barrier(sycl::access::fence_space::local_space);
    const int64_t eq_before = s_b[t];
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0)
        my_eq = 0;
    if (my_eq > eq)
        my_eq = eq;
    const int sel = gt + (int)my_eq;
    it.barrier(sycl::access::fence_space::local_space);
    s_a[t] = sel;
    it.barrier(sycl::access::fence_space::local_space);
    if (t == 0) {
        int a = 0;
        for (int i = 0; i < TOPK_T; ++i) {
            const int c = s_a[i];
            s_a[i] = a;
            a += c;
        }
    }
    it.barrier(sycl::access::fence_space::local_space);
    int64_t wpos = s_a[t];
    int64_t eq_left = my_eq;
    for (int64_t b = b0; b < b1; ++b) {
        const int w = weight(b);
        if (w == 0)
            continue;
        const uint32_t k = order_key(sc[b]);
        if (k > thr) {
            for (int c = 0; c < w; ++c)
                out[wpos++] = (int32_t)(b * R + c);
        } else if (k == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left)
                out[wpos++] = (int32_t)(b * R + c);
        }
    }
}

void block_topk_kernel(Grid grid, Grid block, void *stream, const float *__restrict__ scores,
                       const int32_t *__restrict__ steps, int64_t max_blocks, int64_t cap,
                       int32_t *__restrict__ ids) {
    strata::sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler &h) {
        sycl::local_accessor<int, 1> l_hist(sycl::range<1>((256)), h);
        sycl::local_accessor<int, 1> l_s_a(sycl::range<1>((TOPK_T)), h);
        sycl::local_accessor<int, 1> l_s_b(sycl::range<1>((TOPK_T)), h);
        sycl::local_accessor<int, 1> l_s_digit(sycl::range<1>(1), h);
        sycl::local_accessor<int, 1> l_s_above(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>(grid.size() * block.size(), block.size()),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           block_topk_kernel_impl(
                               it, grid, block, scores, steps, max_blocks, cap, ids,
                               l_hist.template get_multi_ptr<sycl::access::decorated::no>().get(),
                               l_s_a.template get_multi_ptr<sycl::access::decorated::no>().get(),
                               l_s_b.template get_multi_ptr<sycl::access::decorated::no>().get(),
                               l_s_digit[0], l_s_above[0]);
                       });
    });
}

// ---- the same top-k with each query's keys read once: 1,024 threads hold up to TK_PER consecutive blocks'
// keys in registers (contexts up to 4 * 1024 * TK_PER cells), per-warp histograms, block-wide scans. The
// selection rule is block_topk_kernel's (radix threshold, ties to the lowest index, cells ascending):
// identical ids.
constexpr int TK_T = 1024;
constexpr int TK_PER = 33;

inline int block_excl_scan(sycl::nd_item<1> it, int v, int *s_warp, int &total) {
    const int lane = int(it.get_local_id(0)) & 31, warp = int(it.get_local_id(0)) >> 5;
    int x = v;
#pragma unroll
    for (int o = 1; o < 32; o <<= 1) {
        const int y = up_lane(it, x, o);
        if (lane >= o)
            x += y;
    }
    if (lane == 31)
        s_warp[warp] = x;
    it.barrier(sycl::access::fence_space::local_space);
    if (warp == 0) {
        int w = s_warp[lane];
        int z = w;
#pragma unroll
        for (int o = 1; o < 32; o <<= 1) {
            const int y = up_lane(it, z, o);
            if (lane >= o)
                z += y;
        }
        s_warp[lane] = z - w; // exclusive per warp
        if (lane == 31)
            s_warp[32] = z; // total
    }
    it.barrier(sycl::access::fence_space::local_space);
    const int r = s_warp[warp] + x - v;
    total = s_warp[32];
    it.barrier(sycl::access::fence_space::local_space);
    return r;
}

template <int PER>
void block_topk_reg_kernel_impl(sycl::nd_item<1> it, Grid grid, Grid block, const float *__restrict__ scores,
                                const int32_t *__restrict__ steps, int64_t max_blocks, int64_t cap,
                                int32_t *__restrict__ ids, int *ptr_hist, int *ptr_s_warp, int &s_digit,
                                int &s_above) {
    auto hist = reinterpret_cast<int (*)[256]>(ptr_hist);
    int *s_warp = ptr_s_warp;

    const int64_t qi = group_x(it, grid);
    const int32_t *st = steps + qi * kStepCount;
    const int64_t n_kv = st[kStepNKv], n_bid = st[kStepNBid], width = st[kStepWidth];
    int32_t *out = ids + qi * cap;
    const int t = local_x(it, block), lane = t & 31, warp = t >> 5;
    if (n_kv <= width) {
        for (int64_t j = t; j < n_kv; j += TK_T)
            out[j] = (int32_t)j;
        return;
    }
    const float *sc = scores + qi * max_blocks;
    const int64_t nb = n_bid + 1;
    const int64_t per = (nb + TK_T - 1) / TK_T; // <= PER (the caller checks)
    const int64_t b0 = (int64_t)t * per, b1 = (b0 + per < nb) ? b0 + per : nb;
    uint32_t key[PER];
#pragma unroll
    for (int j = 0; j < PER; ++j)
        key[j] = (b0 + j < b1) ? order_key(sc[b0 + j]) : 0u;
    auto weight = [&](int64_t b) -> int { return b < n_bid ? R : (int)(n_kv - n_bid * R); };
    uint32_t prefix = 0;
    int above = 0;
    for (int shift = 24; shift >= 0; shift -= 8) {
        for (int i = lane; i < 256; i += 32)
            hist[warp][i] = 0;
        sycl::group_barrier(it.get_sub_group());
        const uint32_t hi_mask = shift == 24 ? 0u : (0xffffffffu << (shift + 8));
#pragma unroll
        for (int j = 0; j < PER; ++j) {
            const int64_t b = b0 + j;
            if (b >= b1)
                break;
            const int w = weight(b);
            if (w == 0)
                continue;
            if ((key[j] & hi_mask) == (prefix & hi_mask))
                atomic_add(&hist[warp][(key[j] >> shift) & 255], w);
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (t < 256) { // fold the warps' histograms into warp 0's
            int s = 0;
            for (int w2 = 0; w2 < TK_T / 32; ++w2)
                s += hist[w2][t];
            hist[0][t] = s;
        }
        it.barrier(sycl::access::fence_space::local_space);
        if (t == 0) {
            int cum = above, d = 255;
            for (; d > 0; --d) {
                if (cum + hist[0][d] >= width)
                    break;
                cum += hist[0][d];
            }
            s_digit = d;
            s_above = cum;
        }
        it.barrier(sycl::access::fence_space::local_space);
        prefix |= (uint32_t)s_digit << shift;
        above = s_above;
        it.barrier(sycl::access::fence_space::local_space);
    }
    const uint32_t thr = prefix;
    const int64_t eq_budget = width - above;
    int gt = 0, eq = 0;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1)
            break;
        const int w = weight(b);
        if (w == 0)
            continue;
        if (key[j] > thr)
            gt += w;
        else if (key[j] == thr)
            eq += w;
    }
    int tot;
    const int eq_before = block_excl_scan(it, eq, s_warp, tot);
    int64_t my_eq = eq_budget - eq_before;
    if (my_eq < 0)
        my_eq = 0;
    if (my_eq > eq)
        my_eq = eq;
    const int sel = gt + (int)my_eq;
    int64_t wpos = block_excl_scan(it, sel, s_warp, tot);
    int64_t eq_left = my_eq;
#pragma unroll
    for (int j = 0; j < PER; ++j) {
        const int64_t b = b0 + j;
        if (b >= b1)
            break;
        const int w = weight(b);
        if (w == 0)
            continue;
        if (key[j] > thr) {
            for (int c = 0; c < w; ++c)
                out[wpos++] = (int32_t)(b * R + c);
        } else if (key[j] == thr) {
            for (int c = 0; c < w && eq_left > 0; ++c, --eq_left)
                out[wpos++] = (int32_t)(b * R + c);
        }
    }
}

template <int PER>
void block_topk_reg_kernel(Grid grid, Grid block, void *stream, const float *__restrict__ scores,
                           const int32_t *__restrict__ steps, int64_t max_blocks, int64_t cap,
                           int32_t *__restrict__ ids) {
    strata::sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler &h) {
        sycl::local_accessor<int, 1> l_hist(sycl::range<1>((TK_T / 32) * (256)), h);
        sycl::local_accessor<int, 1> l_s_warp(sycl::range<1>((33)), h);
        sycl::local_accessor<int, 1> l_s_digit(sycl::range<1>(1), h);
        sycl::local_accessor<int, 1> l_s_above(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>(grid.size() * block.size(), block.size()),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           block_topk_reg_kernel_impl<PER>(
                               it, grid, block, scores, steps, max_blocks, cap, ids,
                               l_hist.template get_multi_ptr<sycl::access::decorated::no>().get(),
                               l_s_warp.template get_multi_ptr<sycl::access::decorated::no>().get(),
                               l_s_digit[0], l_s_above[0]);
                       });
    });
}

// Block scores with every key block read ONCE for all of a call's queries (block_scores_kernel's grid is
// (max_blocks / 8) x nq: ~24,600 mostly-idle blocks per layer at a decode window, each key re-read per
// query).  A fixed grid strides over the blocks; per (block, query) the same arithmetic in the same order as
// block_scores_kernel. qsa_block_scores takes it for every call without an active-block count and at most MQ
// queries: the captured decode window, the uncaptured decode, and prefill's pooled16 call.
constexpr int MQ = 8;
void block_scores_multi_kernel_impl(sycl::nd_item<1> it, Grid grid, Grid block,
                                    const float *__restrict__ pooled, const float *__restrict__ dead,
                                    const float *__restrict__ q_idx, const int32_t *__restrict__ steps,
                                    int nq, int64_t max_blocks, float *__restrict__ out, float *ptr_qs,
                                    int64_t *ptr_s_nkv, int64_t *ptr_s_nbid) {
    float *qs = ptr_qs;
    int64_t *s_nkv = ptr_s_nkv;
    int64_t *s_nbid = ptr_s_nbid;

    for (int i = local_x(it, block); i < nq * IDX_HEADS * IDX_DIM; i += block.x)
        qs[i] = q_idx[i];
    if (local_x(it, block) < nq) {
        s_nkv[local_x(it, block)] = steps[local_x(it, block) * kStepCount + kStepNKv];
        s_nbid[local_x(it, block)] = steps[local_x(it, block) * kStepCount + kStepNBid];
    }
    it.barrier(sycl::access::fence_space::local_space);
    int64_t top = 0;
    for (int q = 0; q < nq; ++q)
        top = s_nbid[q] > top ? s_nbid[q] : top;
    const int lane = local_x(it, block) & 31;
    const int64_t wstride = (int64_t)grid.x * SCORE_WARPS;
    for (int64_t b = (int64_t)group_x(it, grid) * SCORE_WARPS + (local_x(it, block) >> 5);
         b <= top && b < max_blocks; b += wstride) {
        const float4 kp = *reinterpret_cast<const float4 *>(pooled + b * IDX_DIM + lane * 4);
        const float4 kd = *reinterpret_cast<const float4 *>(dead + lane * 4);
        for (int qi = 0; qi < nq; ++qi) {
            const int64_t n_bid = s_nbid[qi];
            if (b > n_bid)
                continue;
            const float4 k4 = (b == n_bid) ? kd : kp;
            const float *q = qs + qi * IDX_HEADS * IDX_DIM + lane * 4;
            float score = 0.0f;
#pragma unroll
            for (int h = 0; h < IDX_HEADS; ++h) {
                const float4 q4 = *reinterpret_cast<const float4 *>(q + h * IDX_DIM);
                float d = k4.x * q4.x + k4.y * q4.y + k4.z * q4.z + k4.w * q4.w;
#pragma unroll
                for (int o = 16; o > 0; o >>= 1)
                    d += xor_lane(it, d, o);
                score += d > 0.0f ? d : 0.0f;
            }
            if (lane == 0) {
                if (b == n_bid && s_nkv[qi] % R != 0)
                    score += 1e9f;
                out[qi * max_blocks + b] = score;
            }
        }
    }
}

void block_scores_multi_kernel(Grid grid, Grid block, void *stream, const float *__restrict__ pooled,
                               const float *__restrict__ dead, const float *__restrict__ q_idx,
                               const int32_t *__restrict__ steps, int nq, int64_t max_blocks,
                               float *__restrict__ out) {
    strata::sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> l_qs(sycl::range<1>((MQ * IDX_HEADS * IDX_DIM)), h);
        sycl::local_accessor<int64_t, 1> l_s_nkv(sycl::range<1>((MQ)), h);
        sycl::local_accessor<int64_t, 1> l_s_nbid(sycl::range<1>((MQ)), h);
        h.parallel_for(sycl::nd_range<1>(grid.size() * block.size(), block.size()),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           block_scores_multi_kernel_impl(
                               it, grid, block, pooled, dead, q_idx, steps, nq, max_blocks, out,
                               l_qs.template get_multi_ptr<sycl::access::decorated::no>().get(),
                               l_s_nkv.template get_multi_ptr<sycl::access::decorated::no>().get(),
                               l_s_nbid.template get_multi_ptr<sycl::access::decorated::no>().get());
                       });
    });
}

} // namespace

void qsa_block_scores(const float *pooled, const float *dead, const float *q_idx, const int32_t *steps,
                      int64_t nq, int64_t max_blocks, const QsaShapes &s, float *scores, void *stream,
                      int64_t active_blocks) {
    if (nq <= 0)
        return;
    if (s.idx_dim != IDX_DIM || s.idx_n_head != IDX_HEADS || s.idx_block != R || nq > 65535) {
        std::fprintf(stderr, "qsa_block_scores: unsupported indexer geometry\n");
        std::exit(1);
    }
    // a block past a query's n_bid returns at once: the grid need only reach the batch's largest n_bid (C-1)
    static const bool multi = [] {
        const char *v = std::getenv("STRATA_SCORES_MULTI");
        return v == nullptr || std::atoi(v) != 0;
    }();
    if (multi && nq <= MQ &&
        active_blocks <= 0) { // no active count: decode (captured or not) and prefill's pooled16
        block_scores_multi_kernel(256, SCORE_WARPS * 32, (cudaStream_t)stream, pooled, dead, q_idx, steps,
                                  (int)nq, max_blocks, scores);
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "qsa_block_scores multi: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
        return;
    }
    const int64_t reach = active_blocks > 0 && active_blocks < max_blocks ? active_blocks : max_blocks;
    const Grid grid((unsigned)((reach + SCORE_WARPS - 1) / SCORE_WARPS), (unsigned)nq);
    block_scores_kernel(grid, SCORE_WARPS * 32, (cudaStream_t)stream, pooled, dead, q_idx, steps, max_blocks,
                        scores);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_block_scores: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

void qsa_block_topk_ref(const float *scores, const int32_t *steps, int64_t nq, int64_t max_blocks,
                        int64_t cap, const QsaShapes &s, int32_t *ids, void *stream) {
    if (nq <= 0)
        return;
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    block_topk_kernel((unsigned)nq, TOPK_T, (cudaStream_t)stream, scores, steps, max_blocks, cap, ids);
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}

// CUDA TF32 / AMD WMMA and CUDA thread-block clusters have no implementation here yet.
// These capability probes launch nothing and retain the public fallback contract.
bool qsa_block_scores_tc(const float *, const float *, const float *, const int32_t *, int64_t nq, int64_t,
                         const QsaShapes &, float *, void *, int64_t) {
    return nq <= 0;
}
bool qsa_block_topk_cluster(const float *, const int32_t *, int64_t, int64_t, int64_t, const QsaShapes &,
                            int32_t *, void *) {
    return false;
}
void qsa_block_topk(const float *scores, const int32_t *steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes &s, int32_t *ids, void *stream, int64_t active_blocks) {
    if (nq <= 0)
        return;
    static const bool old = std::getenv("STRATA_TOPK_OLD") != nullptr;
    static const bool capacity_guard = std::getenv("STRATA_TOPK_CAPACITY_GUARD") != nullptr;
    // A valid live bound changes the kernel selection, never the score-row stride.
    // Captured graphs omit the bound so growing context stays within the selected kernel's capacity.
    const int64_t reach =
        !capacity_guard && active_blocks > 0 && active_blocks <= max_blocks ? active_blocks : max_blocks;
    if (old || reach > int64_t(TK_T) * TK_PER) {
        qsa_block_topk_ref(scores, steps, nq, max_blocks, cap, s, ids, stream);
        return;
    }
    if (s.idx_block != R || cap < qsa_selection_width(kTopkMaxCells, s)) {
        std::fprintf(stderr, "qsa_block_topk: unsupported geometry or cap\n");
        std::exit(1);
    }
    block_topk_reg_kernel<TK_PER>((unsigned)nq, TK_T, (cudaStream_t)stream, scores, steps, max_blocks, cap,
                                  ids);
    auto e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "qsa_block_topk: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
}
} // namespace strata::kernels
