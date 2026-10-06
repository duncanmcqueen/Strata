// src/prefill/sycl/kernels.cpp - SYCL port of src/prefill/kernels.cu; see include/strata/prefill/kernels.hpp.
//
// The per-element arithmetic is the CUDA file's, in the same order, so the variants it keeps bitwise equal (the
// tiled conv vs the serial walk, the column-split recurrences vs the per-head one, gr_mix_r vs gr_norm + gr_mix,
// gr_write_norm_rs vs gr_write + gr_norm_rs) stay equal on this backend: each pair shares one helper.
//
// What SYCL forces to change:
//   * CUDA grids flatten to 1D work-groups (x fastest); shared memory is local accessors; warps are 32-wide
//     sub-groups; `__expf`/`expf`/`log1pf`/`powf` are `sycl::exp`/`log1p`/`pow`, `rsqrtf` `sycl::rsqrt`.
//   * fp16 stores use f16_bits.hpp's round-to-nearest-even (as __float2half_rn).
//   * gdn_rec_kh_kernel (the cp.async key-head variant, taken on sm_80+ CUDA cards that hold its 64 blocks at once)
//     is not ported: the default here is the software-pipelined column kernel, which the CUDA file keeps bitwise
//     equal to it.  Matching it to Intel's memory system is tuning work, measured when the engine runs.
#include "strata/prefill/kernels.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/kernels/router_top10.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "../../kernels/sycl/fp_exact.hpp"

#include <cuda_runtime.h>
#include <sycl/ext/intel/math.hpp>

#include <cfloat>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::prefill {
namespace {

namespace im = strata::kernels::fp_exact;
using strata::kernels::f16_from_f32;
using strata::kernels::f32_from_f16;

constexpr int N = 2560, HC = 4, D = N * HC, LR = 320;
constexpr int S = 128, HK = 16, HV = 48, C = 10240;

inline float warp_sum(const sycl::sub_group& sg, float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
    return v;
}
inline float warp_max(const sycl::sub_group& sg, float v) {
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) v = sycl::fmax(v, sycl::permute_group_by_xor(sg, v, o));
    return v;
}
inline uint16_t bf(float f) {
    uint32_t u = sycl::bit_cast<uint32_t>(f);
    u += 0x7fffu + ((u >> 16) & 1u);
    return (uint16_t) (u >> 16);
}
// what the BF16 image `hi` left out of f, itself in BF16 (STRATA_PREFILL_BF16X2)
inline uint16_t bf_lo(float f, uint16_t hi) { return bf(f - sycl::bit_cast<float>((uint32_t) hi << 16)); }
inline float sigm(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
inline uint16_t hf(float f) { return f16_from_f32(f); }
// a SwiGLU product for an FP16 GEMM, saturated (see the CUDA file); a NaN stays NaN
inline uint16_t hf_sat(float f) { return hf(sycl::isnan(f) ? f : sycl::fmin(sycl::fmax(f, -65504.0f), 65504.0f)); }

// work-group sum (wg <= 1024), broadcast; sh holds 32 floats (the CUDA block_sum)
inline float block_sum(sycl::nd_item<1> it, float v, float* sh) {
    const int tid = (int) it.get_local_id(0), lane = tid & 31, w = tid >> 5;
    const sycl::sub_group sg = it.get_sub_group();
    v = warp_sum(sg, v);
    sycl::group_barrier(it.get_group());
    if (lane == 0) sh[w] = v;
    sycl::group_barrier(it.get_group());
    const int nw = ((int) it.get_local_range(0) + 31) >> 5;
    float t = tid < nw ? sh[tid] : 0.0f;
    if (w == 0) t = warp_sum(sg, t);
    if (tid == 0) sh[0] = t;
    sycl::group_barrier(it.get_group());
    return sh[0];
}

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "prefill %s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
sycl::queue& Q(void* stream) { return sycl_runtime::queue_from_stream(stream); }

// n work-items, body(i)
template<class F> void items(void* stream, int64_t n, F body) {
    if (n <= 0) return;
    Q(stream).parallel_for(sycl::range<1>((size_t) n), [=](sycl::id<1> id) { body((int64_t) id[0]); });
}
// `groups` work-groups of `wg`, with `local` floats of local memory; body(it, local)
template<class F> void groups(void* stream, int64_t ngroups, int wg, int local, F body) {
    if (ngroups <= 0) return;
    Q(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> lm(sycl::range<1>((size_t) (local > 0 ? local : 1)), h);
        h.parallel_for(sycl::nd_range<1>((size_t) ngroups * wg, (size_t) wg),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] { body(it, &lm[0]); });
    });
}

// ---------------------------------------------------------------- hyper-connection
// the normalized value of row element d (gr_norm's, and gr_mix_r recomputes it bit for bit)
inline float gr_value(float r, float rs, float w) { return r * rs * w; }

void gr_norm_launch(const float* R, const float* w, float eps, float* xn, float* rs_out, uint16_t* xn16,
                    uint16_t* xn16_lo, int64_t T, void* stream) {
    groups(stream, T * HC, 256, 32, [=](sycl::nd_item<1> it, float* sh) {
        const int64_t row = (int64_t) it.get_group(0);   // t * 4 + c
        const int c = (int) (row % HC), tid = (int) it.get_local_id(0);
        const float* r = R + row * N;
        float ss = 0.0f;
        for (int d = tid; d < N; d += 256) ss += r[d] * r[d];
        const float rs = sycl::rsqrt(block_sum(it, ss, sh) / (float) N + eps);
        if (rs_out && tid == 0) rs_out[row] = rs;
        for (int d = tid; d < N; d += 256) {
            const float v = gr_value(r[d], rs, w[c * N + d]);
            if (xn) xn[row * N + d] = v;
            const uint16_t h = bf(v);
            xn16[row * N + d] = h;
            if (xn16_lo) xn16_lo[row * N + d] = bf_lo(v, h);
        }
    });
}

// the mix of one (t, d): sum_c x_c * sigm(g_c), / HC, then its images
inline void gr_mix_store(float s, int64_t i, float* mixed, uint16_t* mixed16, uint16_t* mixed_h, uint16_t* mixed16_lo) {
    s /= (float) HC;
    mixed[i] = s;
    if (mixed16) {
        const uint16_t h = bf(s);
        mixed16[i] = h;
        if (mixed16_lo) mixed16_lo[i] = bf_lo(s, h);
    }
    if (mixed_h) mixed_h[i] = hf(s);
}

// R += bo * 2 sigm(inj / HC), one element
inline float gr_write_value(float r, float bo, float inj) { return sycl::fma(bo, 2.0f * sigm(inj / (float) HC), r); }

// ---------------------------------------------------------------- GDN
inline float gdn_conv_out(float v0, float v1, float v2, float x, float w0, float w1, float w2, float w3) {
    const float s = v0 * w0 + v1 * w1 + v2 * w2 + x * w3;
    return s / (1.0f + sycl::exp(-s));
}
constexpr int CONV_TILE = 64;
constexpr int RG = 4, RPG = S / RG;
constexpr int CB = 32, NCB = S / CB;

// the output norm's value for column col of a head (shared by the per-head kernel and the norm kernel)
inline float gdn_norm_value(float oc, float ss, float eps, float g_col, float z) {
    return oc * sycl::rsqrt(ss / (float) S + eps) * g_col * sigm(z);
}

// the column-split recurrence; PIPE prefetches the next token's inputs into registers (the same arithmetic)
template<bool PIPE>
void gdn_rec_cols(float* state, const float* h, const float* gate, const float* beta, float* oc_out, int64_t T,
                  void* stream) {
    constexpr int NT = CB * RG, LPT = S / NT;
    // local: sk[S], sq[S], red[RG][CB]
    groups(stream, HV * NCB, NT, 2 * S + RG * CB, [=](sycl::nd_item<1> it, float* lm) {
        float* sk = lm;
        float* sq = lm + S;
        float* red = lm + 2 * S;
        const int head = (int) it.get_group(0) / NCB, cb = (int) it.get_group(0) % NCB;
        const int tid = (int) it.get_local_id(0), c = tid % CB, rg = tid / CB, col = cb * CB + c;
        const int qh = head % HK;
        float s[RPG];
        float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
        const size_t rs = (size_t) HV * S;
#pragma unroll
        for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
        float nq[LPT], nk[LPT], nv = 0.0f, ng = 0.0f, nb = 0.0f;
        auto fetch = [&](int64_t t) {
            const float* ht = h + t * C;
#pragma unroll
            for (int u = 0; u < LPT; ++u) { nq[u] = ht[qh * S + tid + u * NT]; nk[u] = ht[HK * S + qh * S + tid + u * NT]; }
            nv = ht[2 * HK * S + head * S + col];
            ng = gate[t * HV + head];
            nb = beta[t * HV + head];
        };
        if (PIPE && T > 0) fetch(0);
        for (int64_t t = 0; t < T; ++t) {
            if (!PIPE) fetch(t);
            float cq[LPT], ck[LPT];
#pragma unroll
            for (int u = 0; u < LPT; ++u) { cq[u] = nq[u]; ck[u] = nk[u]; }
            const float cv = nv, cg = ng, cbt = nb;
            sycl::group_barrier(it.get_group());
#pragma unroll
            for (int u = 0; u < LPT; ++u) { sq[tid + u * NT] = cq[u]; sk[tid + u * NT] = ck[u]; }
            sycl::group_barrier(it.get_group());
            if (PIPE && t + 1 < T) fetch(t + 1);
            const float g = sycl::exp(cg);
            float kv = 0.0f;
#pragma unroll
            for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
            red[rg * CB + c] = kv;
            sycl::group_barrier(it.get_group());
            const float kv_col = red[c] + red[CB + c] + red[2 * CB + c] + red[3 * CB + c];
            const float delta = (cv - g * kv_col) * cbt;
            float o = 0.0f;
#pragma unroll
            for (int r = 0; r < RPG; ++r) {
                s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                o = sycl::fma(s[r], sq[rg * RPG + r], o);
            }
            sycl::group_barrier(it.get_group());
            red[rg * CB + c] = o;
            sycl::group_barrier(it.get_group());
            if (rg == 0)
                oc_out[t * HV * S + head * S + col] = (red[c] + red[CB + c] + red[2 * CB + c] + red[3 * CB + c]) * sycl::rsqrt((float) S);
        }
#pragma unroll
        for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
    });
}

}  // namespace

void kv_append(const float* K, const float* V, int64_t T, int64_t pos0, const int32_t* page_table, int64_t page_size,
               uint16_t* k_pool, uint16_t* v_pool, int8_t* k_q, int8_t* v_q, uint16_t* k_scale, uint16_t* v_scale,
               void* stream, const strata::kernels::KvHostPools* host_p, const strata::kernels::KvHostPools* stage_p) {
    if (T <= 0) return;
    const strata::kernels::KvHostPools host = host_p ? *host_p : strata::kernels::KvHostPools{};
    const strata::kernels::KvHostPools stage = stage_p ? *stage_p : strata::kernels::KvHostPools{};
    // one work-group per (token, kv head, 64-value group, K/V): CUDA's (T, 2, 8) grid, x fastest
    groups(stream, T * 2 * 8, 64, 2, [=](sycl::nd_item<1> it, float* wm) {
        const int64_t gid = (int64_t) it.get_group(0);
        const int64_t t = gid % T;
        const int kvh = (int) ((gid / T) % 2), z = (int) (gid / (T * 2));
        const int g = z >> 1;
        const bool is_v = (z & 1) != 0;
        const int tid = (int) it.get_local_id(0);
        const int d = g * 64 + tid;
        const float x = (is_v ? V : K)[t * 512 + kvh * 256 + d];
        const int64_t pos = pos0 + t;
        const int64_t page = page_table[pos / page_size];
        const int64_t row = (page * 2 + kvh) * page_size + pos % page_size;
        const int64_t row_id = ((pos / page_size) * 2 + kvh) * page_size + pos % page_size;
        if (k_pool != nullptr) {   // uniform over the work-group
            const uint16_t h = hf(x);
            if (page >= 0) (is_v ? v_pool : k_pool)[row * 256 + d] = h;
            if (host.k_pool != nullptr) (is_v ? host.v_pool : host.k_pool)[row_id * 256 + d] = h;
            if (stage.k_pool != nullptr) (is_v ? stage.v_pool : stage.k_pool)[row_id * 256 + d] = h;
            return;
        }
        float a = sycl::fabs(x);
        const sycl::sub_group sg = it.get_sub_group();
        for (int o = 16; o > 0; o >>= 1) a = sycl::fmax(a, sycl::permute_group_by_xor(sg, a, o));
        if ((tid & 31) == 0) wm[tid >> 5] = a;
        sycl::group_barrier(it.get_group());
        const float amax = sycl::fmax(wm[0], wm[1]);
        const uint16_t sb = hf(im::fdiv_rn(amax, 127.0f));
        const float sf = f32_from_f16(sb);
        int q = 0;
        if (sf > 0.0f) {
            q = (int) sycl::rint(im::fdiv_rn(x, sf));
            q = q < -127 ? -127 : (q > 127 ? 127 : q);
        }
        if (page >= 0) {
            (is_v ? v_q : k_q)[row * 256 + d] = (int8_t) q;
            if (tid == 0) (is_v ? v_scale : k_scale)[row * 4 + g] = sb;
        }
        if (host.k_q != nullptr) {
            (is_v ? host.v_q : host.k_q)[row_id * 256 + d] = (int8_t) q;
            if (tid == 0) (is_v ? host.v_scale : host.k_scale)[row_id * 4 + g] = sb;
        }
        if (stage.k_q != nullptr) {
            (is_v ? stage.v_q : stage.k_q)[row_id * 256 + d] = (int8_t) q;
            if (tid == 0) (is_v ? stage.v_scale : stage.k_scale)[row_id * 4 + g] = sb;
        }
    });
    check("kv_append");
}

void to_f16(const float* x, uint16_t* y, int64_t n, void* stream) {
    items(stream, n, [=](int64_t i) { y[i] = hf(x[i]); });
    check("to_f16");
}
void round_f16(const float* x, float* y, int64_t n, void* stream) {
    items(stream, n, [=](int64_t i) { y[i] = f32_from_f16(hf(x[i])); });
    check("round_f16");
}
void to_bf16(const float* x, uint16_t* y, int64_t n, void* stream, uint16_t* ylo) {
    items(stream, n, [=](int64_t i) {
        const uint16_t h = bf(x[i]);
        y[i] = h;
        if (ylo) ylo[i] = bf_lo(x[i], h);
    });
    check("to_bf16");
}

void gr_norm(const float* R, const float* w_norm, float eps, float* xn, uint16_t* xn16, int64_t T, void* stream,
             uint16_t* xn16_lo) {
    gr_norm_launch(R, w_norm, eps, xn, nullptr, xn16, xn16_lo, T, stream);
    check("gr_norm");
}
void gr_norm_rs(const float* R, const float* w_norm, float eps, float* rs, uint16_t* xn16, int64_t T, void* stream,
                uint16_t* xn16_lo) {
    gr_norm_launch(R, w_norm, eps, nullptr, rs, xn16, xn16_lo, T, stream);
    check("gr_norm_rs");
}
void gr_mix_r(const float* R, const float* rs, const float* w_norm, const float* gated, float* mixed, uint16_t* mixed16,
              int64_t T, void* stream, uint16_t* mixed_h, uint16_t* mixed16_lo) {
    items(stream, T * N, [=](int64_t i) {
        const int64_t t = i / N, d = i % N;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            const int64_t j = t * D + c * N + d;
            s = sycl::fma(gr_value(R[j], rs[t * HC + c], w_norm[c * N + d]), sigm(gated[j]), s);   // gr_norm's value, bit for bit
        }
        gr_mix_store(s, i, mixed, mixed16, mixed_h, mixed16_lo);
    });
    check("gr_mix_r");
}
void gr_write_norm_rs(float* R, const float* bo, const float* inj, int64_t inj_ld, const float* w_norm_next, float eps,
                      float* rs, uint16_t* xn16, int64_t T, void* stream, uint16_t* xn16_lo) {
    constexpr int PER = (N + 255) / 256;
    groups(stream, T * HC, 256, 32, [=](sycl::nd_item<1> it, float* sh) {
        const int64_t row = (int64_t) it.get_group(0);
        const int64_t t = row / HC;
        const int c = (int) (row % HC), tid = (int) it.get_local_id(0);
        float* r = R + row * N;
        float v[PER];
        float ss = 0.0f;
        int k = 0;
#pragma unroll
        for (int d = tid; d < N; d += 256, ++k) {
            const float x = gr_write_value(r[d], bo[t * N + d], inj[t * inj_ld + c]);
            r[d] = x;
            v[k] = x;
            ss += x * x;
        }
        const float rsv = sycl::rsqrt(block_sum(it, ss, sh) / (float) N + eps);
        if (tid == 0) rs[row] = rsv;
        k = 0;
#pragma unroll
        for (int d = tid; d < N; d += 256, ++k) {
            const float x = gr_value(v[k], rsv, w_norm_next[c * N + d]);
            const uint16_t h = bf(x);
            xn16[row * N + d] = h;
            if (xn16_lo) xn16_lo[row * N + d] = bf_lo(x, h);
        }
    });
    check("gr_write_norm_rs");
}
void gr_silu(const float* lo, uint16_t* lo16, int64_t T, void* stream, uint16_t* lo16_lo) {
    items(stream, T * LR, [=](int64_t i) {
        const float x = lo[i] / (float) HC;
        const float v = x / (1.0f + sycl::exp(-x));
        const uint16_t h = bf(v);
        lo16[i] = h;
        if (lo16_lo) lo16_lo[i] = bf_lo(v, h);
    });
    check("gr_silu");
}
void gr_mix(const float* xn, const float* gated, float* mixed, uint16_t* mixed16, int64_t T, void* stream,
            uint16_t* mixed_h, uint16_t* mixed16_lo) {
    items(stream, T * N, [=](int64_t i) {
        const int64_t t = i / N, d = i % N;
        float s = 0.0f;
#pragma unroll
        for (int c = 0; c < HC; ++c) {
            const int64_t j = t * D + c * N + d;
            s = sycl::fma(xn[j], sigm(gated[j]), s);
        }
        gr_mix_store(s, i, mixed, mixed16, mixed_h, mixed16_lo);
    });
    check("gr_mix");
}
void gr_write(float* R, const float* bo, const float* inj, int64_t inj_ld, int64_t T, void* stream) {
    items(stream, T * D, [=](int64_t i) {
        const int64_t t = i / D, c = (i % D) / N, d = i % N;
        R[i] = gr_write_value(R[i], bo[t * N + d], inj[t * inj_ld + c]);
    });
    check("gr_write");
}
void gr_broadcast(const float* e, float* R, int64_t T, void* stream) {
    items(stream, T * D, [=](int64_t i) { R[i] = e[(i / D) * N + i % N]; });
    check("gr_broadcast");
}

void gdn_gates(const float* ab, const float* dt, const float* ssm_a, float* gate, float* beta, int64_t T, void* stream) {
    items(stream, T * HV, [=](int64_t i) {
        const int64_t t = i / HV, h = i % HV;
        const float v = ab[t * 2 * HV + h] + dt[h];
        gate[i] = (v > 20.0f ? v : sycl::log1p(sycl::exp(v))) * ssm_a[h];
        beta[i] = sigm(ab[t * 2 * HV + HV + h]);
    });
    check("gdn_gates");
}

void gdn_conv(float* history, const float* qkv, const float* conv_w, float* h, int64_t T, float eps, void* stream) {
    static const bool serial = std::getenv("STRATA_GDN_CONV_SERIAL") != nullptr;   // the old walk (A/B)
    if (serial || T <= CONV_TILE) {
        items(stream, C, [=](int64_t cc) {
            const int c = (int) cc;
            float v0 = history[c * 3], v1 = history[c * 3 + 1], v2 = history[c * 3 + 2];
            const float w0 = conv_w[c * 4], w1 = conv_w[c * 4 + 1], w2 = conv_w[c * 4 + 2], w3 = conv_w[c * 4 + 3];
            for (int64_t t = 0; t < T; ++t) {
                const float x = qkv[t * C + c];
                h[t * C + c] = gdn_conv_out(v0, v1, v2, x, w0, w1, w2, w3);
                v0 = v1; v1 = v2; v2 = x;
            }
            history[c * 3] = v0; history[c * 3 + 1] = v1; history[c * 3 + 2] = v2;
        });
    } else {
        const int64_t tiles = (T + CONV_TILE - 1) / CONV_TILE;
        items(stream, C * tiles, [=](int64_t i) {
            const int c = (int) (i % C);
            const int64_t t0 = (i / C) * CONV_TILE;
            const int64_t t1 = t0 + CONV_TILE < T ? t0 + CONV_TILE : T;
            auto input = [&](int64_t t) -> float { return t >= 0 ? qkv[t * C + c] : history[c * 3 + (int) (t + 3)]; };
            float v0 = input(t0 - 3), v1 = input(t0 - 2), v2 = input(t0 - 1);
            const float w0 = conv_w[c * 4], w1 = conv_w[c * 4 + 1], w2 = conv_w[c * 4 + 2], w3 = conv_w[c * 4 + 3];
            for (int64_t t = t0; t < t1; ++t) {
                const float x = qkv[t * C + c];
                h[t * C + c] = gdn_conv_out(v0, v1, v2, x, w0, w1, w2, w3);
                v0 = v1; v1 = v2; v2 = x;
            }
        });
        items(stream, C, [=](int64_t cc) {   // the history after the chunk (after the tiles read it: in order)
            const int c = (int) cc;
            float v[3];
            for (int k = 0; k < 3; ++k) {
                const int64_t t = T - 3 + k;
                v[k] = t >= 0 ? qkv[t * C + c] : history[c * 3 + (int) (t + 3)];
            }
            history[c * 3] = v[0]; history[c * 3 + 1] = v[1]; history[c * 3 + 2] = v[2];
        });
    }
    // the L2 norm of the 32 q/k heads: one 128-wide work-group per (head, token), head fastest
    groups(stream, 2 * HK * T, S, 4, [=](sycl::nd_item<1> it, float* part) {
        const int head = (int) (it.get_group(0) % (2 * HK));
        const int64_t t = (int64_t) (it.get_group(0) / (2 * HK));
        const int tid = (int) it.get_local_id(0);
        float* x = h + t * C + head * S;
        const float v = x[tid];
        const float sq = warp_sum(it.get_sub_group(), v * v);
        if ((tid & 31) == 0) part[tid >> 5] = sq;
        sycl::group_barrier(it.get_group());
        const float ss = part[0] + part[1] + part[2] + part[3];
        x[tid] = v * sycl::rsqrt(ss + eps);
    });
    check("gdn_conv");
}

void gdn_recurrence(float* state, const float* h, const float* gate, const float* beta, const float* z,
                    const float* gamma, float eps, float* y, uint16_t* y16, int64_t T, void* stream) {
    static const bool serial = std::getenv("STRATA_GDN_REC_HEADS") != nullptr;   // one work-group per head (A/B)
    if (serial || T <= 0) {
        // local: sk[S], sq[S], red[RG][S], wsum[16]
        groups(stream, HV, S * RG, 2 * S + RG * S + 16, [=](sycl::nd_item<1> it, float* lm) {
            float* sk = lm;
            float* sq = lm + S;
            float* red = lm + 2 * S;
            float* wsum = lm + 2 * S + RG * S;
            const int head = (int) it.get_group(0);
            const int tid = (int) it.get_local_id(0), col = tid % S, rg = tid / S;
            const int qh = head % HK;
            float s[RPG];
            float* base = state + ((size_t) (rg * RPG) * HV + head) * S + col;
            const size_t rs = (size_t) HV * S;
#pragma unroll
            for (int r = 0; r < RPG; ++r) s[r] = base[r * rs];
            const float g_col = gamma[col];
            const sycl::sub_group sg = it.get_sub_group();
            for (int64_t t = 0; t < T; ++t) {
                const float* ht = h + t * C;
                sycl::group_barrier(it.get_group());
                if (tid < S) { sq[tid] = ht[qh * S + tid]; sk[tid] = ht[HK * S + qh * S + tid]; }
                sycl::group_barrier(it.get_group());
                const float g = sycl::exp(gate[t * HV + head]);
                float kv = 0.0f;
#pragma unroll
                for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
                red[rg * S + col] = kv;
                sycl::group_barrier(it.get_group());
                const float kv_col = red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col];
                const float delta = (ht[2 * HK * S + head * S + col] - g * kv_col) * beta[t * HV + head];
                float o = 0.0f;
#pragma unroll
                for (int r = 0; r < RPG; ++r) {
                    s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                    o = sycl::fma(s[r], sq[rg * RPG + r], o);
                }
                sycl::group_barrier(it.get_group());
                red[rg * S + col] = o;
                sycl::group_barrier(it.get_group());
                float oc = 0.0f, sp = 0.0f;
                if (rg == 0) {
                    oc = (red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col]) * sycl::rsqrt((float) S);
                    sp = oc * oc;
                }
                sp = warp_sum(sg, sp);
                if ((tid & 31) == 0) wsum[tid >> 5] = sp;
                sycl::group_barrier(it.get_group());
                if (rg == 0) {
                    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                    const float v = gdn_norm_value(oc, ss, eps, g_col, z[t * HV * S + head * S + col]);
                    y[t * HV * S + head * S + col] = v;
                    y16[t * HV * S + head * S + col] = hf(v);
                }
            }
#pragma unroll
            for (int r = 0; r < RPG; ++r) base[r * rs] = s[r];
        });
    } else {
        static const bool pipe = [] { const char* v = std::getenv("STRATA_GDN_PIPELINE"); return v == nullptr || std::atoi(v) != 0; }();
        if (pipe) gdn_rec_cols<true>(state, h, gate, beta, y, T, stream);
        else gdn_rec_cols<false>(state, h, gate, beta, y, T, stream);
        // the output norm over a head's 128 columns, into the FP16 copy: one 128-wide work-group per (token, head)
        groups(stream, T * HV, S, 4, [=](sycl::nd_item<1> it, float* wsum) {
            const int64_t t = (int64_t) (it.get_group(0) % T);
            const int head = (int) (it.get_group(0) / T), col = (int) it.get_local_id(0);
            const size_t at = (size_t) t * HV * S + (size_t) head * S + col;
            const float oc = y[at];
            const float sp = warp_sum(it.get_sub_group(), oc * oc);
            if ((col & 31) == 0) wsum[col >> 5] = sp;
            sycl::group_barrier(it.get_group());
            const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
            y16[at] = hf(gdn_norm_value(oc, ss, eps, gamma[col], z[at]));
        });
    }
    check("gdn_recurrence");
}

void route(const float* logits, int32_t* ids, float* weights, int64_t T, int64_t n_expert, void* stream) {
    auto launch = [&](auto reg) {
        constexpr int REG = decltype(reg)::value;
        groups(stream, (T + 7) / 8, 256, 0, [=](sycl::nd_item<1> it, float*) {
            const int tid = (int) it.get_local_id(0);
            const int64_t t = (int64_t) it.get_group(0) * 8 + (tid >> 5);
            if (t >= T) return;   // whole sub-groups
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = tid & 31;
            const float* lg = logits + t * (REG * 32);
            float v[REG];
#pragma unroll
            for (int i = 0; i < REG; ++i) v[i] = lg[lane + i * 32];
            float mx = -INFINITY;
#pragma unroll
            for (int i = 0; i < REG; ++i) mx = sycl::fmax(mx, v[i]);
            mx = warp_max(sg, mx);
            float sum = 0.0f;
#pragma unroll
            for (int i = 0; i < REG; ++i) { v[i] = sycl::exp(v[i] - mx); sum += v[i]; }
            const float rcp = 1.0f / warp_sum(sg, sum);
#pragma unroll
            for (int i = 0; i < REG; ++i) { v[i] *= rcp; if (sycl::isnan(v[i])) v[i] = -FLT_MAX; }
            float selected = 0.0f, selected_sum = 0.0f;
            for (int rank = 0; rank < 10; ++rank) {
                float best = v[0];
                int ex = lane;
#pragma unroll
                for (int i = 1; i < REG; ++i) if (v[i] > best) { best = v[i]; ex = lane + i * 32; }
#pragma unroll
                for (int m = 16; m; m >>= 1) {
                    const float ob = sycl::permute_group_by_xor(sg, best, m);
                    const int oi = sycl::permute_group_by_xor(sg, ex, m);
                    if (ob > best || (ob == best && oi < ex)) { best = ob; ex = oi; }
                }
                if ((ex & 31) == lane) {
#pragma unroll
                    for (int i = 0; i < REG; ++i) if (i == ex / 32) v[i] = -INFINITY;
                    selected_sum += best;
                }
                if (lane == 0) ids[t * 10 + rank] = ex;
                if (rank == lane) selected = best;
            }
            selected_sum = sycl::fmax(warp_sum(sg, selected_sum), 6.103515625e-5f);
            if (lane < 10) weights[t * 10 + lane] = selected / selected_sum;
        });
    };
    if (n_expert == 512) launch(std::integral_constant<int, 16>{});
    else if (n_expert == 256) launch(std::integral_constant<int, 8>{});
    else strata::kernels::router_top10(logits, (int) T, (int) n_expert, 10, ids, weights, stream);
    check("route");
}

namespace {
// Strata blob: gate/up codes [1280][640 B], down codes [2560][160 B], gate/up scales [1280][40] f16, down [2560][10] f16
template<bool HALF>
void blob_dequant_launch(const uint8_t* blob, uint16_t* gu16, uint16_t* d16, void* stream) {
    constexpr size_t O_D_CODES = (size_t) 1280 * 640, O_GU_SC = O_D_CODES + (size_t) 2560 * 160,
                     O_D_SC = O_GU_SC + (size_t) 1280 * 40 * 2;
    constexpr int64_t n_gu = 1280LL * 640, n_d = 2560LL * 160;
    items(stream, n_gu + n_d, [=](int64_t i) {   // one work-item per 4 weights (one code byte)
        auto put = [&](uint8_t c, const uint8_t* sp, uint16_t* o) {
            const float d = f32_from_f16((uint16_t) (sp[0] | (sp[1] << 8)));
#pragma unroll
            for (int k = 0; k < 4; ++k) {
                const float v = (float) (((c >> (2 * k)) & 3) - 1) * d;
                o[k] = HALF ? hf(v) : bf(v);
            }
        };
        if (i < n_gu) {
            const int64_t row = i / 640, byte = i % 640;
            put(blob[row * 640 + byte], blob + O_GU_SC + (size_t) (row * 40 + (byte * 4) / 64) * 2, gu16 + row * 2560 + byte * 4);
        } else {
            const int64_t j = i - n_gu, row = j / 160, byte = j % 160;
            put(blob[O_D_CODES + row * 160 + byte], blob + O_D_SC + (size_t) (row * 10 + (byte * 4) / 64) * 2,
                d16 + row * 640 + byte * 4);
        }
    });
}
}  // namespace

void blob_dequant(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_launch<false>(blob, gu16, down16, stream);
    check("blob_dequant");
}
void blob_dequant_f16(const uint8_t* blob, uint16_t* gu16, uint16_t* down16, void* stream) {
    blob_dequant_launch<true>(blob, gu16, down16, stream);
    check("blob_dequant_f16");
}
void swiglu_interleaved(const float* gu, uint16_t* h16, int64_t n, void* stream) {
    items(stream, n * 640, [=](int64_t i) {
        const int64_t r = i / 640, k = i % 640;
        const float g = gu[r * 1280 + 2 * k], u = gu[r * 1280 + 2 * k + 1];
        h16[i] = hf_sat(g / (1.0f + sycl::exp(-g)) * u);
    });
    check("swiglu_interleaved");
}
void swiglu_pair(const float* g, const float* u, uint16_t* h16, int64_t n, void* stream) {
    items(stream, n * 640, [=](int64_t i) {
        const float a = g[i];
        h16[i] = hf_sat(a / (1.0f + sycl::exp(-a)) * u[i]);
    });
    check("swiglu_pair");
}
void copy_i32(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    items(stream, n, [=](int64_t i) { dst[i] = src[i]; });
    check("copy_i32");
}
void gather_rows16(const uint16_t* x16, const int32_t* src, uint16_t* dst16, int64_t n, int64_t width, void* stream) {
    const int64_t per = width / 8;
    const sycl::uint4* x = reinterpret_cast<const sycl::uint4*>(x16);
    sycl::uint4* d = reinterpret_cast<sycl::uint4*>(dst16);
    items(stream, n * per, [=](int64_t i) {   // one 16-byte piece (8 halves)
        const int64_t r = i / per, j = i % per;
        d[r * per + j] = x[(int64_t) src[r] * per + j];
    });
    check("gather_rows16");
}
void moe_combine(const float* Dm, const int32_t* slot, const float* w, const float* shared, const float* sg, float* bo,
                 int64_t T, void* stream) {
    items(stream, T * N, [=](int64_t i) {
        const int64_t t = i / N, d = i % N;
        float s = 0.0f;
#pragma unroll
        for (int k = 0; k < 10; ++k) s = sycl::fma(w[t * 10 + k], Dm[(int64_t) slot[t * 10 + k] * N + d], s);
        bo[i] = s + shared[i] * sigm(sg[t]);
    });
    check("moe_combine");
}
void rms_rows(float* x, const float* w, int64_t rows, int64_t cols, int64_t ld, float eps, void* stream) {
    groups(stream, rows, 256, 32, [=](sycl::nd_item<1> it, float* sh) {
        float* r = x + (int64_t) it.get_group(0) * ld;
        const int tid = (int) it.get_local_id(0);
        float ss = 0.0f;
        for (int64_t c = tid; c < cols; c += 256) ss += r[c] * r[c];
        const float s = sycl::rsqrt(block_sum(it, ss, sh) / (float) cols + eps);
        sycl::group_barrier(it.get_group());
        for (int64_t c = tid; c < cols; c += 256) r[c] = s * r[c] * w[c];
    });
    check("rms_rows");
}

void rope(float* x, int64_t T, int64_t heads, int64_t dim, int64_t ld, int64_t pos0,
          const strata::kernels::RopeScaling& scaling, void* stream) {
    if (const char* why = strata::kernels::rope_scaling_invalid(scaling)) {
        std::fprintf(stderr, "prefill rope: invalid rope scaling: %s\n", why);
        std::exit(1);
    }
    const float theta_scale = powf((float) scaling.freq_base, -2.0f / 64.0f);
    const strata::kernels::RopeKernelArgs k = scaling.kernel_args(64);
    const strata::kernels::RopeTab rt = strata::kernels::rope_table_for(scaling);
    const int32_t* mtab = strata::kernels::mrope_table();
    auto launch = [&](auto tab) {
        constexpr bool TAB = decltype(tab)::value;
        items(stream, T * heads * 32, [=](int64_t i) {
            const int64_t row = i / 32;   // t * heads + h
            const int pair = (int) (i % 32);
            const int64_t t = row / heads, hh = row % heads;
            float* p = x + t * ld + hh * dim;
            float c, s;
            if (!(TAB && strata::kernels::rope_tab_cs(rt, strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair), pair, c, s))) {
                const float theta_extrap =
                    (float) strata::kernels::mrope_pos(mtab, (int) (pos0 + t), pair) * sycl::pow(theta_scale, (float) pair);
                strata::kernels::rope_scaled_angle(theta_extrap, k.freq_scale, k.corr_low, k.corr_high, k.ext_factor,
                                                   k.attn_factor, pair, c, s);
            }
            const float a = p[pair], b = p[pair + 32];
            p[pair] = a * c - b * s;
            p[pair + 32] = a * s + b * c;
        });
    };
    if (rt.cos != nullptr) launch(std::true_type{});
    else launch(std::false_type{});
    check("rope");
}
void split_q(const float* q_full, float* q, int64_t T, void* stream) {
    items(stream, T * 24 * 256, [=](int64_t i) {
        const int64_t t = i / (24 * 256), hh = (i / 256) % 24, d = i % 256;
        q[i] = q_full[t * 24 * 512 + hh * 512 + d];
    });
    check("split_q");
}
void gate_attn(const float* attn, const float* q_full, uint16_t* out16, int64_t T, void* stream) {
    items(stream, T * 24 * 256, [=](int64_t i) {
        const int64_t t = i / (24 * 256), hh = (i / 256) % 24, d = i % 256;
        out16[i] = hf(attn[i] * (1.0f / (1.0f + sycl::exp(-q_full[t * 24 * 512 + hh * 512 + 256 + d]))));
    });
    check("gate_attn");
}

}  // namespace strata::prefill
