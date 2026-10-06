// Scratch-free SYCL prompt attention. Each (query, KV head) owns one work-group
// and walks its selected cells with an online softmax. FP32 subgroup dot products
// replace CUDA's matrix instructions; no tensor-core performance claim is made.
#include "strata/kernels/qsa_prompt_attn.hpp"
#include "kv_helpers.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include <cfloat>
#include <cstdlib>

namespace strata::kernels {
namespace {
constexpr int HD = 256, G = 12, CH = 32, THREADS = 256, WARPS = THREADS / 32;

template <int MODE> inline float load(const QsaAttnPools &p, bool value, long long row, int d) {
    if constexpr (MODE == 0) {
        return f32_from_f16((value ? p.v_pool : p.k_pool)[row * HD + d]);
    } else if constexpr (MODE == 1 || MODE == 3) {
        if (MODE == 1 || !value) {
            const auto *codes = value ? p.v_q : p.k_q;
            const auto *scales = value ? p.v_scale : p.k_scale;
            return float(codes[row * HD + d]) * f32_from_f16(scales[row * 4 + d / KV_Q8_GROUP]);
        }
    }
    if constexpr (MODE == 2 || MODE == 3) {
        const auto *blk =
            reinterpret_cast<const block_q4_0 *>((value ? p.v_q4 : p.k_q4) + row * 144) + d / QK4_0;
        const int rem = d % QK4_0;
        const uint8_t byte = blk->qs[rem % 16];
        const int code = rem < 16 ? int(byte & 15) - 8 : int(byte >> 4) - 8;
        return float(code) * f32_from_f16(blk->d);
    }
    return 0.0f;
}
inline float sum(sycl::nd_item<1> it, float v) {
    for (int o = 16; o > 0; o >>= 1)
        v += sycl_kv::xor_lane(it, v, o);
    return v;
}
inline float maximum(sycl::nd_item<1> it, float v) {
    for (int o = 16; o > 0; o >>= 1)
        v = sycl::fmax(v, sycl_kv::xor_lane(it, v, o));
    return v;
}

template <int MODE>
void launch(const float *q, QsaAttnPools pools, const int32_t *ids, const int32_t *steps, int64_t cap,
            int page_size, float *attn, int64_t nq, void *stream) {
    strata::sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 1> queries(G * HD, h), probs(G * CH, h);
        sycl::local_accessor<long long, 1> rows(CH, h);
        sycl::local_accessor<float, 1> maxima(G, h), denominators(G, h), alphas(G, h);
        h.parallel_for(
            sycl::nd_range<1>(size_t(nq) * 2 * THREADS, THREADS),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const size_t qi = it.get_group(0) / 2;
                const int kvh = int(it.get_group(0) % 2), t = int(it.get_local_id(0)), lane = t & 31,
                          warp = t >> 5;
                const int n = steps[qi * kStepCount + kStepWidth];
                for (int i = t; i < G * HD; i += THREADS)
                    queries[i] = q[(qi * 24 + kvh * G) * HD + i];
                if (t < G) {
                    maxima[t] = -FLT_MAX;
                    denominators[t] = 0.0f;
                }
                float acc[G] = {};
                it.barrier(sycl::access::fence_space::local_space);
                for (int c0 = 0; c0 < n; c0 += CH) {
                    const int here = sycl::min(CH, n - c0);
                    if (t < CH) {
                        long long row = -1;
                        if (t < here) {
                            const int cell = ids[qi * size_t(cap) + c0 + t];
                            const int page = pools.page_table[cell / page_size];
                            if (page >= 0)
                                row = (static_cast<long long>(page) * 2 + kvh) * page_size + cell % page_size;
                        }
                        rows[t] = row;
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int c = warp; c < CH; c += WARPS) {
                        if (rows[c] < 0) {
                            if (lane < G)
                                probs[lane * CH + c] = -FLT_MAX;
                            continue;
                        }
                        float key[8];
                        for (int d = 0; d < 8; ++d)
                            key[d] = load<MODE>(pools, false, rows[c], lane * 8 + d);
                        for (int head = 0; head < G; ++head) {
                            float dot = key[0] * queries[head * HD + lane * 8];
                            for (int d = 1; d < 8; ++d)
                                dot += key[d] * queries[head * HD + lane * 8 + d];
                            dot = sum(it, dot);
                            if (lane == 0)
                                probs[head * CH + c] = dot * (1.0f / 16.0f);
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    for (int head = warp; head < G; head += WARPS) {
                        const float score = probs[head * CH + lane];
                        const float m = sycl::fmax(maxima[head], maximum(it, score));
                        const float alpha = denominators[head] > 0.0f ? sycl::exp(maxima[head] - m) : 0.0f;
                        const float p = rows[lane] >= 0 ? sycl::exp(score - m) : 0.0f;
                        probs[head * CH + lane] = p;
                        const float l = sum(it, p);
                        // All lanes finish reading the previous softmax state before lane 0 updates it.
                        sycl::group_barrier(it.get_sub_group());
                        if (lane == 0) {
                            maxima[head] = m;
                            alphas[head] = alpha;
                            denominators[head] = sycl::fma(denominators[head], alpha, l);
                        }
                    }
                    it.barrier(sycl::access::fence_space::local_space);
                    // Accumulate each tile separately: keep the long-context sum to one addition per tile.
                    float tile_acc[G] = {};
                    for (int c = 0; c < here; ++c) {
                        if (rows[c] < 0)
                            continue;
                        const float v = load<MODE>(pools, true, rows[c], t);
                        for (int head = 0; head < G; ++head)
                            tile_acc[head] = sycl::fma(probs[head * CH + c], v, tile_acc[head]);
                    }
                    for (int head = 0; head < G; ++head)
                        acc[head] = sycl::fma(acc[head], alphas[head], tile_acc[head]);
                    it.barrier(sycl::access::fence_space::local_space);
                }
                for (int head = 0; head < G; ++head) {
                    const float l = denominators[head];
                    attn[(qi * 24 + kvh * G + head) * HD + t] = l > 0.0f ? acc[head] / l : 0.0f;
                }
            });
    });
}
} // namespace

bool qsa_prompt_attn_batch(const float *q, const QsaAttnPools &pools, const int32_t *ids,
                           const int32_t *steps, int64_t cap, const QsaShapes &s, float *attn, int64_t nq,
                           void *stream) {
    if (nq <= 0)
        return true;
    if (s.head_dim != HD || s.n_head != 24 || s.n_head_kv != 2 || s.page_size <= 0 || cap <= 0 || !q ||
        !ids || !steps || !attn || !pools.page_table)
        return false;
    if (pools.k_q4) {
        const char *off = std::getenv("STRATA_PROMPT_ATTN_Q4");
        if ((off && off[0] == '0') || !pools.v_q4)
            return false;
        launch<2>(q, pools, ids, steps, cap, int(s.page_size), attn, nq, stream);
    } else if (pools.k_q && pools.v_q4) {
        if (!pools.k_scale)
            return false;
        launch<3>(q, pools, ids, steps, cap, int(s.page_size), attn, nq, stream);
    } else if (pools.k_q) {
        if (!pools.v_q || !pools.k_scale || !pools.v_scale)
            return false;
        launch<1>(q, pools, ids, steps, cap, int(s.page_size), attn, nq, stream);
    } else {
        if (!pools.k_pool || !pools.v_pool)
            return false;
        launch<0>(q, pools, ids, steps, cap, int(s.page_size), attn, nq, stream);
    }
    return true;
}
} // namespace strata::kernels
