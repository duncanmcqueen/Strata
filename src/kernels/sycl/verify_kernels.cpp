// src/kernels/sycl/verify_kernels.cpp - SYCL port of src/kernels/cuda/verify_kernels.cu; see
// include/strata/kernels/verify_kernels.hpp.
//
// The per-token arithmetic is the single-token kernels' (fused_gdn.cpp, elementwise.cpp) in the same order, so a
// verify window reproduces plain decode bit for bit on this backend too.
//
// What SYCL forces to change:
//   * CUDA (x, y) grids flatten to 1D work-groups (x fastest); shared memory is local accessors.
//   * every device read of mapped host memory the HOST updates (the spin-wait flags and the copies "from mapped")
//     is an UNCACHED load (sycl_ext_intel_cache_controls read hint, L1 + L3).  Measured on the A770: a running
//     kernel never saw a host store to host USM through a plain, volatile, device- or system-scope atomic load, or
//     after a system-scope acquire fence (2,000,000 polls each, all served from the GPU cache); the first spin wait
//     ran until the driver reset the device.  The uncached hint saw it at once.  CUDA's `__threadfence()` is a
//     device-scope fence.
//   * `gpu_stamp` reads sycl_ext_oneapi_clock's device counter and scales it to ns with the device's profiling
//     timer resolution; a device without that clock (the A770) throws instead of recording wrong times.
//   * 16-byte vector copies are `sycl::uint4`/`sycl::float4`.
#include "strata/kernels/verify_kernels.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "fp_exact.hpp"
#include "mapped_host.hpp"

#include <cuda_runtime.h>
#include <sycl/ext/intel/math.hpp>

#include <cstdio>
#include <cstdlib>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // GDN state size
constexpr int RG = 4;
constexpr int RPG = S / RG;

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}
sycl::queue& Q(void* stream) { return sycl_runtime::queue_from_stream(stream); }

using sycl_mapped::load;
using sycl_mapped::spin_until;
template<class T> T load_uncached(const T* p, size_t i = 0) { return load(p, i); }

// work-groups of `wg`, `groups` of them, flattened (x fastest); body(it)
template<class F> void launch(void* stream, size_t groups, size_t wg, F body) {
    if (groups == 0) return;
    Q(stream).parallel_for(sycl::nd_range<1>(groups * wg, wg), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] { body(it); });
}

}  // namespace

void fetch_blobs(const unsigned long long* src, const int32_t* n, uint8_t* dst, int64_t blob_bytes, int cap, void* stream) {
    if (cap <= 0) return;
    if (blob_bytes % 16 != 0) { std::fprintf(stderr, "fetch_blobs: blob size must be a multiple of 16\n"); std::exit(1); }
    const long long per = (long long) (blob_bytes / 16);
    sycl::uint4* d = (sycl::uint4*) dst;
    launch(stream, 48 * 8, 256, [=](sycl::nd_item<1> it) {
        const long long total = (long long) *n * per;
        for (long long i = (long long) it.get_global_id(0); i < total; i += (long long) it.get_global_range(0)) {
            const long long k = i / per, off = i - k * per;
            d[i] = ((const sycl::uint4*) src[k])[off];
        }
    });
    check("fetch_blobs");
}

void rebase_ptrs(unsigned long long* ptr, const int32_t* n, uint8_t* base, int64_t blob_bytes, void* stream) {
    const unsigned long long b = (unsigned long long) base;
    const long long bytes = (long long) blob_bytes;
    launch(stream, 1, 128, [=](sycl::nd_item<1> it) {
        const int k = (int) it.get_local_id(0);
        if (k < *n) ptr[k] = b + (unsigned long long) k * (unsigned long long) bytes;
    });
    check("rebase_ptrs");
}

void add_streams_broadcast(const float* h, const float* e, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    const int64_t n = n_embd;
    Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) (n * hc)), [=](sycl::id<2> id) {
        const size_t t = id[0], i = id[1];
        R[t * n * hc + i] = h[t * n * hc + i] + e[t * n + i % n];
    });
    check("add_streams_broadcast");
}

void ident_hits(const int32_t* ids, int n, int32_t* slot, int32_t* dst, int32_t* count, void* stream) {
    if (n < 1 || n > 1024) { std::fprintf(stderr, "ident_hits: n out of range\n"); std::exit(1); }
    Q(stream).parallel_for(sycl::range<1>(1024), [=](sycl::id<1> id) {
        const int i = (int) id[0];
        if (i < n) { slot[i] = ids[i]; dst[i] = i; }
        if (i == 0) *count = n;
    });
    check("ident_hits");
}

void mtp_select(const float* R_src, int64_t R_stride, const int32_t* ids, const int32_t* row_dev, float* R_dst,
                int32_t* tok_dst, int32_t* out, int j, void* stream, const float* probs, float* out_p) {
    const int64_t stride = R_stride;
    launch(stream, 16, 256, [=](sycl::nd_item<1> it) {
        const int row = *row_dev;
        for (int64_t i = (int64_t) it.get_global_id(0); i < stride; i += (int64_t) it.get_global_range(0))
            R_dst[i] = R_src[(size_t) row * stride + i];
        if (it.get_global_id(0) == 0) {
            const int32_t tok = ids[row];
            *tok_dst = tok;
            if (out != nullptr) sycl_mapped::store(out, tok, (size_t) j);   // mapped: the host reads it mid-stream
            if (probs != nullptr && out_p != nullptr) sycl_mapped::store(out_p, probs[row], (size_t) j);
        }
    });
    check("mtp_select");
}

void gather_rows(const uint8_t* src, int64_t row_bytes, const int32_t* ids, int64_t n, uint8_t* dst, void* stream) {
    auto run = [&](auto* s, auto* d, long long row_e) {
        launch(stream, 48 * 8, 256, [=](sycl::nd_item<1> it) {
            const long long total = n * row_e;
            for (long long i = (long long) it.get_global_id(0); i < total; i += (long long) it.get_global_range(0)) {
                const long long r = i / row_e, o = i - r * row_e;
                d[i] = s[(long long) ids[r] * row_e + o];
            }
        });
    };
    // the widest element the row size divides into (16, 4 or 1 bytes): a Q6_K head row of 2560 values is 2100 bytes
    if (row_bytes % 16 == 0) run((const sycl::uint4*) src, (sycl::uint4*) dst, row_bytes / 16);
    else if (row_bytes % 4 == 0) run((const uint32_t*) src, (uint32_t*) dst, row_bytes / 4);
    else run(src, dst, row_bytes);
    check("gather_rows");
}

void map_ids(int32_t* ids, const int32_t* table, int n, void* stream) {
    Q(stream).parallel_for(sycl::range<1>(64), [=](sycl::id<1> id) {
        const int i = (int) id[0];
        if (i < n) ids[i] = table[ids[i]];
    });
    check("map_ids");
}

void row_top_prob(const float* logits, int n_rows, int n_vocab, const int32_t* ids, float* probs, void* stream) {
    Q(stream).submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_rows * 1024, 1024), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int t = (int) it.get_group(0), tid = (int) it.get_local_id(0);
            const float* l = logits + (size_t) t * n_vocab;
            const float m = l[ids[t]];
            float s = 0.0f;
            for (int i = tid; i < n_vocab; i += 1024) s += sycl::exp(l[i] - m);
            const sycl::sub_group sg = it.get_sub_group();
            for (int o = 16; o > 0; o >>= 1) s += sycl::permute_group_by_xor(sg, s, o);
            if ((tid & 31) == 0) part[tid >> 5] = s;
            sycl::group_barrier(it.get_group());
            if (tid == 0) {
                float tot = 0.0f;
                for (int w = 0; w < 1024 / 32; ++w) tot += part[w];
                probs[t] = 1.0f / tot;
            }
        });
    });
    check("row_top_prob");
}

void window_ids(int32_t* steps, int n, int window, int32_t* ids, int64_t ids_stride, void* stream) {
    const long long stride = (long long) ids_stride;
    // CUDA's (8, n) grid: the 8 x-blocks of each query stride its ids; block 0 then records the width.  The width
    // write needs no barrier with the ids of other blocks (none reads them), only within its own work-group.
    launch(stream, (size_t) 8 * n, 256, [=](sycl::nd_item<1> it) {
        const int bx = (int) (it.get_group(0) % 8), q = (int) (it.get_group(0) / 8);
        int32_t* st = steps + q * 4;
        const int n_kv = st[1];
        const int start = n_kv > window ? n_kv - window : 0;
        const int width = n_kv - start;
        for (int j = bx * 256 + (int) it.get_local_id(0); j < width; j += 8 * 256) ids[q * stride + j] = start + j;
        sycl::group_barrier(it.get_group());
        if (bx == 0 && it.get_local_id(0) == 0) st[3] = width;
    });
    check("window_ids");
}

void dense_steps(const int32_t* cells, int n, int32_t* steps, void* stream) {
    Q(stream).parallel_for(sycl::range<1>(64), [=](sycl::id<1> id) {
        const int i = (int) id[0];
        if (i >= n) return;
        const int c = cells[i];
        steps[i * 4 + 0] = c;
        steps[i * 4 + 1] = c + 1;
        steps[i * 4 + 2] = (c + 1) / 4;
        steps[i * 4 + 3] = c + 1;
    });
    check("dense_steps");
}

void gdn_conv_l2_multi(const float* history, const float* qkv, const float* conv_w, float* h, int channels,
                       int qk_heads, float eps, int n_tok, void* stream, int t_begin) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_conv_l2_multi: invalid arguments\n");
        std::exit(1);
    }
    const int C = channels, heads = channels / S;
    Q(stream).submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(S / 32), cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) heads * n_tok * S, S), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int bx = (int) (it.get_group(0) % heads), t = t_begin + (int) (it.get_group(0) / heads);
            const int tid = (int) it.get_local_id(0);
            const int c = bx * S + tid;
            // the window of token t: [hist0, hist1, hist2, x_0, ..., x_t], its last four entries
            float win[3];
#pragma unroll
            for (int j = 0; j < 3; ++j) {
                const int src = t + j;
                win[j] = src < 3 ? history[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
            }
            const float v0 = win[0], v1 = win[1], v2 = win[2], x = qkv[(size_t) t * C + c];
            const float sum = v0 * conv_w[c * 4] + v1 * conv_w[c * 4 + 1] + v2 * conv_w[c * 4 + 2] + x * conv_w[c * 4 + 3];
            float y = sum / (1.0f + sycl::exp(-sum));
            if (bx < qk_heads) {   // uniform over the work-group
                float sq = y * y;
                const sycl::sub_group sg = it.get_sub_group();
                for (int o = 16; o > 0; o >>= 1) sq += sycl::permute_group_by_xor(sg, sq, o);
                if ((tid & 31) == 0) part[tid >> 5] = sq;
                sycl::group_barrier(it.get_group());
                const float ss = part[0] + part[1] + part[2] + part[3];
                y *= sycl::rsqrt(ss + eps);
            }
            h[(size_t) t * C + c] = y;
        });
    });
    check("gdn_conv_l2_multi");
}

void gdn_conv_commit(float* history, const float* qkv, int channels, const int32_t* n_keep, void* stream) {
    const int C = channels;
    Q(stream).parallel_for(sycl::range<1>((size_t) C), [=](sycl::id<1> id) {
        const int c = (int) id[0];
        const int n = *n_keep;
        if (n <= 0) return;
        float seq[3];
#pragma unroll
        for (int j = 0; j < 3; ++j) {
            const int src = n + j;   // the last three of [hist(3) | x_0..x_{n-1}]
            seq[j] = src < 3 ? history[c * 3 + src] : qkv[(size_t) (src - 3) * C + c];
        }
        history[c * 3] = seq[0];
        history[c * 3 + 1] = seq[1];
        history[c * 3 + 2] = seq[2];
    });
    check("gdn_conv_commit");
}

void gdn_ab_multi(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, int n_tok, void* stream) {
    if (n_embd % 8 != 0 || n_tok < 1 || n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_ab_multi: invalid arguments\n");
        std::exit(1);
    }
    const int n = n_embd, T = n_tok;
    launch(stream, (size_t) ((2 * h_v + 7) / 8), 256, [=](sycl::nd_item<1> it) {
        const int tid = (int) it.get_local_id(0);
        const int row = (int) it.get_group(0) * 8 + (tid >> 5), lane = tid & 31;
        if (row >= 2 * h_v) return;   // whole sub-groups leave together
        const bool is_beta = row >= h_v;
        const int r = is_beta ? row - h_v : row;
        const uint32_t* w4 = reinterpret_cast<const uint32_t*>((is_beta ? w_beta : w_alpha) + (size_t) r * n);
        float acc[kVerifyMaxT];
#pragma unroll
        for (int t = 0; t < kVerifyMaxT; ++t) acc[t] = 0.0f;
        for (int j = lane; j < n / 8; j += 32) {
            const uint32_t* wv = w4 + 4 * j;
#pragma unroll
            for (int t = 0; t < kVerifyMaxT; ++t) {
                if (t >= T) break;
                const float* xa = x + (size_t) t * n + j * 8;
                float a = acc[t];
#pragma unroll
                for (int e = 0; e < 4; ++e) {   // fused_gdn_ab's order: low half then high half of each word
                    a = sycl::fma(sycl::bit_cast<float>(wv[e] << 16), xa[2 * e], a);
                    a = sycl::fma(sycl::bit_cast<float>(wv[e] & 0xffff0000u), xa[2 * e + 1], a);
                }
                acc[t] = a;
            }
        }
        const sycl::sub_group sg = it.get_sub_group();
#pragma unroll
        for (int t = 0; t < kVerifyMaxT; ++t) {
            if (t >= T) break;
            float a = acc[t];
            for (int o = 16; o > 0; o >>= 1) a += sycl::permute_group_by_xor(sg, a, o);
            if (lane != 0) continue;
            if (is_beta) {
                beta[(size_t) t * h_v + r] = 1.0f / (1.0f + sycl::exp(-a));
            } else {
                const float v = a + dt[r];
                const float sp = v > 20.0f ? v : sycl::log1p(sycl::exp(v));
                gate[(size_t) t * h_v + r] = sp * ssm_a[r];
            }
        }
    });
    check("gdn_ab_multi");
}

void gdn_step_norm_multi(float* state, const float* hbuf, int conv_channels, const float* gate, const float* beta,
                         const float* z, const float* gamma, float eps, float* y, int h_k, int h_v, int n_tok,
                         const int32_t* n_keep, void* stream, int t_out_begin) {
    if (!state || !hbuf || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v % h_k || n_tok < 1 ||
        n_tok > kVerifyMaxT) {
        std::fprintf(stderr, "gdn_step_norm_multi: invalid arguments\n");
        std::exit(1);
    }
    const int C = conv_channels, T = n_tok;
    Q(stream).submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> sk(sycl::range<1>(S), cgh), sq(sycl::range<1>(S), cgh);
        sycl::local_accessor<float, 1> red(sycl::range<1>(RG * S), cgh), wsum(sycl::range<1>(S * RG / 32), cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) h_v * S * RG, S * RG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int head = (int) it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            const int col = tid % S, rg = tid / S;
            const int qh = head % h_k;
            const int qk = S * h_k;   // q at [0, qk), k at [qk, 2qk), v at [2qk, ...)
            const int value_dim = S * h_v;
            const int n = n_keep ? *n_keep : T;
            float s[RPG];
            float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
            const size_t row_stride = (size_t) h_v * S;
#pragma unroll
            for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
            const sycl::sub_group sg = it.get_sub_group();
            for (int t = 0; t < n; ++t) {
                const float* ht = hbuf + (size_t) t * C;
                sycl::group_barrier(it.get_group());   // the previous token is done with sk/sq/red/wsum
                if (tid < S) { sk[tid] = ht[qk + qh * S + tid]; sq[tid] = ht[qh * S + tid]; }
                sycl::group_barrier(it.get_group());
                const float g = sycl::exp(gate[(size_t) t * h_v + head]);
                float kv = 0.0f;
#pragma unroll
                for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
                red[rg * S + col] = kv;
                sycl::group_barrier(it.get_group());
                const float kv_col = red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col];
                const float delta = (ht[2 * qk + head * S + col] - g * kv_col) * beta[(size_t) t * h_v + head];
                float o = 0.0f;
#pragma unroll
                for (int r = 0; r < RPG; ++r) {
                    s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                    o = sycl::fma(s[r], sq[rg * RPG + r], o);
                }
                sycl::group_barrier(it.get_group());
                red[rg * S + col] = o;
                sycl::group_barrier(it.get_group());
                float oc = 0.0f, sq_part = 0.0f;
                if (rg == 0) {
                    oc = (red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col]) * sycl::rsqrt((float) S);
                    sq_part = oc * oc;
                }
                if (t < t_out_begin) continue;   // a replayed token: its state update is needed, its output is not
                for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += sycl::permute_group_by_xor(sg, sq_part, o2);
                if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
                sycl::group_barrier(it.get_group());
                if (rg == 0) {
                    const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                    const float scale = sycl::rsqrt(ss / (float) S + eps);
                    const float zz = z[(size_t) t * value_dim + head * S + col];
                    y[(size_t) t * value_dim + head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + sycl::exp(-zz)));
                }
            }
            if (n_keep != nullptr && n > 0) {
#pragma unroll
                for (int r = 0; r < RPG; ++r) base[r * row_stride] = s[r];
            }
        });
    });
    check("gdn_step_norm_multi");
}

void resident_plan(const int32_t* ids, int n_entries, int k, const int32_t* res_layer, int n_expert,
                   const uint8_t* cache_base, const unsigned long long* slot_off, long long blob, int32_t* plan,
                   long long capx, uint32_t* skip, uint32_t ring, void* stream) {
    const int n = n_entries;
    Q(stream).single_task([=]() {
        // one work-item: at most kVerifyMaxT * 10 entries, the host's exact loop
        for (int i = 0; i < n; ++i) {
            const int32_t e = ids[i];
            if (e < 0 || e >= n_expert || res_layer[e] < 0) { *skip = 0; return; }
        }
        int32_t* counts = plan;
        int32_t* start = plan + 4;
        int32_t* dst = start + capx + 1;
        int32_t* tok = dst + capx;
        const long long ptr_off = ((4 + (capx + 1) + 2 * capx) + 1) & ~1ll;
        unsigned long long* ptr = (unsigned long long*) (plan + ptr_off);
        int32_t* start2 = plan + ptr_off + 4 * capx;
        int groups = 0, entries = 0;
        for (int i0 = 0; i0 < n; ++i0) {
            bool first = true;
            for (int j = 0; j < i0; ++j)
                if (ids[j] == ids[i0]) { first = false; break; }
            if (!first) continue;
            const int32_t slot = res_layer[ids[i0]];
            ptr[groups] = (unsigned long long) (cache_base + (slot_off ? (size_t) slot_off[slot] : (size_t) slot * (size_t) blob));
            start[groups] = entries;
            for (int i = i0; i < n; ++i)
                if (ids[i] == ids[i0]) {
                    dst[entries] = i;
                    tok[entries] = i / k;
                    ++entries;
                }
            ++groups;
        }
        start[groups] = entries;
        start2[0] = entries;
        counts[0] = groups;
        counts[1] = entries;
        counts[2] = 0;
        sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::device);
        *skip = ring;
    });
    check("resident_plan");
}

void wait_flag_ge_or(const uint32_t* flag, uint32_t value, const uint32_t* skip, void* stream) {
    Q(stream).single_task([=]() {
        if (*skip == value) return;   // device memory
        spin_until(flag, [=](uint32_t v) { return v >= value; });
    });
    check("wait_flag_ge_or");
}

void copy_i32_from_mapped_unless(int32_t* dst, const int32_t* src, long long n, const uint32_t* skip, uint32_t value,
                                 void* stream) {
    if (n <= 0) return;
    const int nn = (int) n;
    launch(stream, 1, 128, [=](sycl::nd_item<1> it) {
        if (*skip == value) return;
        for (int i = (int) it.get_local_id(0); i < nn; i += 128) dst[i] = load_uncached(src, (size_t) i);
    });
    check("copy_i32_from_mapped_unless");
}

void copy_or_zero_from_mapped(float* dst, const float* src, long long n, const uint32_t* skip, uint32_t value,
                              void* stream) {
    if (n <= 0) return;
    const long long n4 = n / 4;
    const int blocks = (int) ((n4 + 255) / 256 < 64 ? (n4 + 255) / 256 : 64);
    sycl::float4* d = (sycl::float4*) dst;
    const float* s = src;
    launch(stream, (size_t) blocks, 256, [=](sycl::nd_item<1> it) {
        const bool zero = *skip == value;
        for (long long i = (long long) it.get_global_id(0); i < n4; i += (long long) it.get_global_range(0))
            d[i] = zero ? sycl::float4(0.f, 0.f, 0.f, 0.f)
                        : sycl::float4(load_uncached(s, 4 * i), load_uncached(s, 4 * i + 1), load_uncached(s, 4 * i + 2),
                                       load_uncached(s, 4 * i + 3));
    });
    check("copy_or_zero_from_mapped");
}

void wait_flag_ge(const uint32_t* flag, uint32_t value, void* stream) {
    Q(stream).single_task([=]() {
        spin_until(flag, [=](uint32_t v) { return v >= value; });
    });
    check("wait_flag_ge");
}

void embedding_gather_dev(const uint8_t* codes, const float* scales, const float* offsets, const int32_t* tokens,
                          int n_tok, int64_t n, int code_bits, int code_bias, int group_elems, uint64_t row_codes,
                          uint64_t row_groups, float* out, void* stream) {
    Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) n), [=](sycl::id<2> id) {
        const size_t t = id[0];
        const int64_t i = (int64_t) id[1];
        const unsigned long long token = (unsigned long long) tokens[t];
        const uint8_t* c = codes + token * row_codes;
        const float* sc = scales + token * row_groups;
        const float* of = offsets ? offsets + token * row_groups : nullptr;
        const int per_byte = 8 / code_bits;
        const unsigned mask = (1u << code_bits) - 1u;
        const int code = (c[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
        const int64_t group = i / group_elems;
        const float product = strata::kernels::fp_exact::fmul_rn((float) (code + code_bias), sc[group]);
        out[t * n + i] = strata::kernels::fp_exact::fadd_rn(product, of ? of[group] : 0.0f);
    });
    check("embedding_gather_dev");
}

void broadcast_streams(const float* x, float* R, int64_t n_embd, int hc, int n_tok, void* stream) {
    const int64_t n = n_embd;
    Q(stream).parallel_for(sycl::range<2>((size_t) n_tok, (size_t) (n * hc)), [=](sycl::id<2> id) {
        const size_t t = id[0], i = id[1];
        R[t * n * hc + i] = x[t * n + i % n];
    });
    check("broadcast_streams");
}

void copy_indexed(float* dst, const float* src, int64_t stride, const int32_t* index, int64_t n, void* stream) {
    const unsigned blocks = (unsigned) ((n + 255) / 256 < 64 ? (n + 255) / 256 : 64);
    launch(stream, blocks, 256, [=](sycl::nd_item<1> it) {
        const int idx = *index;
        if (idx < 0) return;
        for (int64_t i = (int64_t) it.get_global_id(0); i < n; i += (int64_t) it.get_global_range(0))
            dst[i] = src[(size_t) idx * stride + i];
    });
    check("copy_indexed");
}

// gpu_stamp: src/kernels/sycl/gpu_stamp.cpp (in the base kernel library: the fused GR read stamps too)

}  // namespace strata::kernels
