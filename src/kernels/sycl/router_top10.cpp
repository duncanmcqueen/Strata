// src/kernels/sycl/router_top10.cpp - SYCL port of src/kernels/cuda/router_top10.cu
// (P2.S2: the MoE router's softmax / top-k / renormalise half).  The CUDA file's
// header carries the semantics (softmax over ALL experts, stable descending argsort
// with ties by ascending index, gather, renormalise with ggml's 2**-14 clamp) and
// the bit-exactness contract of each stage; they are not repeated here.  Only what
// SYCL forces to change:
//
//   * `__shfl_down_sync(mask, v, off)` -> `sycl::shift_group_left(sg, v, off)` under a
//     required 32-wide sub-group (the runtime refuses devices without one).
//   * `__shared__` + dynamic smem -> typed local accessors.  The CUDA file's
//     `s_raw` layout dance (taken-mask first, rounded to 16 so the doubles and floats
//     stay aligned) disappears: separate `unsigned char` / `double` / `float`
//     accessors are each correctly aligned by construction.
//   * `exp` on doubles -> `sycl::exp` (a SPIR-V instruction; libm calls are
//     unresolved device imports on this toolchain).  FP64 runs emulated
//     (`-cl-fp64-gen-emu`), as for the other double kernels.
//   * one block per token stays one work-group per token, same thread count.
//
// The arithmetic is line-for-line the CUDA portable kernel's: the max tree of
// fmaxf, the 512 exponentials computed once, the ascending serial double sum on one
// thread, p[] hoisted out of the selection, the k block-wide argmax passes with the
// strict-`>` tie rule, and the renormalise tail.  The default for 64 < n_expert <= 512, k <= 32 is the HIP file's
// fast kernel (launch_fast below; STRATA_SYCL_ROUTER_OLD=1 keeps the portable one).
#include "strata/kernels/router_top10.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>
#include <limits>

namespace strata::kernels {
namespace {

constexpr int RT_MAX_THREADS = 512;

bool check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

void sync_if_needed(sycl::queue& q, void* stream, const char* what) {
    if (stream != nullptr) return;
    q.wait();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr float NEG_INF = -std::numeric_limits<float>::infinity();

// The HIP file's fast kernel (router_top10_fast_kernel), ported: the same router, BIT-IDENTICAL, without its serial
// parts.  The CUDA file's comment above that kernel carries the argument (the tree sum brackets the serial sum, the
// selection on sub-group 0 with the probabilities in registers, the renormalisation from registers).  On the A770
// FP64 is emulated, so the serial parts cost far more than on RDNA4: the portable kernel took 410-760 us per call on
// the Coder's 256 experts, a fifth of the decode GPU time.  `__shfl_xor_sync` -> `permute_group_by_xor`,
// `__shfl_sync` -> `select_from_group`, the tail's `__threadfence` -> a sub-group barrier.
template<bool FORCE_SERIAL>
void launch_fast(sycl::queue& q, const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights) {
    int threads = n_expert < RT_MAX_THREADS ? n_expert : RT_MAX_THREADS;
    threads = (threads + 31) & ~31;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> s_ex(sycl::range<1>((size_t) n_expert), h);
        sycl::local_accessor<float, 1> s_red(sycl::range<1>(RT_MAX_THREADS / 32), h);
        sycl::local_accessor<double, 1> s_dred(sycl::range<1>(RT_MAX_THREADS / 32), h);
        sycl::local_accessor<float, 1> s_inv(sycl::range<1>(1), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) n_tokens * threads), sycl::range<1>((size_t) threads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const int t = static_cast<int>(it.get_group(0));
                const int tid = static_cast<int>(it.get_local_id(0)), lane = tid & 31, warp = tid >> 5;
                const int nt = static_cast<int>(it.get_local_range(0)), nw = (nt + 31) >> 5;
                const float* l = logits + (size_t) t * n_expert;
                auto bar = [&] { sycl::group_barrier(it.get_group()); };
                float mx = NEG_INF;
                for (int e = tid; e < n_expert; e += nt) mx = sycl::fmax(mx, l[e]);
                for (int off = 16; off > 0; off >>= 1)
                    mx = sycl::fmax(mx, sycl::shift_group_left(sg, mx, static_cast<uint32_t>(off)));
                if (lane == 0) s_red[warp] = mx;
                bar();
                if (tid < 32) {
                    float v = (tid < nw) ? s_red[tid] : NEG_INF;
                    for (int off = 16; off > 0; off >>= 1)
                        v = sycl::fmax(v, sycl::shift_group_left(sg, v, static_cast<uint32_t>(off)));
                    if (tid == 0) s_red[0] = v;
                }
                bar();
                mx = s_red[0];
                double part = 0.0;
                for (int e = tid; e < n_expert; e += nt) {
                    const double x = sycl::exp((double) l[e] - (double) mx);
                    s_ex[e] = x;
                    part += x;
                }
                for (int off = 16; off > 0; off >>= 1) part += sycl::permute_group_by_xor(sg, part, off);
                if (lane == 0) s_dred[warp] = part;
                bar();
                double tot = 0.0;
                for (int w = 0; w < nw; ++w) tot += s_dred[w];
                const double lo = tot * (1.0 - 0x1p-40), hi = tot * (1.0 + 0x1p-40);
                const float f_lo = (float) (1.0 / hi), f_hi = (float) (1.0 / lo);
                float inv;
                if (!FORCE_SERIAL && tot >= 1.0 && hi < 1e300 && f_lo == f_hi) {
                    inv = f_lo;
                } else {   // work-group uniform: the portable kernel's serial sum
                    if (tid == 0) {
                        double sum = 0.0;
                        for (int e = 0; e < n_expert; ++e) sum += s_ex[e];
                        s_inv[0] = (float) (1.0 / sum);
                    }
                    bar();
                    inv = s_inv[0];
                }
                if (warp != 0) return;
                constexpr int PER = 16;
                float pv[PER];
                unsigned live = 0;
#pragma unroll
                for (int j = 0; j < PER; ++j) {
                    const int e = lane + 32 * j;
                    pv[j] = NEG_INF;
                    if (e < n_expert) { pv[j] = (float) (s_ex[e] * inv); live |= 1u << j; }
                }
                float my_w = 0.0f;
                int my_id = 0, nsel = 0;
                for (int i = 0; i < k; ++i) {
                    float bv = NEG_INF;
                    int bi = n_expert;
#pragma unroll
                    for (int j = 0; j < PER; ++j)
                        if (((live >> j) & 1u) && pv[j] > bv) { bv = pv[j]; bi = lane + 32 * j; }
                    for (int off = 16; off > 0; off >>= 1) {
                        const float ov = sycl::permute_group_by_xor(sg, bv, off);
                        const int oi = sycl::permute_group_by_xor(sg, bi, off);
                        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                    }
                    if (bi >= n_expert) break;   // every p left is NaN: no rank i or later is written
                    if ((bi & 31) == lane) live &= ~(1u << (bi >> 5));
                    if (lane == i) { my_w = bv; my_id = bi; }
                    nsel = i + 1;
                }
                if (nsel == k) {
                    double s = 0.0;
                    for (int i = 0; i < k; ++i) s += (double) sycl::select_from_group(sg, my_w, i);
                    const double sc = sycl::fmax(s, 6.103515625e-05);       // 2**-14
                    if (lane < k) {
                        ids[(size_t) t * k + lane] = my_id;
                        weights[(size_t) t * k + lane] = (float) ((double) my_w / sc);
                    }
                } else {   // the portable kernel's tail: unwritten ranks keep what the buffer held
                    if (lane < nsel) { ids[(size_t) t * k + lane] = my_id; weights[(size_t) t * k + lane] = my_w; }
                    sycl::group_barrier(sg);
                    if (lane == 0) {
                        double s = 0.0;
                        for (int i = 0; i < k; ++i) s += (double) weights[(size_t) t * k + i];
                        const double sc = sycl::fmax(s, 6.103515625e-05);
                        for (int i = 0; i < k; ++i)
                            weights[(size_t) t * k + i] = (float) ((double) weights[(size_t) t * k + i] / sc);
                    }
                }
            });
    });
}

bool fast_geometry(int n_tokens, int n_expert, int k) {
    return n_tokens > 0 && n_expert > 64 && n_expert <= 512 && k > 0 && k <= 32;
}

void launch_portable(sycl::queue& q, const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights);

}  // namespace

bool router_top10_variant(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                          void* stream, int variant) {
    if (!fast_geometry(n_tokens, n_expert, k) || variant < 0 || variant > 2) return false;
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    if (variant == 0) launch_portable(q, logits, n_tokens, n_expert, k, ids, weights);
    else if (variant == 1) launch_fast<false>(q, logits, n_tokens, n_expert, k, ids, weights);
    else launch_fast<true>(q, logits, n_tokens, n_expert, k, ids, weights);
    sync_if_needed(q, stream, "router_top10_variant");
    return cudaGetLastError() == cudaSuccess;
}

void router_top10(const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights,
                  void* stream) {
    if (n_tokens <= 0 || n_expert <= 0 || k <= 0) return;
    if (k > 64) {
        std::fprintf(stderr, "router_top10: k %d exceeds the kernel's 64\n", k);
        std::exit(1);
    }
    if (n_expert > RT_MAX_THREADS * 64) {
        std::fprintf(stderr, "router_top10: n_expert %d is past the kernel's %d\n", n_expert,
                     RT_MAX_THREADS * 64);
        std::exit(1);
    }
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    static const bool old = std::getenv("STRATA_SYCL_ROUTER_OLD") != nullptr;
    if (!old && fast_geometry(n_tokens, n_expert, k)) launch_fast<false>(q, logits, n_tokens, n_expert, k, ids, weights);
    else launch_portable(q, logits, n_tokens, n_expert, k, ids, weights);
    check_launch("router_top10");
    sync_if_needed(q, stream, "router_top10");
}

namespace {
void launch_portable(sycl::queue& q, const float* logits, int n_tokens, int n_expert, int k, int* ids, float* weights) {
    int threads = n_expert < RT_MAX_THREADS ? n_expert : RT_MAX_THREADS;
    threads = (threads + 31) & ~31;                  // at least one full warp, for the reductions

    q.submit([&](sycl::handler& h) {
        // taken-mask, the exponentials, the probabilities, the warp-reduction scratch, the sum
        sycl::local_accessor<unsigned char, 1> s_taken(sycl::range<1>((size_t) n_expert), h);
        sycl::local_accessor<double, 1> s_ex(sycl::range<1>((size_t) n_expert), h);
        sycl::local_accessor<float, 1> s_p(sycl::range<1>((size_t) n_expert), h);
        sycl::local_accessor<float, 1> s_red(sycl::range<1>(RT_MAX_THREADS / 32), h);
        sycl::local_accessor<int, 1> s_rid(sycl::range<1>(RT_MAX_THREADS / 32), h);
        sycl::local_accessor<double, 1> s_sum(sycl::range<1>(1), h);
        h.parallel_for(
            sycl::nd_range<1>(sycl::range<1>((size_t) n_tokens * threads), sycl::range<1>((size_t) threads)),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const sycl::sub_group sg = it.get_sub_group();
                const int t = static_cast<int>(it.get_group(0));
                const int tid = static_cast<int>(it.get_local_id(0));
                const int nt = static_cast<int>(it.get_local_range(0));
                const float* l = logits + (size_t) t * n_expert;
                auto bar = [&] { sycl::group_barrier(it.get_group()); };

                // ---- softmax over ALL experts, for stability: the max.  A tree of fmaxf is EXACT and
                // order-independent, so this is bit-identical to the serial scan.
                float mx = NEG_INF;
                for (int e = tid; e < n_expert; e += nt) mx = sycl::fmax(mx, l[e]);
                for (int off = 16; off > 0; off >>= 1)
                    mx = sycl::fmax(mx, sycl::shift_group_left(sg, mx, static_cast<uint32_t>(off)));
                if ((tid & 31) == 0) s_red[tid >> 5] = mx;
                bar();
                if (tid < 32) {
                    const int nw = (nt + 31) >> 5;
                    float v = (tid < nw) ? s_red[tid] : NEG_INF;
                    for (int off = 16; off > 0; off >>= 1)
                        v = sycl::fmax(v, sycl::shift_group_left(sg, v, static_cast<uint32_t>(off)));
                    if (tid == 0) s_red[0] = v;
                }
                bar();
                mx = s_red[0];

                // ---- THE 512 EXPONENTIALS, ONCE EACH AND IN PARALLEL.
                for (int e = tid; e < n_expert; e += nt) s_ex[e] = sycl::exp((double) l[e] - (double) mx);
                bar();

                // ---- the sum, ascending, on one thread: the accumulation order - and therefore the last
                // bits - stays identical to the reference-faithful serial version.
                if (tid == 0) {
                    double sum = 0.0;
                    for (int e = 0; e < n_expert; ++e) sum += s_ex[e];
                    s_sum[0] = sum;
                }
                bar();
                const float inv = (float) (1.0 / s_sum[0]);

                // ---- p[] ONCE: `p[e] = (float)(s_ex[e] * inv)` is the old expression exactly, computed
                // once instead of inside the O(n^2) compare.  Bit-identical, not merely equivalent.
                for (int e = tid; e < n_expert; e += nt) s_p[e] = (float) (s_ex[e] * inv);
                bar();

                // ---- the selection: k block-wide argmax passes.  Scanning `e` ascending with a strict `>`
                // keeps the LOWEST index on a tie - a stable descending top-k, ties by index.
                for (int e = tid; e < n_expert; e += nt) s_taken[e] = 0;
                bar();

                for (int i = 0; i < k; ++i) {
                    float bv = NEG_INF;
                    int bi = n_expert;               // a sentinel that loses to every real index
                    for (int e = tid; e < n_expert; e += nt) {
                        if (s_taken[e]) continue;
                        const float pe = s_p[e];
                        if (pe > bv) { bv = pe; bi = e; }
                    }
                    for (int off = 16; off > 0; off >>= 1) {
                        const float ov = sycl::shift_group_left(sg, bv, static_cast<uint32_t>(off));
                        const int oi = sycl::shift_group_left(sg, bi, static_cast<uint32_t>(off));
                        if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
                    }
                    if ((tid & 31) == 0) { s_red[tid >> 5] = bv; s_rid[tid >> 5] = bi; }
                    bar();
                    if (tid < 32) {
                        const int nw = (nt + 31) >> 5;
                        float v = (tid < nw) ? s_red[tid] : NEG_INF;
                        int ix = (tid < nw) ? s_rid[tid] : n_expert;
                        for (int off = 16; off > 0; off >>= 1) {
                            const float ov = sycl::shift_group_left(sg, v, static_cast<uint32_t>(off));
                            const int oi = sycl::shift_group_left(sg, ix, static_cast<uint32_t>(off));
                            if (ov > v || (ov == v && oi < ix)) { v = ov; ix = oi; }
                        }
                        if (tid == 0 && ix < n_expert) {
                            ids[(size_t) t * k + i] = ix;
                            weights[(size_t) t * k + i] = v;
                            s_taken[ix] = 1;
                        }
                    }
                    bar();
                }
                bar();

                // ---- renormalise, with ggml's lower clamp.  Order preserved.
                if (tid == 0) {
                    double s = 0.0;
                    for (int i = 0; i < k; ++i) s += (double) weights[(size_t) t * k + i];
                    const double sc = sycl::fmax(s, 6.103515625e-05);       // 2**-14
                    for (int i = 0; i < k; ++i)
                        weights[(size_t) t * k + i] = (float) ((double) weights[(size_t) t * k + i] / sc);
                }
            });
    });
}
}  // namespace

}  // namespace strata::kernels
