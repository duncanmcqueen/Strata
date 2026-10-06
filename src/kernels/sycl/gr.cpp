// SYCL port of cuda/gr.cu; preserves activation variants and reduction trees.

#include "strata/kernels/gr.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/native_gr_norm.hpp"
#include "strata/kernels/native_gr_postops.hpp"

#include "strata/sycl_runtime/queue_bridge.hpp"
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;
bool fp32_activations = false;
bool native_mmvf = false;

inline uint16_t f32_to_bf16_bits(float f) { return bf16_from_f32(f); }
inline float bf16_bits_to_f32(uint16_t h) { return f32_from_bf16(h); }
inline float activation_f32(float x) { return x; }
inline float activation_f32(uint16_t x) { return bf16_bits_to_f32(x); }
inline void store_activation(float *dst, int i, float x) { dst[i] = x; }
inline void store_activation(uint16_t *dst, int i, float x) { dst[i] = f32_to_bf16_bits(x); }

inline float silu_f(float x) { return x / (1.0f + sycl::exp(-x)); }
inline float sigmoid_f(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

inline float warp_sumf(sycl::nd_item<1> it, float v) {
    for (int off = 16; off > 0; off >>= 1)
        v += sycl::shift_group_left(it.get_sub_group(), v, off);
    return sycl::group_broadcast(it.get_sub_group(), v, 0);
}

float block_sumf(sycl::nd_item<1> it, float v, float *scratch) {

    it.barrier(sycl::access::fence_space::local_space);
    const int lane = it.get_local_id(0) & 31, warp = it.get_local_id(0) >> 5;
    v = warp_sumf(it, v);
    if (lane == 0)
        scratch[warp] = v;
    it.barrier(sycl::access::fence_space::local_space);
    const int nw = ((int)it.get_local_range(0) + 31) >> 5;
    v = (int(it.get_local_id(0)) < nw) ? scratch[it.get_local_id(0)] : 0.0f;
    if (warp == 0)
        v = warp_sumf(it, v);
    if (it.get_local_id(0) == 0)
        scratch[0] = v;
    it.barrier(sycl::access::fence_space::local_space);
    return scratch[0];
}

template <bool FP32_ACT>
void gr_norm_kernel(sycl::queue &queue, size_t groups, size_t threads, const float *__restrict__ R,
                    const float *__restrict__ w_norm, float eps, int n_embd, float *__restrict__ xn,
                    uint16_t *__restrict__ xq) {
    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> scratch_local(8, cgh);
        cgh.parallel_for(sycl::nd_range<1>(groups * threads, threads),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                             auto *scratch = &scratch_local[0];
                             const int c = it.get_group(0);
                             const float *Rc = R + (size_t)c * n_embd;
                             float *xnc = xn + (size_t)c * n_embd;
                             uint16_t *xqc = xq + (size_t)c * n_embd;

                             float ss = 0.0f;
                             for (int d = it.get_local_id(0); d < n_embd; d += it.get_local_range(0)) {
                                 const float v = Rc[d];
                                 ss += v * v;
                             }
                             const float ms = block_sumf(it, ss, scratch) / (float)n_embd;
                             const float rs = sycl::rsqrt(ms + eps);
                             for (int d = it.get_local_id(0); d < n_embd; d += it.get_local_range(0)) {
                                 const float x = Rc[d] * rs * w_norm[(size_t)c * n_embd + d];
                                 xnc[d] = x;
                                 if constexpr (!FP32_ACT)
                                     xqc[d] = f32_to_bf16_bits(x);
                             }
                         });
    });
}

template <typename Activation>
void gr_down_kernel(sycl::queue &queue, size_t groups, size_t threads, const Activation *__restrict__ xq,
                    const uint16_t *__restrict__ w_down, int hc_dim, int hc_lr, int hc,
                    Activation *__restrict__ lq) {
    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> part_local(WARPS, cgh);
        cgh.parallel_for(sycl::nd_range<1>(groups * threads, threads),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                             const int k = it.get_group(0);
                             if (k >= hc_lr)
                                 return;
                             const int lane = it.get_local_id(0) & 31;
                             const int warp = it.get_local_id(0) >> 5;
                             const int nw = (int)(it.get_local_range(0) >> 5);
                             const uint16_t *row = w_down + (size_t)k * hc_dim;

                             float acc = 0.0f;
                             for (int i = warp * 32 + lane; i < hc_dim; i += nw * 32)
                                 acc += activation_f32(xq[i]) * bf16_bits_to_f32(row[i]);
                             acc = warp_sumf(it, acc);

                             auto *part = &part_local[0];
                             if (lane == 0)
                                 part[warp] = acc;
                             it.barrier(sycl::access::fence_space::local_space);
                             if (warp == 0) {
                                 float t = (lane < nw) ? part[lane] : 0.0f;
                                 t = warp_sumf(it, t);

                                 if (lane == 0)
                                     store_activation(lq, k, silu_f(t / (float)hc));
                             }
                         });
    });
}

template <typename Activation>
void gr_gate_kernel(sycl::queue &queue, size_t groups, size_t threads, const Activation *__restrict__ lq,
                    const uint16_t *__restrict__ w_up, const float *__restrict__ xn, int hc_dim, int hc_lr,
                    float *__restrict__ gated) {
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(groups * threads, threads),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                             const int i = it.get_group(0) * WARPS + (it.get_local_id(0) >> 5);
                             if (i >= hc_dim)
                                 return;
                             const int lane = it.get_local_id(0) & 31;
                             const uint16_t *row = w_up + (size_t)i * hc_lr;
                             float acc = 0.0f;
                             for (int k = lane; k < hc_lr; k += 32)
                                 acc += activation_f32(lq[k]) * bf16_bits_to_f32(row[k]);
                             acc = warp_sumf(it, acc);

                             if (lane == 0)
                                 gated[i] = xn[i] * sigmoid_f(acc);
                         });
    });
}

void gr_mean_kernel(sycl::queue &queue, size_t groups, size_t threads, const float *__restrict__ gated,
                    int n_embd, int hc, float *__restrict__ mixed) {
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(groups * threads, threads),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                             const int d = it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);
                             if (d >= n_embd)
                                 return;
                             float m = 0.0f;
                             for (int c = 0; c < hc; ++c)
                                 m += gated[(size_t)c * n_embd + d];
                             mixed[d] = m / (float)hc;
                         });
    });
}

template <typename Activation>
void gr_inject_kernel(sycl::queue &queue, size_t groups, size_t threads, const Activation *__restrict__ xq,
                      const uint16_t *__restrict__ w_inject, int hc_dim, int hc, float *__restrict__ inject) {
    queue.submit([&](sycl::handler &cgh) {
        cgh.parallel_for(sycl::nd_range<1>(groups * threads, threads),
                         [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                             const int c = it.get_local_id(0) >> 5;
                             if (c >= hc)
                                 return;
                             const int lane = it.get_local_id(0) & 31;
                             const uint16_t *row = w_inject + (size_t)c * hc_dim;

                             float acc = 0.0f;
                             for (int i = lane; i < hc_dim; i += 32)
                                 acc += activation_f32(xq[i]) * bf16_bits_to_f32(row[i]);
                             acc = warp_sumf(it, acc);
                             if (lane == 0)
                                 inject[c] = acc;
                         });
    });
}

void gr_write_kernel(sycl::queue &queue, size_t groups, size_t threads, const float *__restrict__ R,
                     const float *__restrict__ block_out, const float *__restrict__ inject, int n_embd,
                     int hc, float *__restrict__ out) {
    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 1> w_local(hc, cgh);
        cgh.parallel_for(
            sycl::nd_range<1>(groups * threads, threads),
            [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                auto *w = &w_local[0];
                if ((int)it.get_local_id(0) < hc)
                    w[it.get_local_id(0)] = 2.0f * sigmoid_f(inject[it.get_local_id(0)] / (float)hc);
                it.barrier(sycl::access::fence_space::local_space);
                const long long n = (long long)hc * n_embd;
                for (long long i = (long long)it.get_group(0) * it.get_local_range(0) + it.get_local_id(0);
                     i < n; i += (long long)it.get_group_range(0) * it.get_local_range(0)) {
                    const int c = (int)(i / n_embd), d = (int)(i % n_embd);

                    out[i] = R[i] + block_out[d] * w[c];
                }
            });
    });
}

} // namespace

void gr_set_fp32_activations(bool enabled) { fp32_activations = enabled; }
void gr_set_native_mmvf(bool enabled) { native_mmvf = enabled; }

size_t gr_workspace_init(const GrShapes &s, void *base, GrWorkspace &out) {
    const size_t hc_dim = (size_t)s.hc * (size_t)s.n_embd;

    const size_t sz[5] = {
        hc_dim * sizeof(float), hc_dim * sizeof(uint16_t),       (size_t)s.hc_lr * sizeof(uint16_t),
        hc_dim * sizeof(float), (size_t)s.hc_lr * sizeof(float),
    };
    size_t al[5], bytes = 0;
    for (int k = 0; k < 5; ++k) {
        al[k] = (sz[k] + 15) & ~(size_t)15;
        bytes += al[k];
    }
    out.bytes = bytes;
    if (base != nullptr) {
        unsigned char *p = (unsigned char *)base;
        void *ptr[5];
        for (int k = 0; k < 5; ++k) {
            ptr[k] = p;
            p += al[k];
        }
        out.xn = (float *)ptr[0];
        out.xq = (uint16_t *)ptr[1];
        out.lq = (uint16_t *)ptr[2];
        out.gated = (float *)ptr[3];
        out.lo = (float *)ptr[4];
    }
    return bytes;
}

void gr_read(const float *R, const float *w_norm, const uint16_t *w_down, const uint16_t *w_up,
             const uint16_t *w_inject, float eps, const GrShapes &s, const GrWorkspace &ws, float *mixed,
             float *inject, void *stream) {
    if (s.n_embd <= 0 || s.hc <= 0 || s.hc_lr <= 0)
        return;
    if (ws.xn == nullptr || ws.xq == nullptr || ws.lq == nullptr || ws.gated == nullptr || ws.lo == nullptr) {
        std::fprintf(stderr, "gr_read: GrWorkspace is not initialised (see gr_workspace_init)\n");
        std::exit(1);
    }
    if (ws.bytes < gr_workspace_bytes(s)) {
        std::fprintf(stderr, "gr_read: GrWorkspace is %zu bytes but this geometry needs %zu\n", ws.bytes,
                     gr_workspace_bytes(s));
        std::exit(1);
    }
    const int n_embd = (int)s.n_embd, hc = (int)s.hc, hc_lr = (int)s.hc_lr;
    const int hc_dim = (int)(s.hc * s.n_embd);
    auto &queue = sycl_runtime::queue_from_stream(stream);

    const bool use_native = native_mmvf;
    const bool use_fp32 = fp32_activations || use_native;
    if (use_native) {
        if ((hc_dim & 1) != 0 || (hc_lr & 1) != 0)
            throw std::invalid_argument("gr_read native MMVF requires even hc*n_embd and hc_lr");
        native_gr_rms_norm_weighted(R, w_norm, ws.xn, n_embd, hc, eps, stream);
        bf16_gemv_fp32_mmvf(ws.xn, w_down, ws.lo, hc_dim, hc_lr, stream);
        native_gr_down_silu(ws.lo, hc_lr, hc, stream);
        bf16_gemv_fp32_mmvf(ws.lo, w_up, ws.gated, hc_lr, hc_dim, stream);

        native_gr_pre_gated(ws.xn, ws.gated, mixed, n_embd, hc, w_inject != nullptr, stream);
    } else if (use_fp32) {
        gr_norm_kernel<true>(queue, hc, THREADS, R, w_norm, eps, n_embd, ws.xn, ws.xq);
        gr_down_kernel<float>(queue, hc_lr, THREADS, ws.xn, w_down, hc_dim, hc_lr, hc, ws.lo);
        gr_gate_kernel<float>(queue, (hc_dim + WARPS - 1) / WARPS, THREADS, ws.lo, w_up, ws.xn, hc_dim, hc_lr,
                              ws.gated);
    } else {
        gr_norm_kernel<false>(queue, hc, THREADS, R, w_norm, eps, n_embd, ws.xn, ws.xq);
        gr_down_kernel<uint16_t>(queue, hc_lr, THREADS, ws.xq, w_down, hc_dim, hc_lr, hc, ws.lq);
        gr_gate_kernel<uint16_t>(queue, (hc_dim + WARPS - 1) / WARPS, THREADS, ws.lq, w_up, ws.xn, hc_dim,
                                 hc_lr, ws.gated);
    }
    if (!use_native)
        gr_mean_kernel(queue, (n_embd + THREADS - 1) / THREADS, THREADS, ws.gated, n_embd, hc, mixed);

    if (w_inject != nullptr) {
        const int nthreads = 32 * hc;
        if (use_native)
            bf16_gemv_fp32_mmvf(ws.xn, w_inject, inject, hc_dim, hc, stream);
        else if (use_fp32)
            gr_inject_kernel<float>(queue, 1, nthreads, ws.xn, w_inject, hc_dim, hc, inject);
        else
            gr_inject_kernel<uint16_t>(queue, 1, nthreads, ws.xq, w_inject, hc_dim, hc, inject);
    }

    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "gr_read launch: %s\n", cudaGetErrorString(e));
        std::exit(1);
    }
    if (stream == nullptr) {
        const cudaError_t se = cudaDeviceSynchronize();
        if (se != cudaSuccess) {
            std::fprintf(stderr, "gr_read: %s\n", cudaGetErrorString(se));
            std::exit(1);
        }
    }
}

void gr_write(const float *R, const float *block_out, const float *inject, const GrShapes &s, float *R_out,
              void *stream) {
    if (s.n_embd <= 0 || s.hc <= 0)
        return;
    const long long n = (long long)s.hc * s.n_embd;
    const int blocks = (int)((n + THREADS - 1) / THREADS);
    auto &queue = sycl_runtime::queue_from_stream(stream);
    if (native_mmvf)
        native_gr_post(R, block_out, inject, R_out, (int)s.n_embd, (int)s.hc, stream);
    else
        gr_write_kernel(queue, blocks, THREADS, R, block_out, inject, (int)s.n_embd, (int)s.hc, R_out);
    if (stream == nullptr) {
        const cudaError_t e = cudaDeviceSynchronize();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "gr_write: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

} // namespace strata::kernels
