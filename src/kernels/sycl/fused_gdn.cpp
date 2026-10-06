// src/kernels/sycl/fused_gdn.cpp - SYCL port of src/kernels/cuda/fused_gdn.cu; see
// include/strata/kernels/fused_gdn.hpp.
//
// The CUDA (128, 4) blocks become 512-wide work-groups with tid = rg * 128 + col, so sub-groups 0-3 are row group
// 0 as CUDA's warps 0-3 are; shared arrays become local accessors; `__expf`/`log1pf` become `sycl::exp`/
// `sycl::log1p`, `fmaf` `sycl::fma`, `rsqrtf` `sycl::rsqrt`; the BF16 weight words are read as four 32-bit loads.
#include "strata/kernels/fused_gdn.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int S = 128;          // state size (rows = cols = 128)
constexpr int RG = 4;           // row groups
constexpr int RPG = S / RG;     // 32 rows per work-item

void check(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(1); }
}

}  // namespace

void fused_gdn_conv_l2(float* history, const float* qkv, const float* conv_w, float* h, int channels, int qk_heads,
                       float eps, void* stream) {
    if (!history || !qkv || !conv_w || !h || channels % S != 0 || qk_heads < 0 || qk_heads > channels / S) {
        std::fprintf(stderr, "fused_gdn_conv_l2: invalid arguments\n");
        std::exit(1);
    }
    sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(S / 32), cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) channels, S), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int tid = (int) it.get_local_id(0);
            const int c = (int) it.get_global_id(0);
            const float v0 = history[c * 3], v1 = history[c * 3 + 1], v2 = history[c * 3 + 2], x = qkv[c];
            const float sum = v0 * conv_w[c * 4] + v1 * conv_w[c * 4 + 1] + v2 * conv_w[c * 4 + 2] + x * conv_w[c * 4 + 3];
            history[c * 3] = v1;
            history[c * 3 + 1] = v2;
            history[c * 3 + 2] = x;
            float y = sum / (1.0f + sycl::exp(-sum));
            if ((int) it.get_group(0) < qk_heads) {   // uniform over the work-group
                float sq = y * y;
                const sycl::sub_group sg = it.get_sub_group();
                for (int o = 16; o > 0; o >>= 1) sq += sycl::permute_group_by_xor(sg, sq, o);
                if ((tid & 31) == 0) part[tid >> 5] = sq;
                sycl::group_barrier(it.get_group());
                const float ss = part[0] + part[1] + part[2] + part[3];
                y *= sycl::rsqrt(ss + eps);
            }
            h[c] = y;
        });
    });
    check("fused_gdn_conv_l2");
}

void fused_gdn_ab(const float* x, const uint16_t* w_alpha, const uint16_t* w_beta, const float* dt, const float* ssm_a,
                  float* gate, float* beta, int n_embd, int h_v, void* stream) {
    if (!x || !w_alpha || !w_beta || !dt || !ssm_a || !gate || !beta || n_embd % 8 != 0 || h_v <= 0) {
        std::fprintf(stderr, "fused_gdn_ab: invalid arguments\n");
        std::exit(1);
    }
    const int n = n_embd;
    sycl_runtime::queue_from_stream(stream).parallel_for(
        sycl::nd_range<1>((size_t) ((2 * h_v + 7) / 8) * 256, 256), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int tid = (int) it.get_local_id(0);
            const int row = (int) it.get_group(0) * 8 + (tid >> 5), lane = tid & 31;
            if (row >= 2 * h_v) return;   // whole sub-groups leave together
            const bool is_beta = row >= h_v;
            const int r = is_beta ? row - h_v : row;
            const uint32_t* w4 = reinterpret_cast<const uint32_t*>((is_beta ? w_beta : w_alpha) + (size_t) r * n);
            float acc = 0.0f;
            for (int j = lane; j < n / 8; j += 32) {
                const uint32_t* wv = w4 + 4 * j;
                const float* xa = x + j * 8;
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    acc = sycl::fma(sycl::bit_cast<float>(wv[e] << 16), xa[2 * e], acc);
                    acc = sycl::fma(sycl::bit_cast<float>(wv[e] & 0xffff0000u), xa[2 * e + 1], acc);
                }
            }
            const sycl::sub_group sg = it.get_sub_group();
            for (int o = 16; o > 0; o >>= 1) acc += sycl::permute_group_by_xor(sg, acc, o);
            if (lane != 0) return;
            if (is_beta) {
                beta[r] = 1.0f / (1.0f + sycl::exp(-acc));
            } else {
                const float v = acc + dt[r];
                const float sp = v > 20.0f ? v : sycl::log1p(sycl::exp(v));
                gate[r] = sp * ssm_a[r];
            }
        });
    check("fused_gdn_ab");
}

void fused_gdn_step_norm(float* state, const float* q, const float* k, const float* v, const float* gate,
                         const float* beta, const float* z, const float* gamma, float eps, float* y, int h_k, int h_v,
                         void* stream) {
    if (!state || !q || !k || !v || !gate || !beta || !z || !gamma || !y || h_k <= 0 || h_v <= 0 || h_v % h_k) {
        std::fprintf(stderr, "fused_gdn_step_norm: invalid arguments\n");
        std::exit(1);
    }
    sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> sk(sycl::range<1>(S), cgh), sq(sycl::range<1>(S), cgh);
        sycl::local_accessor<float, 1> red(sycl::range<1>(RG * S), cgh), wsum(sycl::range<1>(S * RG / 32), cgh);
        cgh.parallel_for(sycl::nd_range<1>((size_t) h_v * S * RG, S * RG), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const int head = (int) it.get_group(0);
            const int tid = (int) it.get_local_id(0);
            const int col = tid % S, rg = tid / S;
            const int qh = head % h_k;
            if (tid < S) { sk[tid] = k[qh * S + tid]; sq[tid] = q[qh * S + tid]; }
            float s[RPG];
            float* base = state + ((size_t) (rg * RPG) * h_v + head) * S + col;
            const size_t row_stride = (size_t) h_v * S;
#pragma unroll
            for (int r = 0; r < RPG; ++r) s[r] = base[r * row_stride];
            sycl::group_barrier(it.get_group());
            const float g = sycl::exp(gate[head]);
            float kv = 0.0f;
#pragma unroll
            for (int r = 0; r < RPG; ++r) kv = sycl::fma(s[r], sk[rg * RPG + r], kv);
            red[rg * S + col] = kv;
            sycl::group_barrier(it.get_group());
            const float kv_col = red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col];
            const float delta = (v[head * S + col] - g * kv_col) * beta[head];
            float o = 0.0f;
#pragma unroll
            for (int r = 0; r < RPG; ++r) {
                s[r] = sycl::fma(g, s[r], sk[rg * RPG + r] * delta);
                o = sycl::fma(s[r], sq[rg * RPG + r], o);
                base[r * row_stride] = s[r];
            }
            sycl::group_barrier(it.get_group());   // every work-item has read red[] for kv_col
            red[rg * S + col] = o;
            sycl::group_barrier(it.get_group());
            float oc = 0.0f, sq_part = 0.0f;
            if (rg == 0) {
                oc = (red[col] + red[S + col] + red[2 * S + col] + red[3 * S + col]) * sycl::rsqrt((float) S);
                sq_part = oc * oc;
            }
            // the RMS over the head's 128 outputs: the sub-groups of row group 0 are work-items 0..127
            const sycl::sub_group sg = it.get_sub_group();
            for (int o2 = 16; o2 > 0; o2 >>= 1) sq_part += sycl::permute_group_by_xor(sg, sq_part, o2);
            if ((tid & 31) == 0) wsum[tid >> 5] = sq_part;
            sycl::group_barrier(it.get_group());
            if (rg == 0) {
                const float ss = wsum[0] + wsum[1] + wsum[2] + wsum[3];
                const float scale = sycl::rsqrt(ss / (float) S + eps);
                const float zz = z[head * S + col];
                y[head * S + col] = oc * scale * gamma[col] * (1.0f / (1.0f + sycl::exp(-zz)));
            }
        });
    });
    check("fused_gdn_step_norm");
}

}  // namespace strata::kernels
