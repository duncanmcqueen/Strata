// SYCL port of cuda/gdn.cu. Each work item owns a state column;
// norms retain the CUDA 32-lane double reduction tree.
#include "strata/kernels/gdn.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include <stdexcept>

namespace strata::kernels {
namespace {
void finish(sycl::queue& q, void* stream) { if (!stream) q.wait_and_throw(); }
float sigmoid(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }
}
void gdn_step(float* state, const float* q, const float* k, const float* v,
              const float* gate, const float* beta, float* o, const GdnShapes& s, void* stream) {
    if (s.S <= 0 || s.h_k <= 0 || s.h_v <= 0) return;
    if (s.S > 128) throw std::invalid_argument("gdn_step: S exceeds 128");
    auto& queue = sycl_runtime::queue_from_stream(stream);
    const int S = s.S, hk = s.h_k, hv = s.h_v;
    const size_t tiles = (S + 31) / 32;
    queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 1> ks(128, cgh), qs(128, cgh), scalars(2, cgh);
        cgh.parallel_for(sycl::nd_range<1>(tiles * hv * 32, 32), [=](sycl::nd_item<1> it) {
            const int lane = it.get_local_id(0), h = it.get_group(0) / tiles;
            const int j = (it.get_group(0) % tiles) * 32 + lane;
            for (int i = lane; i < S; i += 32) {
                ks[i] = k[(h % hk) * S + i]; qs[i] = q[(h % hk) * S + i];
            }
            if (lane == 0) { scalars[0] = sycl::exp(gate[h]); scalars[1] = beta[h]; }
            it.barrier(sycl::access::fence_space::local_space);
            if (j >= S) return;
            float* col = state + size_t(h) * S + j;
            const size_t stride = size_t(hv) * S;
            float sk = 0;
            for (int i = 0; i < S; ++i) {
                float value = col[size_t(i) * stride] * scalars[0];
                col[size_t(i) * stride] = value; sk += value * ks[i];
            }
            const float d = (v[size_t(h) * S + j] - sk) * scalars[1];
            float dot = 0;
            for (int i = 0; i < S; ++i) {
                float value = col[size_t(i) * stride] + ks[i] * d;
                col[size_t(i) * stride] = value; dot += value * qs[i];
            }
            o[size_t(h) * S + j] = dot;
        });
    });
    finish(queue, stream);
}
void gdn_conv_step(float* cs, const float* x, const float* w, float* out,
                   int64_t channels, int64_t dc, void* stream) {
    if (channels <= 0 || dc < 1) return;
    auto& queue = sycl_runtime::queue_from_stream(stream);
    queue.parallel_for(sycl::range<1>(channels), [=](sycl::id<1> id) {
        size_t c = id[0];
        // A one-tap convolution has no history buffer.
        if (dc == 1) { out[c] = x[c] * w[c]; return; }
        float* st = cs + c * (dc - 1);
        float acc = 0;
        for (int i = 0; i < dc - 1; ++i) acc += st[i] * w[c * dc + i];
        out[c] = acc + x[c] * w[c * dc + dc - 1];
        for (int i = 0; i < dc - 2; ++i) st[i] = st[i + 1];
        st[dc - 2] = x[c];
    });
    finish(queue, stream);
}
void gdn_l2_norm(float* x, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    if (cols > 1024) throw std::invalid_argument("gdn_l2_norm: cols exceeds 1024");
    auto& queue = sycl_runtime::queue_from_stream(stream);
    queue.parallel_for(sycl::nd_range<1>(rows * 32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        float* p = x + it.get_group(0) * cols;
        int lane = it.get_local_id(0);
        double sum = 0;
        for (int i = lane; i < cols; i += 32) sum += double(p[i]) * double(p[i]);
        auto sg = it.get_sub_group();
        for (int off = 16; off; off >>= 1) sum += sycl::shift_group_left(sg, sum, off);
        sum = sycl::group_broadcast(sg, sum, 0);
        float inv = float(1.0 / sycl::sqrt(sum + double(eps)));
        for (int i = lane; i < cols; i += 32) p[i] *= inv;
    });
    finish(queue, stream);
}
void gdn_beta_gate(float* beta, int64_t hv, void* stream) {
    if (!beta || hv <= 0) return;
    auto& queue = sycl_runtime::queue_from_stream(stream);
    queue.parallel_for(sycl::range<1>(hv), [=](sycl::id<1> i) { beta[i[0]] = sigmoid(beta[i[0]]); });
    finish(queue, stream);
}
void gdn_out_norm(const float* o, const float* z, const float* norm, float* y,
                  int64_t hv, int64_t S, float eps, void* stream) {
    if (hv <= 0 || S <= 0) return;
    auto& queue = sycl_runtime::queue_from_stream(stream);
    queue.parallel_for(sycl::nd_range<1>(hv * 32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        size_t base = it.get_group(0) * S;
        int lane = it.get_local_id(0);
        double sum = 0;
        for (int i = lane; i < S; i += 32) sum += double(o[base+i]) * double(o[base+i]);
        auto sg = it.get_sub_group();
        for (int off = 16; off; off >>= 1) sum += sycl::shift_group_left(sg, sum, off);
        sum = sycl::group_broadcast(sg, sum, 0);
        float inv = float(1.0 / sycl::sqrt(sum / double(S) + double(eps)));
        for (int i = lane; i < S; i += 32) y[base+i] = o[base+i] * inv * norm[i] * sigmoid(z[base+i]);
    });
    finish(queue, stream);
}
} // namespace strata::kernels
