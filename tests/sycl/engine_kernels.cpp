// tests/sycl/engine_kernels.cpp - the engine-only kernels (no existing parity test calls them; the engine does):
// what each header promises, checked against an independent host oracle or, where the header promises bit
// equality with another kernel, bitwise.  Synthetic data, no model.
//
//   verify window   gdn_conv_l2_multi / gdn_conv_commit / gdn_ab_multi / gdn_step_norm_multi bitwise equal to
//                   T sequential fused_gdn_* calls (outputs, committed history and state); verify leaves state alone
//   router          native_router_top10 ids == a host softmax top-10 (lower index on ties), weights within 1e-6;
//                   the multi call bitwise equal to single calls
//   moe combine     bitwise equal to the header's rounding contract evaluated on the host (std::fma)
//   qsa score       bitwise equal to the scalar contract evaluated on the host; suppressed on invalid counts
//   flash attn      against FP64 attention, with and without a mask; an invalid step writes status 1 and NaN
//   native gdn      native_gdn_step against an FP64 recurrence
//   preprocess      native_gdn_* and native_qsa_* against FP64
//   flag wait       wait_flag_ge blocks its stream until a host thread sets the mapped flag
//   gpu_stamp       monotonic, and a 200 ms host sleep between stamps reads as 200 ms (within 25%), from the
//                   device clock or, without one (the A770), the host clock word
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/fused_gdn.hpp"
#include "strata/kernels/native_flash_attn.hpp"
#include "strata/kernels/native_gdn.hpp"
#include "strata/kernels/native_gdn_preprocess.hpp"
#include "strata/kernels/native_moe.hpp"
#include "strata/kernels/native_qsa.hpp"
#include "strata/kernels/native_qsa_score.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/kernels/verify_kernels.hpp"

#include <cuda_runtime.h>

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <numeric>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <vector>

namespace k = strata::kernels;

namespace {

int g_fail = 0;
void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) { std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e)); std::exit(2); }
}
void expect(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("  %-64s %s %s\n", what.c_str(), ok ? "ok  " : "FAIL", detail.c_str());
    if (!ok) ++g_fail;
}
template<class T> T* dev(const std::vector<T>& v) {
    T* p = nullptr;
    ck(cudaMalloc(&p, std::max<size_t>(1, v.size()) * sizeof(T)), "malloc");
    if (!v.empty()) ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}
template<class T> std::vector<T> host(const T* p, size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}
template<class T> bool same(const std::vector<T>& a, const std::vector<T>& b) {
    return a.size() == b.size() && std::memcmp(a.data(), b.data(), a.size() * sizeof(T)) == 0;
}
double rel(const std::vector<float>& got, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) return INFINITY;
        num += std::fabs(got[i] - ref[i]);
        den += std::fabs(ref[i]);
    }
    return num / (den + 1e-300);
}
std::string fmt(double v) { char b[32]; std::snprintf(b, sizeof b, "rel %.2e", v); return b; }
uint16_t bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u = (u + ((u >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t) (u >> 16);
}

std::mt19937 rng(5);
std::vector<float> randv(size_t n, float s = 1.0f) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> v(n);
    for (auto& x : v) x = nd(rng) * s;
    return v;
}

}  // namespace

int main() {
    cudaStream_t s;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");
    constexpr int S = 128;

    // ---------------------------------------------------------------- verify window vs single tokens
    std::printf("-- verify-window GDN kernels vs T sequential single-token kernels\n");
    {
        const int h_k = 4, h_v = 8, qk = S * h_k, C = 2 * qk + S * h_v, T = 6, n_embd = 512;
        const std::vector<float> hist0 = randv((size_t) C * 3), qkv = randv((size_t) T * C), conv_w = randv((size_t) C * 4, 0.5f);
        float *d_hist_seq = dev(hist0), *d_hist_mul = dev(hist0), *d_qkv = dev(qkv), *d_w = dev(conv_w);
        float *h_seq = dev(std::vector<float>((size_t) T * C)), *h_mul = dev(std::vector<float>((size_t) T * C));
        for (int t = 0; t < T; ++t)
            k::fused_gdn_conv_l2(d_hist_seq, d_qkv + (size_t) t * C, d_w, h_seq + (size_t) t * C, C, 2 * h_k, 1e-6f, s);
        k::gdn_conv_l2_multi(d_hist_mul, d_qkv, d_w, h_mul, C, 2 * h_k, 1e-6f, T, s);
        ck(cudaStreamSynchronize(s), "sync");
        expect(same(host(h_seq, (size_t) T * C), host(h_mul, (size_t) T * C)), "gdn_conv_l2_multi == T fused_gdn_conv_l2, bitwise");
        expect(same(host(d_hist_mul, (size_t) C * 3), hist0), "gdn_conv_l2_multi leaves the history alone");
        // commit 4 of the 6: the history the sequential path had after 4 tokens
        float* d_hist_4 = dev(hist0);
        float* scratch = dev(std::vector<float>((size_t) C));
        for (int t = 0; t < 4; ++t) k::fused_gdn_conv_l2(d_hist_4, d_qkv + (size_t) t * C, d_w, scratch, C, 2 * h_k, 1e-6f, s);
        int32_t* d_keep = dev(std::vector<int32_t>{4});
        k::gdn_conv_commit(d_hist_mul, d_qkv, C, d_keep, s);
        ck(cudaStreamSynchronize(s), "sync");
        expect(same(host(d_hist_mul, (size_t) C * 3), host(d_hist_4, (size_t) C * 3)), "gdn_conv_commit(4) == history after 4 tokens, bitwise");

        // alpha / beta
        std::vector<float> x = randv((size_t) T * n_embd), dt = randv(h_v), ssm_a = randv(h_v);
        std::vector<uint16_t> wa((size_t) h_v * n_embd), wb((size_t) h_v * n_embd);
        for (auto& w : wa) w = bf16(randv(1, 0.05f)[0]);
        for (auto& w : wb) w = bf16(randv(1, 0.05f)[0]);
        float *d_x = dev(x), *d_dt = dev(dt), *d_a = dev(ssm_a);
        uint16_t *d_wa = dev(wa), *d_wb = dev(wb);
        float *g_seq = dev(std::vector<float>((size_t) T * h_v)), *b_seq = dev(std::vector<float>((size_t) T * h_v));
        float *g_mul = dev(std::vector<float>((size_t) T * h_v)), *b_mul = dev(std::vector<float>((size_t) T * h_v));
        for (int t = 0; t < T; ++t)
            k::fused_gdn_ab(d_x + (size_t) t * n_embd, d_wa, d_wb, d_dt, d_a, g_seq + t * h_v, b_seq + t * h_v, n_embd, h_v, s);
        k::gdn_ab_multi(d_x, d_wa, d_wb, d_dt, d_a, g_mul, b_mul, n_embd, h_v, T, s);
        ck(cudaStreamSynchronize(s), "sync");
        expect(same(host(g_seq, (size_t) T * h_v), host(g_mul, (size_t) T * h_v)) &&
                   same(host(b_seq, (size_t) T * h_v), host(b_mul, (size_t) T * h_v)),
               "gdn_ab_multi == T fused_gdn_ab (gate and beta), bitwise");

        // the recurrence + norm
        const std::vector<float> state0 = randv((size_t) S * h_v * S, 0.1f), z = randv((size_t) T * S * h_v),
                                 gamma = randv(S), gate = randv((size_t) T * h_v, 0.3f);
        std::vector<float> beta((size_t) T * h_v);
        for (auto& b : beta) b = 0.5f + 0.4f * std::tanh(randv(1)[0]);
        std::vector<float> hb = randv((size_t) T * C, 0.2f);
        float *d_h = dev(hb), *d_gate = dev(gate), *d_beta = dev(beta), *d_z = dev(z), *d_gamma = dev(gamma);
        float *st_seq = dev(state0), *st_mul = dev(state0), *st_keep = dev(state0);
        float *y_seq = dev(std::vector<float>((size_t) T * S * h_v)), *y_mul = dev(std::vector<float>((size_t) T * S * h_v));
        std::vector<float> state_after3;
        for (int t = 0; t < T; ++t) {
            const float* ht = d_h + (size_t) t * C;
            k::fused_gdn_step_norm(st_seq, ht, ht + qk, ht + 2 * qk, d_gate + t * h_v, d_beta + t * h_v, d_z + (size_t) t * S * h_v,
                                   d_gamma, 1e-6f, y_seq + (size_t) t * S * h_v, h_k, h_v, s);
            if (t == 2) { ck(cudaStreamSynchronize(s), "sync"); state_after3 = host(st_seq, state0.size()); }
        }
        k::gdn_step_norm_multi(st_mul, d_h, C, d_gate, d_beta, d_z, d_gamma, 1e-6f, y_mul, h_k, h_v, T, nullptr, s);
        int32_t* d_keep3 = dev(std::vector<int32_t>{3});
        float* y_scratch = dev(std::vector<float>((size_t) T * S * h_v));
        k::gdn_step_norm_multi(st_keep, d_h, C, d_gate, d_beta, d_z, d_gamma, 1e-6f, y_scratch, h_k, h_v, T, d_keep3, s);
        ck(cudaStreamSynchronize(s), "sync");
        expect(same(host(y_seq, (size_t) T * S * h_v), host(y_mul, (size_t) T * S * h_v)),
               "gdn_step_norm_multi (verify) outputs == T fused_gdn_step_norm, bitwise");
        expect(same(host(st_mul, state0.size()), state0), "gdn_step_norm_multi (verify) leaves the state alone");
        expect(same(host(st_keep, state0.size()), state_after3), "gdn_step_norm_multi (commit 3) == state after 3 tokens, bitwise");
    }

    // ---------------------------------------------------------------- router
    std::printf("-- native router top-10\n");
    {
        const int n_tok = 5;
        std::vector<float> logits = randv((size_t) n_tok * 512, 2.0f);
        logits[7] = logits[300] = 9.0f;   // an exact tie: the lower index must win the earlier rank
        float* d_l = dev(logits);
        int32_t *ids1 = dev(std::vector<int32_t>((size_t) n_tok * 10)), *idsm = dev(std::vector<int32_t>((size_t) n_tok * 10));
        float *w1 = dev(std::vector<float>((size_t) n_tok * 10)), *wm = dev(std::vector<float>((size_t) n_tok * 10));
        for (int t = 0; t < n_tok; ++t) k::native_router_top10(d_l + t * 512, ids1 + t * 10, w1 + t * 10, s);
        k::native_router_top10_multi(d_l, idsm, wm, n_tok, s);
        ck(cudaStreamSynchronize(s), "sync");
        const auto gi = host(ids1, (size_t) n_tok * 10);
        const auto gw = host(w1, (size_t) n_tok * 10);
        bool ids_ok = true;
        std::vector<double> ref_w;
        std::vector<float> got_w;
        for (int t = 0; t < n_tok; ++t) {
            const float* l = &logits[(size_t) t * 512];
            const double m = *std::max_element(l, l + 512);
            std::vector<double> p(512);
            double z = 0;
            for (int e = 0; e < 512; ++e) z += (p[e] = std::exp(l[e] - m));
            for (auto& v : p) v /= z;
            std::vector<int> order(512);
            std::iota(order.begin(), order.end(), 0);
            std::stable_sort(order.begin(), order.end(), [&](int a, int b) { return p[a] > p[b]; });
            double sel = 0;
            for (int r = 0; r < 10; ++r) sel += p[order[r]];
            for (int r = 0; r < 10; ++r) {
                ids_ok = ids_ok && gi[t * 10 + r] == order[r];
                ref_w.push_back(p[order[r]] / sel);
                got_w.push_back(gw[t * 10 + r]);
            }
        }
        expect(ids_ok, "ids == host softmax top-10, lower index first on a tie");
        const double e = rel(got_w, ref_w);
        expect(e < 1e-6, "weights vs FP64", fmt(e));
        expect(same(gi, host(idsm, (size_t) n_tok * 10)) && same(gw, host(wm, (size_t) n_tok * 10)), "multi == single calls, bitwise");
    }

    // ---------------------------------------------------------------- moe combine
    std::printf("-- native MoE combine\n");
    {
        const int N = 2560, K = 10, n_tok = 3;
        const std::vector<float> parts = randv((size_t) n_tok * K * N), wts = randv((size_t) n_tok * K), shared = randv((size_t) n_tok * N);
        float *dp = dev(parts), *dw = dev(wts), *ds = dev(shared);
        float *o1 = dev(std::vector<float>((size_t) n_tok * N)), *om = dev(std::vector<float>((size_t) n_tok * N));
        for (int t = 0; t < n_tok; ++t)
            k::native_moe_combine(dp + (size_t) t * K * N, dw + t * K, ds + (size_t) t * N, o1 + (size_t) t * N, N, K, s);
        k::native_moe_combine_multi(dp, dw, ds, om, N, K, n_tok, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> want((size_t) n_tok * N);
        for (int t = 0; t < n_tok; ++t)
            for (int c = 0; c < N; ++c) {
                const float* p = &parts[(size_t) t * K * N];
                float sum = p[c] * wts[t * K];
                for (int e = 1; e < K; ++e) sum = std::fma(p[(size_t) e * N + c], wts[t * K + e], sum);
                want[(size_t) t * N + c] = sum + shared[(size_t) t * N + c];
            }
        expect(same(host(o1, want.size()), want), "== the rounding contract on the host (fma chain), bitwise");
        expect(same(host(om, want.size()), want), "multi == the same, bitwise");
    }

    // ---------------------------------------------------------------- qsa score
    std::printf("-- native QSA score (scalar FP32 contract)\n");
    {
        const k::QsaShapes sh = k::qsa_real_shapes();
        const int max_cells = 4096, max_blocks = max_cells / 4 + 1, n = 1027, full = n / 4;
        const std::vector<float> pooled = randv((size_t) max_blocks * 128, 0.2f), query = randv(4 * 128, 0.2f), bias = randv(max_blocks);
        const std::vector<int32_t> step = {n - 1, n, full, std::min(n, 2051)};
        float *dp = dev(pooled), *dq = dev(query), *db = dev(bias), *dc = dev(std::vector<float>(max_cells, -7.0f));
        int32_t* dstep = dev(step);
        k::native_qsa_score(dp, dq, db, sh, dstep, max_blocks, max_cells, dc, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> want(max_cells, -7.0f);
        for (int row = 0; row <= full; ++row) {
            float h[4];
            for (int j = 0; j < 4; ++j) {
                float dot = 0.0f;
                for (int d = 0; d < 128; ++d) dot = std::fma(pooled[(size_t) row * 128 + d], query[j * 128 + d], dot);
                h[j] = dot > 0.0f ? dot : 0.0f;
            }
            float sum = 0.0f + h[0];
            sum = sum + h[1];
            sum = sum + h[2];
            sum = sum + h[3];
            sum = sum + bias[row];
            sum = sum + (row == full && n % 4 ? 1e9f : 0.0f);
            sum = sum + 0.0f;
            for (int i = row * 4; i < n && i < (row + 1) * 4; ++i) want[i] = sum;
        }
        expect(same(host(dc, max_cells), want), "scores == the scalar contract on the host, bitwise; padding untouched");
        const std::vector<int32_t> bad = {n - 1, n, full + 1, std::min(n, 2051)};
        ck(cudaMemcpy(dstep, bad.data(), 16, cudaMemcpyHostToDevice), "bad step");
        ck(cudaMemset(dc, 0, (size_t) max_cells * 4), "zero");
        k::native_qsa_score(dp, dq, db, sh, dstep, max_blocks, max_cells, dc, s);
        ck(cudaStreamSynchronize(s), "sync");
        expect(same(host(dc, max_cells), std::vector<float>(max_cells, 0.0f)), "an inconsistent step record writes nothing");
    }

    // ---------------------------------------------------------------- flash attention
    std::printf("-- native FlashAttention (short context) vs FP64\n");
    {
        const k::QsaShapes sh = k::qsa_real_shapes();
        const int cap = 256;
        const std::vector<float> q = randv(24 * 256);
        std::vector<uint16_t> kv_k((size_t) cap * 2 * 256), kv_v((size_t) cap * 2 * 256), mask(256);
        for (auto& x : kv_k) x = k::f16_from_f32(randv(1)[0]);
        for (auto& x : kv_v) x = k::f16_from_f32(randv(1)[0]);
        float *dq = dev(q), *dout = dev(std::vector<float>(24 * 256));
        uint16_t *dk = dev(kv_k), *dv = dev(kv_v);
        int32_t *dstat = dev(std::vector<int32_t>{-1}), *dstep = dev(std::vector<int32_t>(4));
        for (int with_mask = 0; with_mask < 2; ++with_mask)
            for (int width : {1, 37, 200, 256}) {
                for (int c = 0; c < 256; ++c) mask[c] = k::f16_from_f32(c % 5 == 1 ? -INFINITY : (c % 3) * 0.25f);
                if (with_mask) mask[0] = k::f16_from_f32(0.0f);   // one valid key stays unmasked
                uint16_t* dm = with_mask ? dev(mask) : nullptr;
                const std::vector<int32_t> step = {width - 1, width, width / 4, width};
                ck(cudaMemcpy(dstep, step.data(), 16, cudaMemcpyHostToDevice), "step");
                k::native_flash_attn_short_step(dq, dk, dv, dstep, cap, 256, sh, dout, dstat, dm, s);
                ck(cudaStreamSynchronize(s), "sync");
                std::vector<double> ref(24 * 256, 0.0);
                for (int h = 0; h < 24; ++h) {
                    const int kvh = h / 12;
                    std::vector<double> sc(width);
                    double m = -INFINITY;
                    for (int c = 0; c < width; ++c) {
                        double dot = 0;
                        for (int d = 0; d < 256; ++d) dot += (double) q[h * 256 + d] * k::f32_from_f16(kv_k[((size_t) c * 2 + kvh) * 256 + d]);
                        sc[c] = dot / 16.0 + (with_mask ? (double) k::f32_from_f16(mask[c]) : 0.0);
                        m = std::max(m, sc[c]);
                    }
                    double z = 0;
                    for (int c = 0; c < width; ++c) z += (sc[c] = std::exp(sc[c] - m));
                    for (int c = 0; c < width; ++c)
                        for (int d = 0; d < 256; ++d) ref[h * 256 + d] += sc[c] / z * k::f32_from_f16(kv_v[((size_t) c * 2 + kvh) * 256 + d]);
                }
                const double e = rel(host(dout, 24 * 256), ref);
                expect(e < 2e-6 && host(dstat, 1)[0] == 0,
                       "width " + std::to_string(width) + (with_mask ? ", masked" : "") + ": vs FP64, status 0", fmt(e));
                if (dm) cudaFree(dm);
            }
        const std::vector<int32_t> bad = {9, 10, 3, 10};   // n_bid must be 2
        ck(cudaMemcpy(dstep, bad.data(), 16, cudaMemcpyHostToDevice), "step");
        k::native_flash_attn_short_step(dq, dk, dv, dstep, cap, 256, sh, dout, dstat, nullptr, s);
        ck(cudaStreamSynchronize(s), "sync");
        const auto out = host(dout, 24 * 256);
        expect(host(dstat, 1)[0] == k::kNativeFlashAttnUnsupportedStep &&
                   std::all_of(out.begin(), out.end(), [](float v) { return std::isnan(v); }),
               "an unsupported step: status 1 and NaN output");
    }

    // ---------------------------------------------------------------- native gdn step
    std::printf("-- native GDN step vs FP64\n");
    {
        const int h_k = 4, h_v = 8;
        const k::GdnShapes sh{S, h_k, h_v};
        const std::vector<float> st = randv((size_t) S * h_v * S, 0.1f), q = randv(h_k * S, 0.1f), kk = randv(h_k * S, 0.1f),
                                 v = randv(h_v * S), gate = randv(h_v, 0.3f);
        std::vector<float> beta(h_v);
        for (auto& b : beta) b = 0.5f;
        float *ds = dev(st), *dq = dev(q), *dk = dev(kk), *dv = dev(v), *dg = dev(gate), *db = dev(beta), *dout = dev(std::vector<float>(h_v * S));
        k::native_gdn_step(ds, dq, dk, dv, dg, db, dout, sh, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<double> ref_s(st.begin(), st.end()), ref_o(h_v * S);
        for (int h = 0; h < h_v; ++h) {
            const int qh = h % h_k;
            const double g = std::exp((double) gate[h]);
            for (int j = 0; j < S; ++j) {
                double kv = 0;
                for (int i = 0; i < S; ++i) kv += ref_s[((size_t) i * h_v + h) * S + j] * kk[qh * S + i];
                const double delta = (v[h * S + j] - g * kv) * beta[h];
                double o = 0;
                for (int i = 0; i < S; ++i) {
                    double& x = ref_s[((size_t) i * h_v + h) * S + j];
                    x = g * x + kk[qh * S + i] * delta;
                    o += x * q[qh * S + i];
                }
                ref_o[h * S + j] = o / std::sqrt((double) S);
            }
        }
        const double eo = rel(host(dout, h_v * S), ref_o), es = rel(host(ds, st.size()), ref_s);
        expect(eo < 1e-5 && es < 1e-6, "output and state vs FP64", fmt(eo) + ", state " + fmt(es));
    }

    // ---------------------------------------------------------------- preprocessing and QSA norm/gate
    std::printf("-- native GDN preprocessing and QSA norm/gate vs FP64\n");
    {
        const int heads = 16;
        std::vector<float> x = randv((size_t) heads * S);
        float* dx = dev(x);
        k::native_gdn_l2_norm(dx, heads, S, 1e-6f, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<double> ref(x.size());
        for (int r = 0; r < heads; ++r) {
            double ss = 0;
            for (int c = 0; c < S; ++c) ss += (double) x[r * S + c] * x[r * S + c];
            const double sc = 1.0 / std::sqrt(ss / S + 1e-6 / S) / std::sqrt((double) S);
            for (int c = 0; c < S; ++c) ref[r * S + c] = x[r * S + c] * sc;
        }
        double e = rel(host(dx, x.size()), ref);
        expect(e < 1e-6, "l2_norm", fmt(e));

        const std::vector<float> o = randv((size_t) heads * S), z = randv((size_t) heads * S), gamma = randv(S);
        float *dO = dev(o), *dz = dev(z), *dgm = dev(gamma), *dd = dev(std::vector<float>((size_t) heads * S));
        k::native_gdn_out_norm(dO, dz, dgm, dd, heads, S, 1e-6f, s);
        ck(cudaStreamSynchronize(s), "sync");
        for (int r = 0; r < heads; ++r) {
            double ss = 0;
            for (int c = 0; c < S; ++c) ss += (double) o[r * S + c] * o[r * S + c];
            const double sc = 1.0 / std::sqrt(ss / S + 1e-6);
            for (int c = 0; c < S; ++c) ref[r * S + c] = o[r * S + c] * sc * gamma[c] / (1.0 + std::exp(-(double) z[r * S + c]));
        }
        e = rel(host(dd, ref.size()), ref);
        expect(e < 1e-6, "out_norm", fmt(e));

        const std::vector<float> alpha = randv(heads, 4.0f), dtv = randv(heads), ssm = randv(heads);
        float *da = dev(alpha), *ddt = dev(dtv), *dssm = dev(ssm), *dgate = dev(std::vector<float>(heads)), *dbeta = dev(alpha);
        k::native_gdn_gate(da, ddt, dssm, dgate, heads, s);
        k::native_gdn_beta_gate(dbeta, heads, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<double> rg(heads), rb(heads);
        for (int h = 0; h < heads; ++h) {
            const double v = (double) (alpha[h] + dtv[h]);
            rg[h] = (v > 20 ? v : std::log1p(std::exp(v))) * ssm[h];
            rb[h] = 1.0 / (1.0 + std::exp(-(double) alpha[h]));
        }
        const double eg = rel(host(dgate, heads), rg), eb = rel(host(dbeta, heads), rb);
        expect(eg < 1e-6 && eb < 1e-6, "gate softplus and beta sigmoid", fmt(eg) + ", " + fmt(eb));

        const int C = 1024;
        const std::vector<float> hist = randv((size_t) C * 3), in = randv(C), w = randv((size_t) C * 4, 0.5f);
        float *dh = dev(hist), *din = dev(in), *dw = dev(w), *draw = dev(std::vector<float>(C)), *dsilu = dev(std::vector<float>(C));
        k::native_gdn_conv_silu(dh, din, dw, draw, dsilu, C, 4, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<double> rr(C), rs(C);
        for (int c = 0; c < C; ++c) {
            const double sum = (double) hist[c * 3] * w[c * 4] + (double) hist[c * 3 + 1] * w[c * 4 + 1] +
                               (double) hist[c * 3 + 2] * w[c * 4 + 2] + (double) in[c] * w[c * 4 + 3];
            rr[c] = sum;
            rs[c] = sum / (1.0 + std::exp(-sum));
        }
        const auto nh = host(dh, (size_t) C * 3);
        bool shifted = true;
        for (int c = 0; c < C; ++c)
            shifted = shifted && nh[c * 3] == hist[c * 3 + 1] && nh[c * 3 + 1] == hist[c * 3 + 2] && nh[c * 3 + 2] == in[c];
        const double er = rel(host(draw, C), rr), es = rel(host(dsilu, C), rs);
        expect(er < 1e-6 && es < 1e-6 && shifted, "conv_silu (raw, SiLU, history shift)", fmt(er) + ", " + fmt(es));

        const int rows = 6, cols = 256;
        const std::vector<float> xi = randv((size_t) rows * cols), gm = randv(cols);
        float *dxi = dev(xi), *dgm2 = dev(gm);
        k::native_qsa_rms_norm_weighted(dxi, dgm2, dxi, cols, rows, 1e-6f, s);   // the exact in-place call
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<double> rn(xi.size());
        for (int r = 0; r < rows; ++r) {
            double ss = 0;
            for (int c = 0; c < cols; ++c) ss += (double) xi[r * cols + c] * xi[r * cols + c];
            const double sc = 1.0 / std::sqrt(ss / cols + 1e-6);
            for (int c = 0; c < cols; ++c) rn[r * cols + c] = xi[r * cols + c] * sc * gm[c];
        }
        e = rel(host(dxi, xi.size()), rn);
        expect(e < 1e-6, "qsa rms_norm, in place", fmt(e));

        const int nh2 = 24, hd = 256;
        const std::vector<float> at = randv((size_t) nh2 * hd), qf = randv((size_t) nh2 * hd * 2);
        float *dat = dev(at), *dqf = dev(qf), *dgo = dev(std::vector<float>((size_t) nh2 * hd));
        k::native_qsa_gate_apply(dat, dqf, dgo, nh2, hd, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<double> rgo(at.size());
        for (int h = 0; h < nh2; ++h)
            for (int c = 0; c < hd; ++c)
                rgo[h * hd + c] = at[h * hd + c] / (1.0 + std::exp(-(double) qf[h * 2 * hd + hd + c]));
        e = rel(host(dgo, rgo.size()), rgo);
        expect(e < 1e-6, "qsa gate_apply", fmt(e));
    }

    // ---------------------------------------------------------------- the mapped flag wait
    std::printf("-- wait_flag_ge on a mapped host flag\n");
    {
        uint32_t* flag = nullptr;
        ck(cudaHostAlloc((void**) &flag, 64, cudaHostAllocMapped), "host alloc");
        *(volatile uint32_t*) flag = 0;
        uint32_t* dflag = nullptr;
        ck(cudaHostGetDevicePointer((void**) &dflag, flag, 0), "device pointer");
        int32_t* d_done = dev(std::vector<int32_t>{0});
        k::wait_flag_ge(dflag, 3, s);
        ck(cudaMemsetAsync(d_done, 1, 1, s), "after");
        std::this_thread::sleep_for(std::chrono::milliseconds(100));
        const bool held = cudaStreamQuery(s) == cudaErrorNotReady;
        *(volatile uint32_t*) flag = 2;   // below the value: still held
        std::this_thread::sleep_for(std::chrono::milliseconds(50));
        const bool held2 = cudaStreamQuery(s) == cudaErrorNotReady;
        const auto t0 = std::chrono::steady_clock::now();
        *(volatile uint32_t*) flag = 3;
        ck(cudaStreamSynchronize(s), "sync");
        const double ms = std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - t0).count();
        expect(held && held2 && host(d_done, 1)[0] != 0, "held below the value, released at it",
               "released in " + std::to_string(ms).substr(0, 6) + " ms");
        cudaFreeHost(flag);
    }

    // ---------------------------------------------------------------- gpu_stamp
    std::printf("-- gpu_stamp\n");
    {
        unsigned long long* buf = nullptr;
        ck(cudaMalloc(&buf, 16), "stamp buf");
        k::gpu_stamp(buf, 0, s);   // a device clock, or the host clock word (the A770)
        ck(cudaStreamSynchronize(s), "sync");
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
        k::gpu_stamp(buf, 1, s);
        ck(cudaStreamSynchronize(s), "sync");
        const auto st = host(buf, 2);
        const double dms = (double) (st[1] - st[0]) / 1e6;
        expect(st[1] > st[0] && dms > 150.0 && dms < 250.0, "monotonic, 200 ms sleep reads as 200 ms (+-25%)",
               std::to_string(dms).substr(0, 7) + " ms");
    }

    std::printf("engine_kernels: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
