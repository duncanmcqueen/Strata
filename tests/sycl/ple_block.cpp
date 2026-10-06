// tests/sycl/ple_block.cpp - the PLE block against an independent FP64 transcription of include/strata/kernels/
// ple.hpp, with synthetic weights (no model).  The original ple_parity needs captured ggml fixtures
// (bench/micro/ple_in.bin, ple_out.bin) and the model's GGUF; this test does not replace it, it checks what can be
// checked without them:
//
//   (1) every stage (key, value, gate, gated, normalized, conv, result) of ple_block against FP64, for the legacy
//       and the native post-ops, with a BF16 key, a native Q8_0 key, the BF16 and the native value projection;
//   (2) native_ple_postops_batch over T tokens BITWISE equal to T native_ple_postops calls with
//       ple_history_advance after each (its header's contract), results and history, for T = 1, 5, 9 and 13;
//   (3) a captured ple_block replayed with changed inputs equals the direct call, bitwise;
//   (4) ple_history_advance against a host shift, and the overlap refusals.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/ple.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

namespace k = strata::kernels;

namespace {

constexpr int N = k::NG_N_EMBD, H = k::NG_HC, D = k::NG_HC_DIM, HIST = k::NG_HIST, KERN = k::PLE_CONV_KERNEL;
int g_fail = 0;

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(2);
    }
}
void expect(bool ok, const std::string& what, const std::string& detail = "") {
    std::printf("  %-66s %s %s\n", what.c_str(), ok ? "ok  " : "FAIL", detail.c_str());
    if (!ok) ++g_fail;
}
template<class T> T* dev(const std::vector<T>& v) {
    T* p = nullptr;
    ck(cudaMalloc(&p, v.size() * sizeof(T)), "malloc");
    ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice), "upload");
    return p;
}
template<class T> T* dev_zero(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T)), "malloc");
    ck(cudaMemset(p, 0, n * sizeof(T)), "zero");
    return p;
}
template<class T> std::vector<T> host(const T* p, size_t n) {
    std::vector<T> v(n);
    ck(cudaMemcpy(v.data(), p, n * sizeof(T), cudaMemcpyDeviceToHost), "download");
    return v;
}
uint16_t bf16(float f) {
    uint32_t u;
    std::memcpy(&u, &f, 4);
    u = (u + ((u >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t) (u >> 16);
}
double from_bf16(uint16_t b) {
    const uint32_t u = (uint32_t) b << 16;
    float f;
    std::memcpy(&f, &u, 4);
    return f;
}
double rel_l1(const std::vector<float>& got, const std::vector<double>& ref) {
    double num = 0, den = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (!std::isfinite(got[i])) return INFINITY;
        num += std::fabs(got[i] - ref[i]);
        den += std::fabs(ref[i]);
    }
    return num / (den + 1e-300);
}

// ---- the FP64 transcription of ple.hpp
std::vector<double> gnorm(const std::vector<double>& x, const std::vector<float>& w) {
    std::vector<double> y(D);
    for (int c = 0; c < H; ++c) {
        double ss = 0;
        for (int d = 0; d < N; ++d) ss += x[c * N + d] * x[c * N + d];
        const double scale = 1.0 / std::sqrt(ss / N + (double) k::NG_RMS_EPS);
        for (int d = 0; d < N; ++d) y[c * N + d] = x[c * N + d] * scale * w[c * N + d];
    }
    return y;
}
struct Ref { std::vector<double> key, value, gate, gated, normalized, conv, result; };
Ref oracle(const std::vector<double>& key_proj, const std::vector<double>& value, const std::vector<float>& hidden,
           const std::vector<float>& hist, const std::vector<float>& nk, const std::vector<float>& nq,
           const std::vector<float>& nc, const std::vector<uint16_t>& taps) {
    Ref r;
    r.key = gnorm(key_proj, nk);
    const std::vector<double> query = gnorm(std::vector<double>(hidden.begin(), hidden.end()), nq);
    r.value = value;
    r.gate.resize(H);
    for (int c = 0; c < H; ++c) {
        double s = 0;
        for (int d = 0; d < N; ++d) s += r.key[c * N + d] * query[c * N + d];
        s /= std::sqrt((double) N);
        const double g = (s > 0 ? 1 : s < 0 ? -1 : 0) * std::sqrt(std::max(std::fabs(s), 1e-6));
        r.gate[c] = 1.0 / (1.0 + std::exp(-g));
    }
    r.gated.resize(D);
    for (int i = 0; i < D; ++i) r.gated[i] = value[i % N] * r.gate[i / N];
    r.normalized = gnorm(r.gated, nc);
    r.conv.resize(D);
    r.result.resize(D);
    for (int c = 0; c < D; ++c) {
        double acc = 0;
        for (int t = 0; t < KERN; ++t) {
            const int row = HIST - (KERN - 1 - t) * k::NGRAM_SIZE;
            const double v = row == HIST ? r.normalized[c] : hist[(size_t) row + (size_t) HIST * c];
            acc += k::f32_from_f16(taps[t + KERN * c]) * v;
        }
        r.conv[c] = acc / (1.0 + std::exp(-acc));
        r.result[c] = hidden[c] + r.gated[c] + r.conv[c];
    }
    return r;
}

// Q8_0 blocks of `rows` x `cols` (ggml's quantize_row_q8_0) and their dequantized values
void q8_0(const std::vector<float>& w, int rows, int cols, std::vector<uint8_t>& blocks, std::vector<double>& deq) {
    blocks.assign((size_t) rows * cols / 32 * 34, 0);
    deq.assign((size_t) rows * cols, 0);
    for (size_t b = 0; b < (size_t) rows * cols / 32; ++b) {
        float amax = 0;
        for (int j = 0; j < 32; ++j) amax = std::max(amax, std::fabs(w[b * 32 + j]));
        const float d = amax / 127.0f, id = d ? 1.0f / d : 0.0f;
        const uint16_t dh = k::f16_from_f32(d);
        std::memcpy(&blocks[b * 34], &dh, 2);
        for (int j = 0; j < 32; ++j) {
            const int8_t q = (int8_t) std::lround(w[b * 32 + j] * id);
            blocks[b * 34 + 2 + j] = (uint8_t) q;
            deq[b * 32 + j] = (double) k::f32_from_f16(dh) * q;
        }
    }
}

}  // namespace

int main() {
    std::mt19937 rng(11);
    std::normal_distribution<float> nd(0.f, 1.f);
    std::uniform_real_distribution<float> ud(0.5f, 1.5f);
    auto randv = [&](size_t n, float s) { std::vector<float> v(n); for (auto& x : v) x = nd(rng) * s; return v; };

    // weights (synthetic, model-shaped)
    const std::vector<float> wk = randv((size_t) D * N, 0.02f), wv = randv((size_t) N * N, 0.02f);
    std::vector<uint16_t> wk16(wk.size()), wv16(wv.size());
    for (size_t i = 0; i < wk.size(); ++i) wk16[i] = bf16(wk[i]);
    for (size_t i = 0; i < wv.size(); ++i) wv16[i] = bf16(wv[i]);
    std::vector<uint8_t> wk_q8;
    std::vector<double> wk_q8_deq;
    q8_0(wk, D, N, wk_q8, wk_q8_deq);
    std::vector<float> nk(D), nq(D), nc(D);
    for (int i = 0; i < D; ++i) { nk[i] = ud(rng); nq[i] = ud(rng); nc[i] = ud(rng); }
    std::vector<uint16_t> taps((size_t) KERN * D);
    for (auto& t : taps) t = k::f16_from_f32(nd(rng) * 0.5f);

    uint16_t* d_wk16 = dev(wk16);
    uint16_t* d_wv16 = dev(wv16);
    uint8_t* d_wk_q8 = dev(wk_q8);
    float *d_nk = dev(nk), *d_nq = dev(nq), *d_nc = dev(nc);
    uint16_t* d_taps = dev(taps);
    void* d_q81 = nullptr;
    ck(cudaMalloc(&d_q81, k::native_q8_1_bytes(N)), "q8_1");
    void* d_scratch = nullptr;
    ck(cudaMalloc(&d_scratch, k::ple_block_scratch_bytes()), "scratch");
    cudaStream_t s;
    ck(cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking), "stream");

    auto weights = [&](bool native_key) {
        k::PleWeights w;
        w.value_bf16 = d_wv16;
        w.norm_key = d_nk; w.norm_query = d_nq; w.norm_conv = d_nc; w.conv1d_f16 = d_taps;
        if (native_key) { w.key_native_data = d_wk_q8; w.key_native_type = 8; w.key_native_q8_1 = d_q81; }
        else w.key_bf16 = d_wk16;
        return w;
    };

    // ---- (1) every stage against FP64
    std::printf("-- ple_block stages vs an FP64 transcription of ple.hpp\n");
    for (int native_key = 0; native_key < 2; ++native_key)
        for (int native_value = 0; native_value < 2; ++native_value)
            for (int native_post = 0; native_post < 2; ++native_post) {
                const std::vector<float> emb = randv(N, 1.0f), hidden = randv(D, 1.0f), hist = randv((size_t) HIST * D, 1.0f);
                std::vector<double> key_proj(D, 0.0), value(N, 0.0);
                for (int o = 0; o < D; ++o)
                    for (int i = 0; i < N; ++i)
                        key_proj[o] += (native_key ? wk_q8_deq[(size_t) o * N + i] : from_bf16(wk16[(size_t) o * N + i])) * emb[i];
                for (int o = 0; o < N; ++o)
                    for (int i = 0; i < N; ++i)   // the legacy value path rounds the activation to BF16 too
                        value[o] += from_bf16(wv16[(size_t) o * N + i]) * (native_value ? (double) emb[i] : from_bf16(bf16(emb[i])));
                const Ref r = oracle(key_proj, value, hidden, hist, nk, nq, nc, taps);
                float *d_emb = dev(emb), *d_hidden = dev(hidden), *d_hist = dev(hist);
                float *o_key = dev_zero<float>(D), *o_value = dev_zero<float>(N), *o_gate = dev_zero<float>(H),
                      *o_gated = dev_zero<float>(D), *o_norm = dev_zero<float>(D), *o_conv = dev_zero<float>(D),
                      *o_res = dev_zero<float>(D);
                k::PleOut out{o_key, o_value, o_gate, o_gated, o_norm, o_conv, o_res};
                k::ple_set_native_bf16(native_value);
                k::ple_set_native_postops(native_post);
                k::ple_block(d_emb, d_hidden, d_hist, weights(native_key), out, d_scratch, s);
                ck(cudaStreamSynchronize(s), "sync");
                // the Q8_1 activation rounding of the native key bounds its stages at ~1e-2; everything else is
                // FP32 against FP64
                const double tol = native_key ? 2e-2 : 1e-5, tol_v = 1e-5;
                const std::string tag = std::string(native_key ? "Q8_0 key" : "BF16 key") +
                                        (native_value ? ", native value" : ", BF16 value") +
                                        (native_post ? ", native postops" : ", legacy postops");
                const struct { const char* name; float* p; const std::vector<double>* ref; double tol; } stages[] = {
                    {"key", o_key, &r.key, tol}, {"value", o_value, &r.value, tol_v}, {"gate", o_gate, &r.gate, tol},
                    {"gated", o_gated, &r.gated, tol}, {"normalized", o_norm, &r.normalized, tol},
                    {"conv", o_conv, &r.conv, tol}, {"result", o_res, &r.result, tol}};
                double worst = 0;
                std::string worst_name;
                for (const auto& st : stages) {
                    const double e = rel_l1(host(st.p, st.ref->size()), *st.ref);
                    if (!(e <= st.tol)) expect(false, tag + ": " + st.name, "rel " + std::to_string(e));
                    if (!(e <= worst)) { worst = e; worst_name = st.name; }
                }
                char buf[96];
                std::snprintf(buf, sizeof buf, "worst rel %.2e (%s)", worst, worst_name.c_str());
                expect(std::isfinite(worst), tag + ": all seven stages", buf);
                for (float* p : {d_emb, d_hidden, d_hist, o_key, o_value, o_gate, o_gated, o_norm, o_conv, o_res}) cudaFree(p);
            }
    k::ple_set_native_bf16(false);
    k::ple_set_native_postops(false);

    // ---- (2) the native batch vs T single-token calls
    std::printf("-- native_ple_postops_batch vs T native_ple_postops calls + ple_history_advance\n");
    const k::PleWeights w = weights(false);
    for (int T : {1, 5, 9, 13}) {
        const std::vector<float> keys = randv((size_t) T * D, 1.0f), hidden = randv((size_t) T * D, 1.0f),
                                 values = randv((size_t) T * N, 0.5f), hist = randv((size_t) HIST * D, 1.0f);
        // sequential
        float *d_hist = dev(hist), *bk = dev_zero<float>(D), *bq = dev_zero<float>(D), *bg = dev_zero<float>(H),
              *bgd = dev_zero<float>(D), *bn = dev_zero<float>(D), *bc = dev_zero<float>(D), *br = dev_zero<float>(D);
        float *d_keys = dev(keys), *d_hidden = dev(hidden), *d_values = dev(values);
        std::vector<float> seq((size_t) T * D);
        for (int t = 0; t < T; ++t) {
            k::NativePlePostopsBuffers b{bk, bq, bg, bgd, bn, bc, br};
            k::native_ple_postops(d_keys + (size_t) t * D, d_hidden + (size_t) t * D, d_values + (size_t) t * N, d_hist, w, b, s);
            k::ple_history_advance(d_hist, bn, s);
            ck(cudaMemcpyAsync(seq.data() + (size_t) t * D, br, D * 4, cudaMemcpyDeviceToHost, s), "seq");
        }
        ck(cudaStreamSynchronize(s), "sync");
        const std::vector<float> seq_hist = host(d_hist, (size_t) HIST * D);
        // batch
        float *e_hist = dev(hist), *e_keys = dev(keys), *e_hidden = dev(hidden), *qn = dev_zero<float>((size_t) T * D),
              *gd = dev_zero<float>((size_t) T * D), *ga = dev_zero<float>((size_t) T * H);
        k::native_ple_postops_batch(e_keys, e_hidden, d_values, e_hist, w, qn, gd, ga, T, s);
        ck(cudaStreamSynchronize(s), "sync");
        const std::vector<float> bat = host(e_hidden, (size_t) T * D), bat_hist = host(e_hist, (size_t) HIST * D);
        const bool same = std::memcmp(seq.data(), bat.data(), seq.size() * 4) == 0;
        const bool same_hist = std::memcmp(seq_hist.data(), bat_hist.data(), seq_hist.size() * 4) == 0;
        expect(same && same_hist, "T = " + std::to_string(T) + ": results and history bitwise equal",
               same ? (same_hist ? "" : "history differs") : "results differ");
        for (float* p : {d_hist, bk, bq, bg, bgd, bn, bc, br, d_keys, d_hidden, d_values, e_hist, e_keys, e_hidden, qn, gd, ga})
            cudaFree(p);
    }

    // ---- (3) a captured block replayed with changed inputs
    std::printf("-- ple_block under graph capture\n");
    for (int native_post = 0; native_post < 2; ++native_post) {
        k::ple_set_native_postops(native_post);
        k::ple_set_native_bf16(native_post);
        const k::PleWeights wq = weights(true);
        float *d_emb = dev_zero<float>(N), *d_hidden = dev_zero<float>(D), *d_hist = dev_zero<float>((size_t) HIST * D),
              *o_res = dev_zero<float>(D), *o_norm = dev_zero<float>(D);
        k::PleOut out{};
        out.result = o_res;
        out.normalized = o_norm;
        cudaGraph_t g;
        cudaGraphExec_t ge;
        ck(cudaStreamBeginCapture(s, cudaStreamCaptureModeThreadLocal), "capture");
        k::ple_block(d_emb, d_hidden, d_hist, wq, out, d_scratch, s);
        ck(cudaStreamEndCapture(s, &g), "end capture");
        ck(cudaGraphInstantiate(&ge, g, nullptr, nullptr, 0), "instantiate");
        bool all = true;
        for (int round = 0; round < 3; ++round) {
            const std::vector<float> emb = randv(N, 1.0f), hidden = randv(D, 1.0f), hist = randv((size_t) HIST * D, 1.0f);
            ck(cudaMemcpy(d_emb, emb.data(), N * 4, cudaMemcpyHostToDevice), "emb");
            ck(cudaMemcpy(d_hidden, hidden.data(), D * 4, cudaMemcpyHostToDevice), "hidden");
            ck(cudaMemcpy(d_hist, hist.data(), (size_t) HIST * D * 4, cudaMemcpyHostToDevice), "hist");
            ck(cudaGraphLaunch(ge, s), "launch");
            ck(cudaStreamSynchronize(s), "sync");
            const std::vector<float> replay = host(o_res, D), replay_n = host(o_norm, D);
            k::ple_block(d_emb, d_hidden, d_hist, wq, out, d_scratch, s);
            ck(cudaStreamSynchronize(s), "sync");
            all = all && std::memcmp(replay.data(), host(o_res, D).data(), D * 4) == 0 &&
                  std::memcmp(replay_n.data(), host(o_norm, D).data(), D * 4) == 0;
        }
        expect(all, std::string(native_post ? "native" : "legacy") + " postops, Q8_0 key: 3 replays == direct calls, bitwise");
        cudaGraphExecDestroy(ge);
        cudaGraphDestroy(g);
        for (float* p : {d_emb, d_hidden, d_hist, o_res, o_norm}) cudaFree(p);
    }
    k::ple_set_native_postops(false);
    k::ple_set_native_bf16(false);

    // ---- (4) history advance and refusals
    std::printf("-- ple_history_advance and refusals\n");
    {
        const std::vector<float> hist = randv((size_t) HIST * D, 1.0f), norm = randv(D, 1.0f);
        float *d_hist = dev(hist), *d_norm = dev(norm);
        k::ple_history_advance(d_hist, d_norm, s);
        ck(cudaStreamSynchronize(s), "sync");
        std::vector<float> want(hist);
        for (int c = 0; c < D; ++c) {
            for (int r = 0; r + 1 < HIST; ++r) want[(size_t) c * HIST + r] = hist[(size_t) c * HIST + r + 1];
            want[(size_t) c * HIST + HIST - 1] = norm[c];
        }
        expect(std::memcmp(want.data(), host(d_hist, want.size()).data(), want.size() * 4) == 0,
               "history advance == host shift, bitwise");
        bool threw = false;
        try { k::ple_history_advance(d_hist, d_hist + 5, s); } catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "history advance refuses an overlapping input");
        threw = false;
        k::PleOut out{};
        out.result = (float*) d_scratch;
        try { k::ple_block(d_norm, d_norm, d_hist, weights(false), out, d_scratch, s); } catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "ple_block refuses an output inside its scratch");
        threw = false;
        k::NativePlePostopsBuffers b{d_norm, d_norm, d_norm, d_norm, d_norm, d_norm, d_norm};
        try { k::native_ple_postops(d_norm, d_norm, d_norm, d_hist, weights(false), b, s); } catch (const std::invalid_argument&) { threw = true; }
        expect(threw, "native postops refuse overlapping writable spans");
        cudaFree(d_hist);
        cudaFree(d_norm);
    }

    std::printf("ple_block: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
