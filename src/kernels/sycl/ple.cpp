// src/kernels/sycl/ple.cpp - SYCL port of src/kernels/cuda/ple.cu (P2.S4: the PLE block's GPU half).
//
// See include/strata/kernels/ple.hpp for the structure and the `normalized`-is-the-conv-input finding, and the
// CUDA file for the legacy arithmetic's ggml provenance (double-accumulated rms_norm of f32-rounded squares, f32
// silu, f32 gate sqrt/sigmoid).
//
// What SYCL forces to change:
//   * `block_sum` (double, warp shuffle down then a cross-warp pass) keeps its order on 32-wide sub-groups; the
//     double arithmetic runs under the FP64 emulation the other ported double reductions use.
//   * every float operation whose rounding matters is explicit: the squares and products are rounded on their
//     own before widening (`fmul_rn`), and `1 / sqrtf`, the sigmoid and the SiLU use `fdiv_rn`/`fsqrt_rn`
//     (IGC's default float division and sqrt are not correctly rounded).
//   * `expf` becomes `sycl::exp` (device libm imports fail AOT; see INTEL_SYCL.md).
#include "strata/kernels/ple.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_ple_postops.hpp"
#include "strata/kernels/ngram.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "fp_exact.hpp"

#include <cuda_runtime.h>
#include <sycl/ext/intel/math.hpp>

#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <stdexcept>
#include <string>

namespace strata::kernels {
namespace {

namespace im = strata::kernels::fp_exact;

constexpr int THREADS = 256;
bool native_bf16 = false;
bool native_postops = false;

bool overlap(const void* a, size_t a_bytes, const void* b, size_t b_bytes) {
    if (a == nullptr || b == nullptr || a_bytes == 0 || b_bytes == 0) return false;
    const uintptr_t aa = reinterpret_cast<uintptr_t>(a), bb = reinterpret_cast<uintptr_t>(b);
    return aa < bb ? bb - aa < a_bytes : aa - bb < b_bytes;
}

inline uint16_t bf16_bits(float f) {
    uint32_t i = sycl::bit_cast<uint32_t>(f);
    i = (i + ((i >> 16) & 1u) + 0x7FFFu) & 0xFFFF0000u;
    return (uint16_t) (i >> 16);
}
inline float bf16_float(uint16_t h) { return sycl::bit_cast<float>((uint32_t) h << 16); }
inline float silu_f(float x) { return im::fdiv_rn(x, im::fadd_rn(1.0f, sycl::exp(-x))); }

/// Work-group sum in double, broadcast to every work-item (the CUDA block_sum: shuffle down per warp, warp
/// partials through local memory, a second pass in warp 0).  The leading barrier keeps a fast work-item from
/// overwriting scratch[0] before a slow one has read the previous result.
double block_sum(sycl::nd_item<1> it, double v, double* scratch) {
    sycl::group_barrier(it.get_group());
    const sycl::sub_group sg = it.get_sub_group();
    const int tid = (int) it.get_local_id(0);
    const int lane = tid & 31, warp = tid >> 5;
    for (int off = 16; off > 0; off >>= 1) v += sycl::shift_group_left(sg, v, off);
    if (lane == 0) scratch[warp] = v;
    sycl::group_barrier(it.get_group());
    const int nw = ((int) it.get_local_range(0) + 31) >> 5;
    v = (tid < nw) ? scratch[tid] : 0.0;
    if (warp == 0)
        for (int off = 16; off > 0; off >>= 1) v += sycl::shift_group_left(sg, v, off);
    if (tid == 0) scratch[0] = v;
    sycl::group_barrier(it.get_group());
    return scratch[0];
}

template<class F>
void launch_groups(sycl::queue& q, size_t groups, F body) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<double, 1> scratch(sycl::range<1>(8), h);
        h.parallel_for(sycl::nd_range<1>(groups * THREADS, THREADS),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            body(it, &scratch[0]);
        });
    });
}

/// `grouped_norm`, one work-group per stream: stream c is [c*n_embd, (c+1)*n_embd).
void gnorm(sycl::queue& q, const float* x, const float* w, float* y, int n_embd, float eps, int hc) {
    launch_groups(q, (size_t) hc, [=](sycl::nd_item<1> it, double* scratch) {
        const int c = (int) it.get_group(0), tid = (int) it.get_local_id(0);
        const float* xc = x + (size_t) c * n_embd;
        const float* wc = w + (size_t) c * n_embd;
        float* yc = y + (size_t) c * n_embd;
        double acc = 0.0;
        for (int d = tid; d < n_embd; d += THREADS) acc += (double) im::fmul_rn(xc[d], xc[d]);
        const float mean = (float) (block_sum(it, acc, scratch) / (double) n_embd);
        const float scale = im::fdiv_rn(1.0f, im::fsqrt_rn(im::fadd_rn(mean, eps)));
        for (int d = tid; d < n_embd; d += THREADS) yc[d] = im::fmul_rn(im::fmul_rn(xc[d], scale), wc[d]);
    });
}

/// `s[c] = sum_d key*query / sqrt(n_embd)`, the signed square root, the sigmoid.
void gate(sycl::queue& q, const float* key, const float* query, float* g, int n_embd, float inv_sqrt_n, int hc) {
    launch_groups(q, (size_t) hc, [=](sycl::nd_item<1> it, double* scratch) {
        const int c = (int) it.get_group(0), tid = (int) it.get_local_id(0);
        const float* kc = key + (size_t) c * n_embd;
        const float* qc = query + (size_t) c * n_embd;
        double acc = 0.0;
        for (int d = tid; d < n_embd; d += THREADS) acc += (double) im::fmul_rn(kc[d], qc[d]);
        const float s = im::fmul_rn((float) block_sum(it, acc, scratch), inv_sqrt_n);
        const float mag = im::fsqrt_rn(sycl::fmax(sycl::fabs(s), 1e-6f));
        const float sgn = (s > 0.0f) ? 1.0f : ((s < 0.0f) ? -1.0f : 0.0f);
        if (tid == 0) g[c] = im::fdiv_rn(1.0f, im::fadd_rn(1.0f, sycl::exp(-(sgn * mag))));
    });
}

void ck(cudaError_t e, const char* what) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "ple_block: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

}  // namespace

void ple_set_native_bf16(bool enabled) { native_bf16 = enabled; }
void ple_set_native_postops(bool enabled) { native_postops = enabled; }
bool ple_native_postops_enabled() { return native_postops; }

void ple_history_advance(float* hist, const float* normalized, void* stream) {
    if (hist == nullptr || normalized == nullptr)
        throw std::invalid_argument("ple_history_advance: null history or normalized input");
    if (overlap(hist, (size_t) NG_HIST * NG_HC_DIM * sizeof(float), normalized, (size_t) NG_HC_DIM * sizeof(float)))
        throw std::invalid_argument("ple_history_advance: history and normalized input overlap");
    sycl_runtime::queue_from_stream(stream).parallel_for(sycl::range<1>(NG_HC_DIM), [=](sycl::id<1> id) {
        float* column = hist + id[0] * NG_HIST;
        for (int row = 0; row + 1 < NG_HIST; ++row) column[row] = column[row + 1];
        column[NG_HIST - 1] = normalized[id[0]];
    });
    ck(cudaGetLastError(), "history advance launch");
}

bool ple_block_available() {
    int n = 0;
    return cudaGetDeviceCount(&n) == cudaSuccess && n > 0;
}

uint64_t ple_block_scratch_bytes() {
    const size_t f = (size_t) (5 * NG_HC_DIM + NG_N_EMBD + NG_HC) * sizeof(float);
    const size_t q = (size_t) (NG_N_EMBD / 32) * 34;
    const size_t e = (size_t) NG_N_EMBD * sizeof(uint16_t);
    return ((f + 15) & ~(size_t) 15) + ((q + 15) & ~(size_t) 15) + e + 256;
}

void ple_block(const float* emb, const float* hidden, const float* hist_rows, const PleWeights& w, PleOut& out,
               void* scratch, void* stream) {
    const bool native_key = w.key_native_data != nullptr && w.key_bf16 == nullptr;
    if (native_key && (!emb || !hidden || !hist_rows || !out.result || !scratch || !stream || !w.key_native_q8_1 ||
                       (w.key_native_type != 42 && w.key_native_type != 18 && w.key_native_type != 23 &&
                        w.key_native_type != 8)))
        throw std::invalid_argument("ple_block: native key requires Q2_0, IQ3_XXS, IQ4_XS or Q8_0 weights, input/output, private scratch and explicit stream");
    if (emb == nullptr || hidden == nullptr || hist_rows == nullptr || out.result == nullptr) return;
    const int n_embd = NG_N_EMBD, hc = NG_HC, hc_dim = NG_HC_DIM;
    static_assert(NG_N_EMBD == 2560 && NG_HC_DIM == 10240, "native PLE key geometry changed");
    const size_t float_bytes = (size_t) (5 * hc_dim + n_embd + hc) * sizeof(float);
    cudaStream_t st = (cudaStream_t) stream;
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);

    // the workspace is the caller's (this function is captured); see the CUDA file
    const size_t q8_bytes = (size_t) (n_embd / 32) * 34;
    if (scratch == nullptr) {
        std::fprintf(stderr, "ple_block: scratch is null; the caller owns it (see ple_block_scratch_bytes)\n");
        std::exit(1);
    }
    const struct Export { const float* pointer; size_t count; const char* name; } exports[] = {
        {out.key, (size_t) hc_dim, "key"}, {out.value, (size_t) n_embd, "value"},
        {out.gate, (size_t) hc, "gate"}, {out.gated, (size_t) hc_dim, "gated"},
        {out.normalized, (size_t) hc_dim, "normalized"}, {out.conv, (size_t) hc_dim, "conv"},
        {out.result, (size_t) hc_dim, "result"}};
    for (const auto& item : exports)
        if (overlap(item.pointer, item.count * sizeof(float), scratch, (size_t) ple_block_scratch_bytes()))
            throw std::invalid_argument(std::string("ple_block: output ") + item.name + " overlaps scratch");
    if (native_key) {
        const size_t native_bytes = native_q8_1_bytes(n_embd);
        const size_t weight_bytes = native_mmvq_weight_bytes(w.key_native_type, n_embd, hc_dim);
        if ((reinterpret_cast<uintptr_t>(w.key_native_data) & 3u) || (reinterpret_cast<uintptr_t>(w.key_native_q8_1) & 3u))
            throw std::invalid_argument("ple_block: native key buffers require four-byte alignment");
        const struct Region { const void* pointer; size_t bytes; } regions[] = {
            {scratch, (size_t) ple_block_scratch_bytes()}, {emb, (size_t) n_embd * 4},
            {hidden, (size_t) hc_dim * 4}, {hist_rows, (size_t) NG_HIST * hc_dim * 4},
            {w.key_native_data, weight_bytes}, {w.value_bf16, (size_t) n_embd * n_embd * 2},
            {w.norm_key, (size_t) hc_dim * 4}, {w.norm_query, (size_t) hc_dim * 4},
            {w.norm_conv, (size_t) hc_dim * 4}, {w.conv1d_f16, (size_t) PLE_CONV_KERNEL * hc_dim * 2}};
        for (const auto& region : regions)
            if (overlap(w.key_native_q8_1, native_bytes, region.pointer, region.bytes))
                throw std::invalid_argument("ple_block: native key scratch overlaps workspace, input or weight");
        for (const auto& item : exports)
            if (overlap(w.key_native_q8_1, native_bytes, item.pointer, item.count * sizeof(float)))
                throw std::invalid_argument(std::string("ple_block: native key scratch overlaps output ") + item.name);
    }
    uint8_t* base = (uint8_t*) scratch;
    float* d_scratch = (float*) base;
    uint8_t* d_act = base + ((float_bytes + 15) & ~(size_t) 15);
    uint16_t* d_emb16 = (uint16_t*) (d_act + ((q8_bytes + 15) & ~(size_t) 15));
    float* d_key = d_scratch;
    float* d_query = d_key + hc_dim;
    float* d_norm = d_query + hc_dim;
    float* d_gated = d_norm + hc_dim;
    float* d_conv = d_gated + hc_dim;
    float* d_value = d_conv + hc_dim;
    float* d_gate = d_value + n_embd;

    // ---- key = grouped_norm(ple_key @ emb)
    if (w.key_bf16 != nullptr) {
        bf16_gemv_fp32_mmvf(emb, w.key_bf16, d_key, n_embd, hc_dim, stream);
    } else if (native_key) {
        native_quantize_q8_1(emb, w.key_native_q8_1, n_embd, 1, stream);
        native_mmvq(w.key_native_type, w.key_native_data, w.key_native_q8_1, d_key, n_embd, hc_dim, 1, stream);
    } else {
        quantize_q8_0(emb, d_act, n_embd, st);
        s2_gemv_q8(d_act, w.key_codes, w.key_scales, d_key, n_embd, hc_dim, 8, st);
    }
    if (!native_postops) {
        gnorm(q, d_key, w.norm_key, d_key, n_embd, NG_RMS_EPS, hc);
        gnorm(q, hidden, w.norm_query, d_query, n_embd, NG_RMS_EPS, hc);
    }

    // ---- value = ple_value @ emb, both operands BF16 (the activation contract for a BF16 tensor)
    if (native_bf16) {
        bf16_gemv_fp32_mmvf(emb, w.value_bf16, d_value, n_embd, n_embd, stream);
    } else {
        q.parallel_for(sycl::range<1>(n_embd), [=](sycl::id<1> i) { d_emb16[i] = bf16_bits(emb[i]); });
        const uint16_t* wv = w.value_bf16;
        q.parallel_for(sycl::range<1>(n_embd), [=](sycl::id<1> id) {
            const uint16_t* row = wv + id[0] * n_embd;
            double acc = 0.0;
            for (int i = 0; i < n_embd; ++i) acc += (double) bf16_float(d_emb16[i]) * (double) bf16_float(row[i]);
            d_value[id[0]] = (float) acc;
        });
    }

    const float* normalized_key = d_key;
    if (native_postops) {
        // the key norm has its own destination; the query storage is reused for the normalized gated values
        NativePlePostopsBuffers buffers{d_query, d_norm, d_gate, d_gated, d_norm, d_conv, out.result};
        native_ple_postops(d_key, hidden, d_value, hist_rows, w, buffers, stream);
        normalized_key = d_query;
    } else {
        gate(q, d_key, d_query, d_gate, n_embd, im::fdiv_rn(1.0f, std::sqrt((float) n_embd)), hc);
        q.parallel_for(sycl::range<1>(hc_dim), [=](sycl::id<1> i) {
            d_gated[i] = im::fmul_rn(d_value[i % n_embd], d_gate[i / n_embd]);
        });
        gnorm(q, d_gated, w.norm_conv, d_norm, n_embd, NG_RMS_EPS, hc);
        // the depthwise causal dilated conv, then SiLU; kW is ggml-native kW[k + kern*c], the history row-fastest
        const uint16_t* kW = w.conv1d_f16;
        const int kern = PLE_CONV_KERNEL, dil = NGRAM_SIZE, nhist = NG_HIST;
        q.parallel_for(sycl::range<1>(hc_dim), [=](sycl::id<1> id) {
            const int c = (int) id[0];
            float acc = 0.0f;
            for (int k = 0; k < kern; ++k) {
                const int row = nhist - (kern - 1 - k) * dil;
                const float v = (row == nhist) ? d_norm[c] : hist_rows[(size_t) row + (size_t) nhist * c];
                acc = im::fadd_rn(acc, im::fmul_rn(f32_from_f16(kW[k + kern * c]), v));
            }
            d_conv[c] = silu_f(acc);
        });
        float* result = out.result;
        q.parallel_for(sycl::range<1>(hc_dim), [=](sycl::id<1> i) {
            result[i] = im::fadd_rn(im::fadd_rn(hidden[i], d_gated[i]), d_conv[i]);
        });
    }

    // the intermediates the oracle comparison needs (`key` is the NORMALIZED key)
    if (out.key) ck(cudaMemcpyAsync(out.key, normalized_key, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "key");
    if (out.value) ck(cudaMemcpyAsync(out.value, d_value, n_embd * sizeof(float), cudaMemcpyDeviceToDevice, st), "value");
    if (out.gate) ck(cudaMemcpyAsync(out.gate, d_gate, hc * sizeof(float), cudaMemcpyDeviceToDevice, st), "gate");
    if (out.gated) ck(cudaMemcpyAsync(out.gated, d_gated, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "gated");
    if (out.normalized)
        ck(cudaMemcpyAsync(out.normalized, d_norm, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "norm");
    if (out.conv) ck(cudaMemcpyAsync(out.conv, d_conv, hc_dim * sizeof(float), cudaMemcpyDeviceToDevice, st), "conv");
    ck(cudaGetLastError(), "launch");
}

}  // namespace strata::kernels
