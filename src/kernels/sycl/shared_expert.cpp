// src/kernels/sycl/shared_expert.cpp - SYCL port of src/kernels/cuda/shared_expert.cu (P2.S2: the shared
// expert).  The CUDA file's comments carry the semantic contract (SILU ON GATE, the per-token SCALAR gate,
// the double accumulator in the gate dot, the shared output added PLAIN) and are not repeated here.
//
// What SYCL forces to change:
//
//   * warp shuffles (float and DOUBLE) -> 32-wide `sycl::sub_group` shifts/broadcasts; FP64 runs under the
//     driver's emulation (`NEO_FP64_EMULATION=1`, `-cl-fp64-gen-emu`) exactly as the already-ported
//     quantize_act/elementwise double arithmetic does.
//   * `__fdividef`/`__expf` (the pinned fast-math sigmoid/silu) -> plain `/` and `sycl::exp`: both are at
//     least as accurate as the fast intrinsics they replace, and the parity test's 2e-6 gate tolerance is
//     two orders above either one's error.
//   * `bf16_gemv_fp32_mmvf(_multi)` is ported here from cuda/native_bf16.cu (the MMVF the native scalar gate
//     runs); `__fmaf_rn` -> `sycl::fma`, the xor shuffle -> `sycl::permute_group_by_xor`.
//   * `native_mmvq` / `native_quantize_q8_1` live in src/kernels/sycl/native_mmvq.cpp, as on CUDA.
//   * `to_f16_kernel` is dropped: it was already dead code in the CUDA file (the intermediate now goes
//     through `quantize_q8_0`/`quantize_q8_K`).
#include "strata/kernels/shared_expert.hpp"
#include "strata/kernels/bf16_gemv.hpp"
#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/quantize_act.hpp"
#include "strata/kernels/s2_gemv_q8.hpp"
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "strata/sycl_runtime/queue_bridge.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <climits>
#include <cstdio>
#include <limits>
#include <string>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 128;
bool native_bf16 = false;

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// silu(x) = x / (1 + exp(-x)), written as `ref/moe.py` writes it, in DOUBLE then cast - the reference works
// in float64 and a float32 exp differs in the last bits.  The multiplication by `up` is the reference's order.
void swiglu_item(int64_t i, const float* gate, const float* up, float* out, int n) {
    if (i >= n) return;
    const double x = (double) gate[i];
    out[i] = (float) (x / (1.0 + sycl::exp(-x))) * up[i];
}

// Pinned CUDA unary.cuh op_silu, then unary_gated_op_kernel's multiply - the fast-math contract (see the
// header note above for the replacement of __fdividef/__expf).
void native_swiglu_item(int64_t i, const float* gate, const float* up, float* out, int n) {
    if (i >= n) return;
    out[i] = (gate[i] / (1.0f + sycl::exp(-gate[i]))) * up[i];
}

double warp_sum_d(sycl::sub_group sg, double v) {
    for (int off = 16; off > 0; off >>= 1)
        v += sycl::shift_group_left(sg, v, static_cast<uint32_t>(off));
    return sycl::group_broadcast(sg, v, 0);
}

// The per-token scalar gate: sigmoid(dot(x, w)) with w = `ffn_gate_inp_shexp` (n_embd,).  ONE work-group,
// because the output is ONE scalar; BOTH OPERANDS ARE BF16, which is the reference's contract (see the CUDA
// file's note).  The DOUBLE accumulator is kept for the same reason the CUDA file keeps it.
void scalar_gate_group(sycl::nd_item<1> it, double* scratch, const uint16_t* x_bf16,
                       const uint16_t* w_bf16, float* out, int n_embd) {
    const int tid = (int) it.get_local_id(0);
    const int block = (int) it.get_local_range(0);
    double acc = 0.0;
    for (int i = tid; i < n_embd; i += block)
        acc += (double) f32_from_bf16(x_bf16[i]) * (double) f32_from_bf16(w_bf16[i]);
    sycl::group_barrier(it.get_group());
    const sycl::sub_group sg = it.get_sub_group();
    const int lane = (int) sg.get_local_linear_id(), warp = tid >> 5;
    acc = warp_sum_d(sg, acc);
    if (lane == 0) scratch[warp] = acc;
    sycl::group_barrier(it.get_group());
    const int nw = (block + 31) >> 5;
    if (warp == 0) {
        acc = (tid < nw) ? scratch[tid] : 0.0;
        acc = warp_sum_d(sg, acc);
        if (tid == 0) out[0] = (float) (1.0 / (1.0 + sycl::exp(-acc)));
    }
}

void scale_item(int64_t i, float* out, const float* g, int n) {
    if (i < n) out[i] *= g[0];
}

void native_sigmoid_item(int64_t t, float* gate) {
    gate[t] = 1.0f / (1.0f + sycl::exp(-gate[t]));
}

// The MoE block's final combination: routed outputs router-WEIGHTED, the shared output added PLAIN, the sum
// accumulated in DOUBLE in the reference's own order (see the CUDA file and the header for the two readings
// this pins).
void moe_combine_item(int64_t j, const float* parts, const float* weights, const float* shared, float* y,
                      int n_embd, int k, int has_shared) {
    if (j >= n_embd) return;
    double acc = 0.0;
    for (int e = 0; e < k; ++e) acc += (double) weights[e] * (double) parts[(size_t) e * n_embd + j];
    if (has_shared) acc += (double) shared[j];
    y[j] = (float) acc;
}

// A flat parallel_for over n items: the CUDA `grid_for(n), THREADS` shape (the elementwise port's idiom).
template <typename F>
void launch_items(int64_t n, void* stream, F&& body) {
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    q.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> idx) {
        body(static_cast<int64_t>(idx[0]));
    });
}

// ---- the FP32-activation BF16 MMVF (ported from cuda/native_bf16.cu, llama.cpp-pinned arithmetic) ---------

float mmvf_warp_sum(sycl::sub_group sg, float value) {
    for (int offset = 16; offset > 0; offset >>= 1)
        value += sycl::permute_group_by_xor(sg, value, static_cast<uint32_t>(offset));
    return value;
}

template <int BLOCK_SIZE>
void mmvf_group(sycl::nd_item<1> it, float* partials, const float* x, const uint16_t* w, float* y, int n_in) {
    const int t = (int) it.get_local_id(0);
    const sycl::sub_group sg = it.get_sub_group();
    const uint16_t* row = w + (size_t) it.get_group(0) * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    if constexpr (BLOCK_SIZE > 32) {
        if (t < 32) partials[t] = 0.0f;
        sycl::group_barrier(it.get_group());
    }
    float acc = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        // Match the two ordered multiply-adds in ggml_cuda_mad, not a pair sum followed by one add.
        acc = sycl::fma(f32_from_bf16((uint16_t) weight), x[2 * pair], acc);
        acc = sycl::fma(f32_from_bf16((uint16_t) (weight >> 16)), x[2 * pair + 1], acc);
    }
    acc = mmvf_warp_sum(sg, acc);
    if constexpr (BLOCK_SIZE > 32) {
        // All lanes have the same reduced value; one store avoids a same-value shared-memory race.
        if ((t & 31) == 0) partials[t / 32] = acc;
        sycl::group_barrier(it.get_group());
        if (t < 32) acc = mmvf_warp_sum(sg, partials[t]);
    }
    if (t == 0) y[it.get_group(0)] = acc;
}

// the same kernel for up to 8 activation rows - the weight row is read ONCE and every token keeps its own
// accumulator with exactly the single-row kernel's order (see the CUDA file).
template <int BLOCK_SIZE, int NT>
void mmvf_multi_group(sycl::nd_item<1> it, float* partials, const float* x, int64_t ldx, const uint16_t* w,
                      float* y, int64_t ldy, int n_in, int n_tok) {
    const int t = (int) it.get_local_id(0);
    const sycl::sub_group sg = it.get_sub_group();
    const uint16_t* row = w + (size_t) it.get_group(0) * n_in;
    const uint32_t* weights2 = reinterpret_cast<const uint32_t*>(row);
    if constexpr (BLOCK_SIZE > 32) {
        if (t < 32)
            for (int k = 0; k < NT; ++k) partials[k * 32 + t] = 0.0f;
        sycl::group_barrier(it.get_group());
    }
    float acc[NT];
    for (int k = 0; k < NT; ++k) acc[k] = 0.0f;
    for (int pair = t; pair < n_in / 2; pair += BLOCK_SIZE) {
        const uint32_t weight = weights2[pair];
        const float w0 = f32_from_bf16((uint16_t) weight), w1 = f32_from_bf16((uint16_t) (weight >> 16));
        for (int k = 0; k < NT; ++k) {
            if (k < n_tok) {
                const float* xk = x + (size_t) k * ldx;
                acc[k] = sycl::fma(w0, xk[2 * pair], acc[k]);
                acc[k] = sycl::fma(w1, xk[2 * pair + 1], acc[k]);
            }
        }
    }
    for (int k = 0; k < NT; ++k) acc[k] = mmvf_warp_sum(sg, acc[k]);
    if constexpr (BLOCK_SIZE > 32) {
        if ((t & 31) == 0)
            for (int k = 0; k < NT; ++k) partials[k * 32 + t / 32] = acc[k];
        sycl::group_barrier(it.get_group());
        if (t < 32)
            for (int k = 0; k < NT; ++k) acc[k] = mmvf_warp_sum(sg, partials[k * 32 + t]);
    }
    if (t == 0)
        for (int k = 0; k < NT; ++k)
            if (k < n_tok) y[(size_t) k * ldy + it.get_group(0)] = acc[k];
}

int mmvf_block_size(int64_t n_in) {
    int best = 32;
    int64_t best_iterations = (n_in + 63) / 64;
    for (int candidate = 64; candidate <= 256; candidate += 32) {
        const int64_t iterations = (n_in + 2 * candidate - 1) / (2 * candidate);
        if (iterations < best_iterations) {
            best_iterations = iterations;
            best = candidate;
        }
    }
    return best;
}

template <int N>
void mmvf_launch(sycl::queue& q, const float* x, const uint16_t* w, float* y, int n_in, size_t n_out) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partials(sycl::range<1>(N > 32 ? 32 : 1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(n_out * N), sycl::range<1>(N)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            mmvf_group<N>(it, &partials[0], x, w, y, n_in);
        });
    });
}

template <int N>
void mmvf_multi_launch(sycl::queue& q, const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                       int n_in, size_t n_out, int n_tok) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partials(sycl::range<1>(N > 32 ? 8 * 32 : 1), h);
        const sycl::nd_range<1> rng(sycl::range<1>(n_out * N), sycl::range<1>(N));
        if (n_tok <= 4) {
            h.parallel_for(rng, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                mmvf_multi_group<N, 4>(it, &partials[0], x, ldx, w, y, ldy, n_in, n_tok);
            });
        } else {
            h.parallel_for(rng, [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                mmvf_multi_group<N, 8>(it, &partials[0], x, ldx, w, y, ldy, n_in, n_tok);
            });
        }
    });
}

void mmvf_throw_if_failed(const char* what) {
    const cudaError_t result = cudaGetLastError();
    if (result != cudaSuccess)
        throw std::runtime_error(std::string(what) + " launch: " + cudaGetErrorString(result));
}

}  // namespace

void bf16_gemv_fp32_mmvf_multi(const float* x, int64_t ldx, const uint16_t* w, float* y, int64_t ldy,
                               int64_t n_in, int64_t n_out, int n_tok, void* stream) {
    if (n_tok == 1 && ldy >= n_out) { bf16_gemv_fp32_mmvf(x, w, y, n_in, n_out, stream); return; }
    if (n_tok < 1 || n_tok > 8 || n_in <= 0 || (n_in & 1) != 0 || n_out <= 0 || (ldx & 1) != 0 || x == nullptr ||
        w == nullptr || y == nullptr || (reinterpret_cast<uintptr_t>(x) & 7u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf_multi: 1..8 rows, even n_in/ldx, aligned pointers");
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    switch (mmvf_block_size(n_in)) {
        case 32:  mmvf_multi_launch<32>(q, x, ldx, w, y, ldy, (int) n_in, (size_t) n_out, n_tok); break;
        case 64:  mmvf_multi_launch<64>(q, x, ldx, w, y, ldy, (int) n_in, (size_t) n_out, n_tok); break;
        case 96:  mmvf_multi_launch<96>(q, x, ldx, w, y, ldy, (int) n_in, (size_t) n_out, n_tok); break;
        case 128: mmvf_multi_launch<128>(q, x, ldx, w, y, ldy, (int) n_in, (size_t) n_out, n_tok); break;
        case 160: mmvf_multi_launch<160>(q, x, ldx, w, y, ldy, (int) n_in, (size_t) n_out, n_tok); break;
        case 192: mmvf_multi_launch<192>(q, x, ldx, w, y, ldy, (int) n_in, (size_t) n_out, n_tok); break;
        case 224: mmvf_multi_launch<224>(q, x, ldx, w, y, ldy, (int) n_in, (size_t) n_out, n_tok); break;
        case 256: mmvf_multi_launch<256>(q, x, ldx, w, y, ldy, (int) n_in, (size_t) n_out, n_tok); break;
    }
    mmvf_throw_if_failed("bf16_gemv_fp32_mmvf_multi");
}

void bf16_gemv_fp32_mmvf(const float* x, const uint16_t* w, float* y,
                         int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || (n_in & 1) != 0 || n_in > std::numeric_limits<int>::max() ||
        n_out <= 0 || n_out > std::numeric_limits<int>::max())
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: require positive even n_in and positive n_out <= INT_MAX");
    if (x == nullptr || w == nullptr || y == nullptr ||
        (reinterpret_cast<uintptr_t>(x) & 7u) != 0 ||
        (reinterpret_cast<uintptr_t>(w) & 3u) != 0 ||
        (reinterpret_cast<uintptr_t>(y) & 3u) != 0)
        throw std::invalid_argument("bf16_gemv_fp32_mmvf: null or misaligned pointer");
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    switch (mmvf_block_size(n_in)) {
        case 32:  mmvf_launch<32>(q, x, w, y, (int) n_in, (size_t) n_out); break;
        case 64:  mmvf_launch<64>(q, x, w, y, (int) n_in, (size_t) n_out); break;
        case 96:  mmvf_launch<96>(q, x, w, y, (int) n_in, (size_t) n_out); break;
        case 128: mmvf_launch<128>(q, x, w, y, (int) n_in, (size_t) n_out); break;
        case 160: mmvf_launch<160>(q, x, w, y, (int) n_in, (size_t) n_out); break;
        case 192: mmvf_launch<192>(q, x, w, y, (int) n_in, (size_t) n_out); break;
        case 224: mmvf_launch<224>(q, x, w, y, (int) n_in, (size_t) n_out); break;
        case 256: mmvf_launch<256>(q, x, w, y, (int) n_in, (size_t) n_out); break;
    }
    mmvf_throw_if_failed("bf16_gemv_fp32_mmvf");
}

void shared_expert_set_native_bf16(bool enabled) { native_bf16 = enabled; }

namespace {

void scale_rows_launch(float* out, const float* g, int64_t n, int n_tok, void* stream) {
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    const size_t blocks = ((size_t) n + THREADS - 1) / THREADS;
    q.parallel_for(
        sycl::nd_range<2>(sycl::range<2>(blocks * THREADS, (size_t) n_tok), sycl::range<2>(THREADS, 1)),
        [=](sycl::nd_item<2> it) {
            const int t = (int) it.get_group(1);
            const int64_t i = (int64_t) it.get_group(0) * THREADS + (int64_t) it.get_local_id(0);
            if (i < n) out[(size_t) t * n + i] *= g[t];
        });
}

}  // namespace

void shared_expert_multi(int n_tok, const float* x, const uint16_t* x_bf16, const NativeSharedWeights& nw,
                         const uint16_t* gate_inp_bf16, float* gate, float* up, float* g, float* out, int64_t n_embd,
                         int64_t n_ff, void* stream) {
    if (n_tok < 1 || n_tok > 8 || !nw.q8_1 || !nw.gate_data || !nw.up_data || !nw.down_data || !stream)
        throw std::invalid_argument("shared_expert_multi: needs 1..8 tokens, native weights, scratch and a stream");
    native_quantize_q8_1(x, nw.q8_1, (int) n_embd, n_tok, stream);
    native_mmvq(nw.gate_type, nw.gate_data, nw.q8_1, gate, (int) n_embd, (int) n_ff, n_tok, stream);
    native_mmvq(nw.up_type, nw.up_data, nw.q8_1, up, (int) n_embd, (int) n_ff, n_tok, stream);
    const int n = (int) (n_ff * n_tok);
    launch_items(n, stream, [=](int64_t i) { native_swiglu_item(i, gate, up, gate, n); });
    native_quantize_q8_1(gate, nw.q8_1, (int) n_ff, n_tok, stream);
    native_mmvq(nw.down_type, nw.down_data, nw.q8_1, out, (int) n_ff, (int) n_embd, n_tok, stream);
    static const bool batch = [] { const char* v = std::getenv("STRATA_DEC_BATCH"); return v == nullptr || std::atoi(v) != 0; }();
    if (native_bf16 && batch && n_tok > 1) {   // one gemv for all rows (outputs identical), one sigmoid launch
        bf16_gemv_fp32_mmvf_multi(x, n_embd, gate_inp_bf16, g, 1, n_embd, 1, n_tok, stream);
        launch_items(n_tok, stream, [=](int64_t t) { native_sigmoid_item(t, g); });
    } else
    for (int t = 0; t < n_tok; ++t) {
        if (native_bf16) {
            bf16_gemv_fp32_mmvf(x + (size_t) t * n_embd, gate_inp_bf16, g + t, n_embd, 1, stream);
            launch_items(1, stream, [=](int64_t i) { native_sigmoid_item(i, g + t); });
        } else {
            sycl::queue& q = sycl_runtime::queue_from_stream(stream);
            q.submit([&](sycl::handler& h) {
                sycl::local_accessor<double, 1> scratch(sycl::range<1>(8), h);
                h.parallel_for(sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
                               [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                    scalar_gate_group(it, &scratch[0], x_bf16 + (size_t) t * n_embd, gate_inp_bf16, g + t,
                                      (int) n_embd);
                });
            });
        }
    }
    scale_rows_launch(out, g, n_embd, n_tok, stream);
    check_launch("shared_expert_multi");
}

uint64_t shared_expert_scratch_bytes(int64_t n_ff) {
    // gate (n_ff f32) | up (n_ff f32) | q8_0 (n_ff/32*34) | q8k (n_ff/256*292) | g (1 f32), 16-byte aligned
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    return a * 2 + q0 + qk + 32;
}

void shared_expert(const uint8_t* x_q8_0, const uint8_t* x_q8k, const uint16_t* x_bf16, const SForm& gate_form,
                   const uint8_t* gate_codes, const float* gate_scales, const float* gate_off,
                   const SForm& up_form, const uint8_t* up_codes, const float* up_scales, const float* up_off,
                   const SForm& down_form, const uint8_t* down_codes, const float* down_scales,
                   const float* down_off, const uint16_t* gate_inp_bf16, float* scratch, float* out,
                   int64_t n_embd, int64_t n_ff, int tpr, void* stream, const float* x_f32,
                   const NativeSharedWeights* native) {
    if (n_embd <= 0 || n_ff <= 0) return;
    const bool use_native = native_bf16;
    const bool native_gate = native && native->gate_data && native_mmvq_supported(native->gate_type);
    const bool native_up = native && native->up_data && native_mmvq_supported(native->up_type);
    const bool native_down = native && native->down_data && native_mmvq_supported(native->down_type);
    const bool native_projection = native_gate || native_up || native_down;
    if ((use_native || native_gate || native_up) && !x_f32)
        throw std::invalid_argument("shared_expert native input projection requires unrounded x_f32");
    if (native_projection) {
        if (!native->q8_1 || !stream || n_embd > INT_MAX || n_ff > INT_MAX)
            throw std::invalid_argument("shared_expert native projections require scratch, stream and int32 dimensions");
        // Validate every active shape before any kernel is enqueued.
        if (native_gate) native_mmvq_weight_bytes(native->gate_type, (int) n_embd, (int) n_ff);
        if (native_up) native_mmvq_weight_bytes(native->up_type, (int) n_embd, (int) n_ff);
        if (native_down) native_mmvq_weight_bytes(native->down_type, (int) n_ff, (int) n_embd);
    }
    if (scratch == nullptr) {
        std::fprintf(stderr, "shared_expert: scratch is null; the caller owns it "
                             "(see shared_expert_scratch_bytes)\n");
        std::exit(1);
    }
    // CARVED FROM THE CALLER'S SCRATCH, as the CUDA file does (no token-path allocations, legal in capture).
    uint8_t* p = (uint8_t*) scratch;
    const uint64_t a = ((uint64_t) n_ff * 4 + 15) & ~15ull;
    const uint64_t q0 = ((uint64_t) (n_ff / 32) * 34 + 15) & ~15ull;
    const uint64_t qk = ((uint64_t) (n_ff / 256) * 292 + 15) & ~15ull;
    float* gate = (float*) p;
    float* up = (float*) (p + a);
    uint8_t* h_q8_0 = (uint8_t*) (p + a * 2);
    uint8_t* h_q8k = (uint8_t*) (p + a * 2 + q0);
    float* g = (float*) (p + a * 2 + q0 + qk);

    // WHICH ACTIVATION THIS PROJECTION WANTS, READ FROM ITS OWN FORM.  See `SForm::act_kind`.
    auto gemv = [&](const SForm& f, const uint8_t* codes, const float* scales, const float* off,
                    const uint8_t* act80, const uint8_t* actq8k, float* y, int64_t nin, int64_t nout) {
        if (f.code_bits == 2) {
            s2_gemv_q8(act80, codes, scales, y, nin, nout, tpr, stream);
        } else if (f.act_kind == 1) {
            s_gemv_q8k_split(actq8k, codes, scales, off, y, nin, nout, f, stream);
        } else {
            s_gemv_q8_0_split(act80, codes, scales, off, y, nin, nout, f, stream);
        }
    };

    // gate and up projections, then silu(gate) * up in place in `gate`
    if (native_gate || native_up)
        native_quantize_q8_1(x_f32, native->q8_1, (int) n_embd, 1, stream);
    if (native_gate)
        native_mmvq(native->gate_type, native->gate_data, native->q8_1, gate, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(gate_form, gate_codes, gate_scales, gate_off, x_q8_0, x_q8k, gate, n_embd, n_ff);
    if (native_up)
        native_mmvq(native->up_type, native->up_data, native->q8_1, up, (int) n_embd, (int) n_ff, 1, stream);
    else
        gemv(up_form, up_codes, up_scales, up_off, x_q8_0, x_q8k, up, n_embd, n_ff);
    if (native_projection)
        launch_items(n_ff, stream, [=](int64_t i) { native_swiglu_item(i, gate, up, gate, (int) n_ff); });
    else
        launch_items(n_ff, stream, [=](int64_t i) { swiglu_item(i, gate, up, gate, (int) n_ff); });

    // down: (n_ff) -> (n_embd), and THE INTERMEDIATE IS QUANTIZED TO THE DOWN WEIGHT'S OWN CONTRACT.
    if (native_down) {
        native_quantize_q8_1(gate, native->q8_1, (int) n_ff, 1, stream);
        native_mmvq(native->down_type, native->down_data, native->q8_1, out, (int) n_ff, (int) n_embd, 1, stream);
    } else if (down_form.act_kind == 1) {
        if (n_ff % 256 != 0) {
            std::fprintf(stderr, "shared_expert: the down weight wants Q8_K but n_ff %lld is not a multiple "
                                 "of 256; Q8_K is structurally impossible here\n", (long long) n_ff);
            std::exit(1);
        }
        quantize_q8_K(gate, h_q8k, n_ff, stream);
        gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
    } else {
        quantize_q8_0(gate, h_q8_0, n_ff, stream);
        gemv(down_form, down_codes, down_scales, down_off, h_q8_0, h_q8k, out, n_ff, n_embd);
    }

    // the per-token scalar gate, then the multiply.  The gate is computed from `x`, the ORIGINAL hidden
    // state.  The historical branch uses BF16-rounded inputs; the native branch uses the pinned FP32
    // activation contract.  ONE work-group of 256, because the output is ONE scalar (see the CUDA file).
    if (use_native) {
        bf16_gemv_fp32_mmvf(x_f32, gate_inp_bf16, g, n_embd, 1, stream);
        launch_items(1, stream, [=](int64_t i) { native_sigmoid_item(i, g); });
    } else {
        sycl::queue& q = sycl_runtime::queue_from_stream(stream);
        q.submit([&](sycl::handler& h) {
            sycl::local_accessor<double, 1> scratch_d(sycl::range<1>(8), h);
            h.parallel_for(sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                scalar_gate_group(it, &scratch_d[0], x_bf16, gate_inp_bf16, g, (int) n_embd);
            });
        });
    }
    launch_items(n_embd, stream, [=](int64_t i) { scale_item(i, out, g, (int) n_embd); });

    check_launch("shared_expert");
    if (stream == nullptr) {
        sycl::queue& q = sycl_runtime::queue_from_stream(stream);
        q.wait();
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "shared_expert: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

void moe_combine(const float* parts, const float* weights, const float* shared, float* y, int64_t n_embd,
                 int64_t k, void* stream) {
    if (n_embd <= 0 || k <= 0) return;
    // k > 64 is refused rather than truncated: silently summing the first 64 of a longer list would be a
    // wrong answer that looks like a right one, and no geometry in this artifact comes close to it.
    if (k > 64) {
        std::fprintf(stderr, "moe_combine: k = %lld exceeds 64\n", (long long) k);
        std::exit(1);
    }
    launch_items(n_embd, stream, [=](int64_t j) {
        moe_combine_item(j, parts, weights, shared, y, (int) n_embd, (int) k, shared != nullptr);
    });
    check_launch("moe_combine");
    if (stream == nullptr) {
        sycl::queue& q = sycl_runtime::queue_from_stream(stream);
        q.wait();
        const cudaError_t e = cudaGetLastError();
        if (e != cudaSuccess) {
            std::fprintf(stderr, "moe_combine: %s\n", cudaGetErrorString(e));
            std::exit(1);
        }
    }
}

}  // namespace strata::kernels
