// src/kernels/sycl/s_gemv.cpp - SYCL port of src/kernels/cuda/s_gemv.cu (P2.S2: the S-family GEMV).
//
// The CUDA file's header and per-kernel comments carry the contract (the offset belongs to the WEIGHT and is
// applied before the activation multiply, the group index is a shift because every group size is a power of
// two, the IQ4NL codebook lives in shared memory because a divergent constant read measured 2.12x of the whole
// kernel) and are not repeated here; the arithmetic below is line-for-line the same.  What SYCL forces to
// change:
//
//   * `__constant__ kIq4Nl` -> a namespace-scope constexpr array, still staged into LOCAL memory once per
//     work-group (the shared-memory reasoning carries over unchanged).
//   * `__shared__` + `__syncthreads()` -> a local accessor + `sycl::group_barrier`, with the load and the
//     barrier BEFORE the early return exactly as the CUDA file orders them (see its note: a thread that
//     returns first would leave the others at a barrier it never reaches).
//   * warp shuffles -> 32-wide `sycl::sub_group` shifts (`[[sycl::reqd_sub_group_size(32)]]`; the runtime
//     refuses devices without 32-wide sub-groups).
//   * `__half2float` -> `f32_from_f16` (exact, oracle-validated - see f16_bits.hpp), `__ldg` -> a plain load.
#include "strata/kernels/s_gemv.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

// The non-linear codebook, verbatim from ggml-common.h / ggml-quants.c - see the CUDA file for why it is
// staged into per-block shared memory rather than read from constant memory.
constexpr signed char kIq4Nl[16] = {-127, -104, -83, -65, -49, -35, -22, -10,
                                    1,   13,   25,  38,  53,  69,  89,  113};

template <int CODE_BITS>
float decode_tbl(int code, int bias, int codebook, const signed char* tbl) {
    if (codebook == (int) Codebook::Iq4Nl) return (float) tbl[code & 0x0F];
    return (float) (code + bias);       // the bias is applied to the CODE, in the integer domain
}

// ---- Q8_K / Q8_0 ACTIVATIONS: the layouts and the stride contract are the CUDA file's, verbatim.

constexpr int Q8K_BLOCK_BYTES = 292;
constexpr int Q8K_BLOCK_ELEMS = 256;
constexpr int Q8_0_BLOCK_BYTES = 34;
constexpr int Q8_0_BLOCK_ELEMS = 32;

float q8k_at(const uint8_t* x, long long i) {
    const uint8_t* blk = x + (i / Q8K_BLOCK_ELEMS) * Q8K_BLOCK_BYTES;
    const float d = *reinterpret_cast<const float*>(blk);
    const int8_t q = ((const int8_t*) (blk + 4))[i % Q8K_BLOCK_ELEMS];
    return d * (float) q;
}

float q8_0_at(const uint8_t* x, long long i) {
    const uint8_t* blk = x + (i / Q8_0_BLOCK_ELEMS) * Q8_0_BLOCK_BYTES;
    // a `uint16_t` needs 2-byte alignment and the stride is 34, so every block's `d` is aligned
    const float d = f32_from_f16(*reinterpret_cast<const uint16_t*>(blk));
    const int8_t q = ((const int8_t*) (blk + 2))[i % Q8_0_BLOCK_ELEMS];
    return d * (float) q;
}

void check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

void sync_checked(sycl::queue& q, const char* what) {
    q.wait();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// ---- ONE WORK-ITEM PER OUTPUT ROW (the CUDA file's one-thread-per-row naive kernels) ---------------------

template <int CODE_BITS>
void s_gemv_row(sycl::nd_item<1> it, const signed char* s_iq4nl, const uint16_t* x, const uint8_t* codes,
                const float* scales, const float* offset, float* y, long long n_in, long long n_out, int bias,
                int codebook, int group_elems, int has_offset) {
    const long long o = (long long) it.get_group(0) * (long long) it.get_local_range(0) +
                        (long long) it.get_local_id(0);
    if (o >= n_out) return;

    constexpr int PER_BYTE = 8 / CODE_BITS;
    const long long n_groups = n_in / group_elems;
    const long long codes_per_row = n_in / PER_BYTE;
    const uint8_t* c = codes + o * codes_per_row;
    const float* s = scales + o * n_groups;
    const float* off = has_offset ? offset + o * n_groups : nullptr;

    float acc = 0.0f;
    for (long long g = 0; g < n_groups; ++g) {
        const float d = s[g];
        const float b = off ? off[g] : 0.0f;
        const long long base = g * (long long) group_elems;
        for (int j = 0; j < group_elems; ++j) {
            const long long i = base + j;
            const int code = (c[i / PER_BYTE] >> ((int) (i % PER_BYTE) * CODE_BITS)) & ((1 << CODE_BITS) - 1);
            // The offset belongs to the WEIGHT: `w = code*scale + offset; acc += w * x` (see the CUDA file's
            // note on why writing it the other way rounds differently).
            const float w = decode_tbl<CODE_BITS>(code, bias, codebook, s_iq4nl) * d + b;
            acc += w * f32_from_f16(x[i]);
        }
    }
    y[o] = acc;
}

template <int CODE_BITS>
void s_gemv_q8k_row(sycl::nd_item<1> it, const signed char* s_iq4nl, const uint8_t* x, const uint8_t* codes,
                    const float* scales, const float* offset, float* y, long long n_in, long long n_out,
                    int bias, int codebook, int group_elems, int has_offset) {
    const long long o = (long long) it.get_group(0) * (long long) it.get_local_range(0) +
                        (long long) it.get_local_id(0);
    if (o >= n_out) return;

    constexpr int PER_BYTE = 8 / CODE_BITS;
    const long long n_groups = n_in / group_elems;
    const long long codes_per_row = n_in / PER_BYTE;
    const uint8_t* c = codes + o * codes_per_row;
    const float* s = scales + o * n_groups;
    const float* off = has_offset ? offset + o * n_groups : nullptr;

    float acc = 0.0f;
    for (long long g = 0; g < n_groups; ++g) {
        const float d = s[g];
        const float b = off ? off[g] : 0.0f;
        const long long base = g * (long long) group_elems;
        for (int j = 0; j < group_elems; ++j) {
            const long long i = base + j;
            const int code = (c[i / PER_BYTE] >> ((int) (i % PER_BYTE) * CODE_BITS)) & ((1 << CODE_BITS) - 1);
            const float w = decode_tbl<CODE_BITS>(code, bias, codebook, s_iq4nl) * d + b;
            acc += w * q8k_at(x, i);
        }
    }
    y[o] = acc;
}

// The codebook staging shared by every kernel below: THE LOAD AND THE BARRIER COME BEFORE THE EARLY RETURN,
// exactly as the CUDA file orders them and for the reason its note gives.
template <typename F>
void launch_with_codebook(sycl::queue& q, size_t global, size_t local, F&& body) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<signed char, 1> tbl(sycl::range<1>(16), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(global), sycl::range<1>(local)),
                       [=](sycl::nd_item<1> it) {
            const int tid = (int) it.get_local_id(0);
            if (tid < 16) tbl[tid] = kIq4Nl[tid];
            sycl::group_barrier(it.get_group());
            body(it, &tbl[0]);
        });
    });
}

// ---- ONE WARP (32-wide sub-group) PER OUTPUT ROW, over a Q8_K or Q8_0 activation -------------------------

template <int CODE_BITS, bool Q8K>
void s_gemv_q8_split_row(sycl::nd_item<1> it, const signed char* s_iq4nl, const uint8_t* x,
                         const uint8_t* codes, const float* scales, const float* offset, float* y,
                         long long n_in, long long n_out, int bias, int codebook, int group_shift,
                         int has_offset) {
    const sycl::sub_group sg = it.get_sub_group();
    constexpr int PER_BYTE = 8 / CODE_BITS;
    const int warps_per_block = (int) (it.get_local_range(0) >> 5);
    const long long o = (long long) it.get_group(0) * warps_per_block + ((long long) it.get_local_id(0) >> 5);
    const int lane = (int) sg.get_local_linear_id();
    if (o >= n_out) return;

    const long long n_groups = n_in >> group_shift;
    const long long codes_per_row = n_in / PER_BYTE;
    const uint8_t* c = codes + o * codes_per_row;
    const float* s = scales + o * n_groups;
    const float* off = has_offset ? offset + o * n_groups : nullptr;

    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    float acc4 = 0.0f, acc5 = 0.0f, acc6 = 0.0f, acc7 = 0.0f;
    float acc8 = 0.0f, acc9 = 0.0f, acc10 = 0.0f, acc11 = 0.0f;
    float acc12 = 0.0f, acc13 = 0.0f, acc14 = 0.0f, acc15 = 0.0f;
    // ---- EIGHT... SIXTEEN CONSECUTIVE ELEMENTS PER LANE, from FOUR code loads and ONE scale.  The bit
    // order, the group_shift shift and the hoisted activation block are the CUDA file's, comment and all;
    // the summation order differs from the naive kernel, which is what the parity tests' tolerances guard.
    constexpr int QE = 16;
    constexpr int QW = 4 * CODE_BITS / 8;              // 1 for S2, 2 for S4, 4 for S8
    constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
    long long i = (long long) lane * QE;
    for (; i + QE <= n_in; i += 32 * QE) {
        const long long g = i >> group_shift;          // the whole octet is in one group: group_elems % 16 == 0
        const float d = s[g];
        const float b = off ? off[g] : 0.0f;
        const uint8_t* cp = c + i / PER_BYTE;
        unsigned v, v2, v3, v4;
        if (QW == 1) { v = cp[0]; v2 = cp[1]; v3 = cp[2]; v4 = cp[3]; }
        else if (QW == 2) { v = *reinterpret_cast<const uint16_t*>(cp); v2 = *reinterpret_cast<const uint16_t*>(cp + 2);
                            v3 = *reinterpret_cast<const uint16_t*>(cp + 4); v4 = *reinterpret_cast<const uint16_t*>(cp + 6); }
        else { v = *reinterpret_cast<const uint32_t*>(cp); v2 = *reinterpret_cast<const uint32_t*>(cp + 4);
               v3 = *reinterpret_cast<const uint32_t*>(cp + 8); v4 = *reinterpret_cast<const uint32_t*>(cp + 12); }
        // THE ACTIVATION BLOCK IS HOISTED OUT OF THE SIXTEEN ELEMENTS: QE=16 and the loop stride 32*QE keep
        // `i % 256` (and `i % 32`) a multiple of 16, so one lane-iteration can never straddle a block.
        const int blk_elems = Q8K ? Q8K_BLOCK_ELEMS : Q8_0_BLOCK_ELEMS;
        const uint8_t* xb = x + (i / blk_elems) * (Q8K ? Q8K_BLOCK_BYTES : Q8_0_BLOCK_BYTES);
        const int xi = (int) (i % blk_elems);
        const float xd = Q8K ? *reinterpret_cast<const float*>(xb)
                             : f32_from_f16(*reinterpret_cast<const uint16_t*>(xb));
        const int8_t* xq = (const int8_t*) (xb + (Q8K ? 4 : 2)) + xi;

        const float w0 = decode_tbl<CODE_BITS>((int) (v & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w1 = decode_tbl<CODE_BITS>((int) ((v >> CODE_BITS) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w2 = decode_tbl<CODE_BITS>((int) ((v >> (2 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w3 = decode_tbl<CODE_BITS>((int) ((v >> (3 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w4 = decode_tbl<CODE_BITS>((int) (v2 & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w5 = decode_tbl<CODE_BITS>((int) ((v2 >> CODE_BITS) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w6 = decode_tbl<CODE_BITS>((int) ((v2 >> (2 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w7 = decode_tbl<CODE_BITS>((int) ((v2 >> (3 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        acc0 += w0 * (xd * (float) xq[0]);
        acc1 += w1 * (xd * (float) xq[1]);
        acc2 += w2 * (xd * (float) xq[2]);
        acc3 += w3 * (xd * (float) xq[3]);
        acc4 += w4 * (xd * (float) xq[4]);
        acc5 += w5 * (xd * (float) xq[5]);
        acc6 += w6 * (xd * (float) xq[6]);
        acc7 += w7 * (xd * (float) xq[7]);
        const float w8  = decode_tbl<CODE_BITS>((int) (v3 & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w9  = decode_tbl<CODE_BITS>((int) ((v3 >> CODE_BITS) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w10 = decode_tbl<CODE_BITS>((int) ((v3 >> (2 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w11 = decode_tbl<CODE_BITS>((int) ((v3 >> (3 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w12 = decode_tbl<CODE_BITS>((int) (v4 & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w13 = decode_tbl<CODE_BITS>((int) ((v4 >> CODE_BITS) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w14 = decode_tbl<CODE_BITS>((int) ((v4 >> (2 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        const float w15 = decode_tbl<CODE_BITS>((int) ((v4 >> (3 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b;
        acc8  += w8  * (xd * (float) xq[8]);
        acc9  += w9  * (xd * (float) xq[9]);
        acc10 += w10 * (xd * (float) xq[10]);
        acc11 += w11 * (xd * (float) xq[11]);
        acc12 += w12 * (xd * (float) xq[12]);
        acc13 += w13 * (xd * (float) xq[13]);
        acc14 += w14 * (xd * (float) xq[14]);
        acc15 += w15 * (xd * (float) xq[15]);
    }
    // the last, partial quad - at most one per lane
    for (; i < n_in; i += 32 * QE) {
        for (int k = 0; k < QE && i + k < n_in; ++k) {
            const long long e = i + k;
            const long long g = e >> group_shift;
            const int code = (c[e / PER_BYTE] >> ((int) (e % PER_BYTE) * CODE_BITS)) & MASK;
            acc0 += (decode_tbl<CODE_BITS>(code, bias, codebook, s_iq4nl) * s[g] + (off ? off[g] : 0.0f)) *
                    (Q8K ? q8k_at(x, e) : q8_0_at(x, e));
        }
    }
    float acc = (((acc0 + acc1) + (acc2 + acc3)) + ((acc4 + acc5) + (acc6 + acc7))) +
                (((acc8 + acc9) + (acc10 + acc11)) + ((acc12 + acc13) + (acc14 + acc15)));
    for (int step = 16; step > 0; step >>= 1)
        acc += sycl::shift_group_left(sg, acc, static_cast<uint32_t>(step));
    if (lane == 0) y[o] = acc;
}

// ---- ONE WORK-GROUP PER OUTPUT ROW, `threads_per_row` items splitting it (fp16 activation) ----------------

template <int CODE_BITS>
void s_gemv_split_row(sycl::nd_item<1> it, float* partial, const signed char* s_iq4nl, const uint16_t* x,
                      const uint8_t* codes, const float* scales, const float* offset, float* y, long long n_in,
                      long long n_out, int bias, int codebook, int group_elems, int group_shift,
                      int has_offset, int threads_per_row) {
    const long long o = it.get_group(0);
    if (o >= n_out) return;
    const int tid = (int) it.get_local_id(0);

    constexpr int PER_BYTE = 8 / CODE_BITS;
    const long long n_groups = n_in / group_elems;
    const uint8_t* c = codes + o * (n_in / PER_BYTE);
    const float* s = scales + o * n_groups;
    const float* off = has_offset ? offset + o * n_groups : nullptr;

    // FOUR ACCUMULATORS, FOUR CONSECUTIVE ELEMENTS from ONE code load and ONE scale - the CUDA file's
    // transformation, comment and all; the parity test's tolerance guards the changed summation order.
    constexpr int QE = 4;
    constexpr int QB = QE * CODE_BITS / 8;             // 1 for S2, 2 for S4, 4 for S8
    constexpr unsigned MASK = (1u << CODE_BITS) - 1u;
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    long long i = (long long) tid * QE;
    for (; i + QE <= n_in; i += (long long) threads_per_row * QE) {
        const long long g = i >> group_shift;
        const float d = s[g];
        const float b = off ? off[g] : 0.0f;
        const uint8_t* cp = c + i / PER_BYTE;
        unsigned v;
        if (QB == 1) v = cp[0];
        else if (QB == 2) v = *reinterpret_cast<const uint16_t*>(cp);
        else v = *reinterpret_cast<const uint32_t*>(cp);
        // four halves in TWO 4-byte loads; `i` is a multiple of four, so `x + i` is 8-byte aligned
        const uint32_t xw01 = *reinterpret_cast<const uint32_t*>(x + i);
        const uint32_t xw23 = *reinterpret_cast<const uint32_t*>(x + i + 2);
        const float f01x = f32_from_f16((uint16_t) xw01);
        const float f01y = f32_from_f16((uint16_t) (xw01 >> 16));
        const float f23x = f32_from_f16((uint16_t) xw23);
        const float f23y = f32_from_f16((uint16_t) (xw23 >> 16));
        acc0 += (decode_tbl<CODE_BITS>((int) (v & MASK), bias, codebook, s_iq4nl) * d + b) * f01x;
        acc1 += (decode_tbl<CODE_BITS>((int) ((v >> CODE_BITS) & MASK), bias, codebook, s_iq4nl) * d + b) * f01y;
        acc2 += (decode_tbl<CODE_BITS>((int) ((v >> (2 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b) * f23x;
        acc3 += (decode_tbl<CODE_BITS>((int) ((v >> (3 * CODE_BITS)) & MASK), bias, codebook, s_iq4nl) * d + b) * f23y;
    }
    // the last, partial quad - at most one per thread
    for (; i < n_in; i += (long long) threads_per_row * QE) {
        for (int k = 0; k < QE && i + k < n_in; ++k) {
            const long long e = i + k;
            const long long g = e >> group_shift;
            const int code = (c[e / PER_BYTE] >> ((int) (e % PER_BYTE) * CODE_BITS)) & MASK;
            acc0 += (decode_tbl<CODE_BITS>(code, bias, codebook, s_iq4nl) * s[g] + (off ? off[g] : 0.0f)) *
                    f32_from_f16(x[e]);
        }
    }
    partial[tid] = (acc0 + acc1) + (acc2 + acc3);
    sycl::group_barrier(it.get_group());
    for (int step = threads_per_row / 2; step > 0; step >>= 1) {
        if (tid < step) partial[tid] += partial[tid + step];
        sycl::group_barrier(it.get_group());
    }
    if (tid == 0) y[o] = partial[0];
}

bool q8k_form_ok(const SForm& form, int64_t n_in, const char* who) {
    if (form.group_elems <= 0 || n_in % form.group_elems != 0) {
        std::fprintf(stderr, "%s: n_in %lld is not a multiple of group_elems %d\n", who, (long long) n_in,
                     form.group_elems);
        return false;
    }
    if (n_in % Q8K_BLOCK_ELEMS != 0) {
        // `quantize_q8_K` requires this too, and a partial block would read past the end of the activation.
        std::fprintf(stderr, "%s: n_in %lld is not a multiple of the Q8_K block %d\n", who, (long long) n_in,
                     Q8K_BLOCK_ELEMS);
        return false;
    }
    return true;
}

}  // namespace

void s_gemv(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset, float* y,
            int64_t n_in, int64_t n_out, const SForm& form) {
    if (n_in <= 0 || n_out <= 0) return;
    if (form.group_elems <= 0 || n_in % form.group_elems != 0) {
        std::fprintf(stderr, "s_gemv: n_in %lld is not a multiple of group_elems %d\n", (long long) n_in,
                     form.group_elems);
        std::exit(1);
    }
    if (form.has_offset && offset == nullptr) {
        std::fprintf(stderr, "s_gemv: form says has_offset but offset is null\n");
        std::exit(1);
    }
    const size_t threads = 128;
    const size_t blocks = ((size_t) n_out + threads - 1) / threads;
    const int cb = (int) form.codebook;
    const int bias = form.code_bias;
    const int ge = form.group_elems;
    const int ho = form.has_offset ? 1 : 0;
    sycl::queue& q = sycl_runtime::queue_from_stream(nullptr);
    switch (form.code_bits) {
        case 2:
            launch_with_codebook(q, blocks * threads, threads, [=](sycl::nd_item<1> it, const signed char* t) {
                s_gemv_row<2>(it, t, x, codes, scales, offset, y, n_in, n_out, bias, cb, ge, ho);
            });
            break;
        case 4:
            launch_with_codebook(q, blocks * threads, threads, [=](sycl::nd_item<1> it, const signed char* t) {
                s_gemv_row<4>(it, t, x, codes, scales, offset, y, n_in, n_out, bias, cb, ge, ho);
            });
            break;
        case 8:
            launch_with_codebook(q, blocks * threads, threads, [=](sycl::nd_item<1> it, const signed char* t) {
                s_gemv_row<8>(it, t, x, codes, scales, offset, y, n_in, n_out, bias, cb, ge, ho);
            });
            break;
        default:
            std::fprintf(stderr, "s_gemv: unsupported code_bits %d\n", form.code_bits);
            std::exit(1);
    }
    check_launch("s_gemv");
    sync_checked(q, "s_gemv");      // the CUDA original's unconditional cudaDeviceSynchronize
}

static void s_gemv_split_impl(const uint16_t* x, const uint8_t* codes, const float* scales,
                              const float* offset, float* y, int64_t n_in, int64_t n_out, const SForm& form,
                              int threads_per_row, void* stream, bool sync) {
    if (n_in <= 0 || n_out <= 0) return;
    if (threads_per_row < 1 || (threads_per_row & (threads_per_row - 1)) != 0 || threads_per_row > 1024) {
        std::fprintf(stderr, "s_gemv_split: threads_per_row must be a power of two in 1..1024, got %d\n",
                     threads_per_row);
        std::exit(1);
    }
    const int cb = (int) form.codebook;
    // Every group size this format defines is a power of two, which is what lets the per-element group index
    // be a shift.  Refuse anything else rather than compute a wrong index quietly.
    if (form.group_elems <= 0 || (form.group_elems & (form.group_elems - 1)) != 0) {
        std::fprintf(stderr, "s_gemv_split: group_elems must be a power of two, got %d\n", form.group_elems);
        std::exit(1);
    }
    // AND THE QUAD MUST FIT INSIDE ONE GROUP: four consecutive elements share a scale and the kernel reads
    // exactly one.  Every group size this format defines is 16, 32 or 64, so this cannot fire on a real pack.
    if (form.group_elems % 4 != 0) {
        std::fprintf(stderr, "s_gemv_split: group_elems %d is not a multiple of 4\n", form.group_elems);
        std::exit(1);
    }
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;
    const size_t tpr = (size_t) threads_per_row;
    const int bias = form.code_bias;
    const int ge = form.group_elems;
    const int ho = form.has_offset ? 1 : 0;
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> partial(sycl::range<1>(tpr), h);
        sycl::local_accessor<signed char, 1> tbl(sycl::range<1>(16), h);
        const sycl::nd_range<1> rng(sycl::range<1>((size_t) n_out * tpr), sycl::range<1>(tpr));
        switch (form.code_bits) {
            case 2:
                h.parallel_for(rng, [=](sycl::nd_item<1> it) {
                    const int tid = (int) it.get_local_id(0);
                    if (tid < 16) tbl[tid] = kIq4Nl[tid];
                    sycl::group_barrier(it.get_group());
                    s_gemv_split_row<2>(it, &partial[0], &tbl[0], x, codes, scales, offset, y, n_in, n_out,
                                        bias, cb, ge, group_shift, ho, threads_per_row);
                });
                break;
            case 4:
                h.parallel_for(rng, [=](sycl::nd_item<1> it) {
                    const int tid = (int) it.get_local_id(0);
                    if (tid < 16) tbl[tid] = kIq4Nl[tid];
                    sycl::group_barrier(it.get_group());
                    s_gemv_split_row<4>(it, &partial[0], &tbl[0], x, codes, scales, offset, y, n_in, n_out,
                                        bias, cb, ge, group_shift, ho, threads_per_row);
                });
                break;
            case 8:
                h.parallel_for(rng, [=](sycl::nd_item<1> it) {
                    const int tid = (int) it.get_local_id(0);
                    if (tid < 16) tbl[tid] = kIq4Nl[tid];
                    sycl::group_barrier(it.get_group());
                    s_gemv_split_row<8>(it, &partial[0], &tbl[0], x, codes, scales, offset, y, n_in, n_out,
                                        bias, cb, ge, group_shift, ho, threads_per_row);
                });
                break;
            default:
                std::fprintf(stderr, "s_gemv_split: unsupported code_bits %d\n", form.code_bits);
                std::exit(1);
        }
    });
    check_launch("s_gemv_split");
    if (sync) sync_checked(q, "s_gemv_split");
}

void s_gemv_split(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                  float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row) {
    s_gemv_split_impl(x, codes, scales, offset, y, n_in, n_out, form, threads_per_row, nullptr, true);
}

void s_gemv_split_async(const uint16_t* x, const uint8_t* codes, const float* scales, const float* offset,
                        float* y, int64_t n_in, int64_t n_out, const SForm& form, int threads_per_row,
                        void* stream) {
    s_gemv_split_impl(x, codes, scales, offset, y, n_in, n_out, form, threads_per_row, stream, false);
}

// ---- the Q8_K entry points -----------------------------------------------------------------------------

void s_gemv_q8k(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset, float* y,
                int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (!q8k_form_ok(form, n_in, "s_gemv_q8k")) std::exit(1);
    const size_t threads = 128;
    const size_t blocks = ((size_t) n_out + threads - 1) / threads;
    const int cb = (int) form.codebook;
    const int bias = form.code_bias;
    const int ge = form.group_elems;
    const int ho = form.has_offset ? 1 : 0;
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    switch (form.code_bits) {
        case 4:
            launch_with_codebook(q, blocks * threads, threads, [=](sycl::nd_item<1> it, const signed char* t) {
                s_gemv_q8k_row<4>(it, t, x_q8k, codes, scales, offset, y, n_in, n_out, bias, cb, ge, ho);
            });
            break;
        case 8:
            launch_with_codebook(q, blocks * threads, threads, [=](sycl::nd_item<1> it, const signed char* t) {
                s_gemv_q8k_row<8>(it, t, x_q8k, codes, scales, offset, y, n_in, n_out, bias, cb, ge, ho);
            });
            break;
        default:
            // 2-bit S2 never has a Q8_K activation: its `vec_dot_type` is Q8_0.  Refusing is better than
            // running a kernel that would be numerically wrong in a way the caller cannot see.
            std::fprintf(stderr, "s_gemv_q8k: code_bits %d has no Q8_K contract "
                                 "(S2 uses Q8_0; see docs/activation-contract.md)\n", form.code_bits);
            std::exit(1);
    }
    check_launch("s_gemv_q8k");
    if (stream == nullptr) sync_checked(q, "s_gemv_q8k");
}

namespace {

// The launcher shared by the Q8_K and Q8_0 warp-per-row split kernels: one work-group of 256 (eight 32-wide
// sub-groups) per eight output rows.
template <int CODE_BITS, bool Q8K>
void q8_split_launch(const uint8_t* x, const uint8_t* codes, const float* scales, const float* offset,
                     float* y, int64_t n_in, int64_t n_out, const SForm& form, int group_shift, void* stream,
                     const char* what) {
    const size_t threads = 256;
    const size_t warps = threads / 32;
    const size_t blocks = ((size_t) n_out + warps - 1) / warps;
    const int cb = (int) form.codebook;
    const int bias = form.code_bias;
    const int ho = form.has_offset ? 1 : 0;
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<signed char, 1> tbl(sycl::range<1>(16), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(blocks * threads), sycl::range<1>(threads)),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            // THE CODEBOOK LOAD AND THE BARRIER COME BEFORE THE EARLY RETURN - the per-warp return below
            // keeps whole sub-groups together, so the shuffle reduction's full width stays valid (the CUDA
            // file's note on the ragged-row UB this ordering fixes).
            const int tid = (int) it.get_local_id(0);
            if (tid < 16) tbl[tid] = kIq4Nl[tid];
            sycl::group_barrier(it.get_group());
            s_gemv_q8_split_row<CODE_BITS, Q8K>(it, &tbl[0], x, codes, scales, offset, y, n_in, n_out, bias,
                                                cb, group_shift, ho);
        });
    });
    check_launch(what);
    if (stream == nullptr) sync_checked(q, what);
}

}  // namespace

void s_gemv_q8k_split(const uint8_t* x_q8k, const uint8_t* codes, const float* scales, const float* offset,
                      float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    if (!q8k_form_ok(form, n_in, "s_gemv_q8k_split")) std::exit(1);
    // THE GROUP SIZE IS PASSED AS ITS LOGARITHM: see the note on the kernel.  A non-power-of-two group would
    // make the shift a WRONG INDEX rather than a slow one, so it is refused here.
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;
    if ((1 << group_shift) != form.group_elems) {
        std::fprintf(stderr, "s_gemv_q8k_split: group_elems %d is not a power of two\n", form.group_elems);
        std::exit(1);
    }
    // AND THE SIXTEEN MUST FIT INSIDE ONE GROUP.  A lane-iteration takes QE = 16 consecutive elements under
    // ONE scale, so a smaller group would read the wrong scale for most of them.
    if (form.group_elems % 16 != 0) {
        std::fprintf(stderr, "s_gemv_q8k_split: group_elems %d is not a multiple of 16\n", form.group_elems);
        std::exit(1);
    }
    switch (form.code_bits) {
        case 4:
            q8_split_launch<4, true>(x_q8k, codes, scales, offset, y, n_in, n_out, form, group_shift, stream,
                                     "s_gemv_q8k_split");
            break;
        case 8:
            q8_split_launch<8, true>(x_q8k, codes, scales, offset, y, n_in, n_out, form, group_shift, stream,
                                     "s_gemv_q8k_split");
            break;
        default:
            std::fprintf(stderr, "s_gemv_q8k_split: code_bits %d has no Q8_K contract\n", form.code_bits);
            std::exit(1);
    }
}

void s_gemv_q8_0_split(const uint8_t* x_q8_0, const uint8_t* codes, const float* scales, const float* offset,
                       float* y, int64_t n_in, int64_t n_out, const SForm& form, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    // THE ACTIVATION IS `block_q8_0`, 32 elements per block, so `n_in` must be a multiple of 32 - which is a
    // WEAKER requirement than Q8_K's 256 and is exactly why this kernel has to exist: `ffn_down_shexp` has
    // n_in = 640, a multiple of 32 that is NOT a multiple of 256.
    if (n_in % Q8_0_BLOCK_ELEMS != 0) {
        std::fprintf(stderr, "s_gemv_q8_0_split: n_in %lld is not a multiple of %d\n", (long long) n_in,
                     Q8_0_BLOCK_ELEMS);
        std::exit(1);
    }
    // the same two checks `s_gemv_q8k_split` makes, for the same reasons - see the notes there
    if (form.group_elems <= 0 || (form.group_elems & (form.group_elems - 1)) != 0) {
        std::fprintf(stderr, "s_gemv_q8_0_split: group_elems must be a power of two, got %d\n",
                     form.group_elems);
        std::exit(1);
    }
    if (form.group_elems % 16 != 0) {
        std::fprintf(stderr, "s_gemv_q8_0_split: group_elems %d is not a multiple of 16\n", form.group_elems);
        std::exit(1);
    }
    int group_shift = 0;
    while ((1 << group_shift) < form.group_elems) ++group_shift;
    switch (form.code_bits) {
        case 4:
            q8_split_launch<4, false>(x_q8_0, codes, scales, offset, y, n_in, n_out, form, group_shift, stream,
                                      "s_gemv_q8_0_split");
            break;
        case 8:
            q8_split_launch<8, false>(x_q8_0, codes, scales, offset, y, n_in, n_out, form, group_shift, stream,
                                      "s_gemv_q8_0_split");
            break;
        default:
            std::fprintf(stderr, "s_gemv_q8_0_split: code_bits %d has no Q8_0 contract\n", form.code_bits);
            std::exit(1);
    }
}

}  // namespace strata::kernels
