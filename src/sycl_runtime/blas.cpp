// src/sycl_runtime/blas.cpp - the cuBLAS subset (cublasCreate/SetStream/SetWorkspace/
// SetMathMode/GemmEx) on oneMKL's SYCL BLAS.
//
// cuBLAS is column-major, so cublasGemmEx maps 1:1 onto oneapi::mkl::blas::
// column_major::gemm, submitted to the queue cublasSetStream bound.  Workspace and
// math mode are accepted and ignored: oneMKL manages both internally.  Supported
// combinations are the two gemm.cu actually launches - bf16 x bf16 -> f32 and
// f16 x f16 -> f32, both with CUBLAS_COMPUTE_32F; anything else is NOT_SUPPORTED.
#include <algorithm>
#include <mutex>
#include <map>
#include <cstdlib>
#include "../sycl_runtime/internal.hpp"

#include "strata/sycl_runtime/queue_bridge.hpp"

#include <cublas_v2.h>

#include <mkl.h>

#include <oneapi/mkl/blas.hpp>

struct strata_cublas_handle {
    cudaStream_t stream = nullptr;  // null: the current device's default stream at call time
};

using strata::sycl_runtime::fail;
using strata::sycl_runtime::from_exception;
using strata::sycl_runtime::queue_from_stream;
using strata::sycl_runtime::device_offset_end;

namespace {

cublasStatus_t blas_fail(cublasStatus_t status) {
    fail(cudaErrorUnknown);
    return status;
}

oneapi::mkl::transpose convert(cublasOperation_t op) {
    switch (op) {
        case CUBLAS_OP_N: return oneapi::mkl::transpose::nontrans;
        case CUBLAS_OP_T: return oneapi::mkl::transpose::trans;
        default: return oneapi::mkl::transpose::conjtrans;
    }
}

// The GEMM staging buffer of one queue (below), at least `bytes`; nullptr when it cannot be allocated.  Growing it
// first waits for the queue: its earlier GEMMs may still read the old buffer.  A buffer lives as long as the process.
uint8_t* queue_scratch(sycl::queue& q, size_t bytes) {
    struct Scratch { uint8_t* p = nullptr; size_t bytes = 0; };
    static std::mutex mu;
    static std::map<const sycl::queue*, Scratch> by_queue;
    const std::lock_guard<std::mutex> lock(mu);
    Scratch& s = by_queue[&q];
    if (s.bytes >= bytes) return s.p;
    if (s.p != nullptr) {
        q.wait();
        sycl::free(s.p, q);
        s = Scratch{};
    }
    s.p = static_cast<uint8_t*>(sycl::malloc_device(bytes, q));
    s.bytes = s.p != nullptr ? bytes : 0;
    return s.p;
}

// C = alpha A^T B + beta C for a small output (m x n <= kSmallMN), A and B 16-bit (BF16 or F16), C FP32: one 16-wide
// sub-group per row of C and up to 16 columns, lanes striding k, the lanes summed by a xor butterfly.  oneMKL 2026.1's
// GEMM is not repeatable when the output is small against k (its partial sums over k meet in a varying order): on an
// Arc A770, BF16 with m <= 512 and k 2560 or 10240 gave different bits on back-to-back calls with the same inputs at
// many column counts (m 48, k 2560: n 1-3, 96, 128; m 16, k 10240: n 1-256); every case had m x n <= 8192, and m >= 1024
// never varied.  The prompt path's small projections (ssm_alpha/beta, 48 rows) did, so a prompt's state - and the
// reply - changed from run to run.  Here every output is one fixed-order dot product.
constexpr size_t kSmallMN = 65536;
template <bool BF>
float to_f(uint16_t v) {
    if constexpr (BF) return sycl::bit_cast<float>((uint32_t) v << 16);
    else return (float) sycl::bit_cast<sycl::half>(v);
}
template <bool BF>
void small_n_gemm(sycl::queue& q, int m, int n, int k, float alpha, const uint16_t* A, int lda, const uint16_t* B,
                  int ldb, float beta, float* C, int ldc) {
    constexpr int SG = 16, ROWS = 8, NC = 16;
    const size_t row_groups = ((size_t) m + ROWS - 1) / ROWS, col_groups = ((size_t) n + NC - 1) / NC;
    q.parallel_for(sycl::nd_range<2>({col_groups, row_groups * ROWS * SG}, {1, ROWS * SG}),
                   [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
        const int lane = (int) it.get_local_id(1) % SG;
        const int row = (int) (it.get_global_id(1) / SG);
        const int c0 = (int) it.get_global_id(0) * NC;
        const int nc = sycl::min(NC, n - c0);
        const bool live = row < m;
        const uint16_t* a = A + (size_t) (live ? row : 0) * lda;
        float acc[NC];
#pragma unroll
        for (int j = 0; j < NC; ++j) acc[j] = 0.0f;
        for (int kk = lane; kk < k; kk += SG) {
            const float av = to_f<BF>(a[kk]);
#pragma unroll
            for (int j = 0; j < NC; ++j)
                if (j < nc) acc[j] = sycl::fma(av, to_f<BF>(B[(size_t) (c0 + j) * ldb + kk]), acc[j]);
        }
        const sycl::sub_group sg = it.get_sub_group();
#pragma unroll
        for (int j = 0; j < NC; ++j) {
            float v = acc[j];
#pragma unroll
            for (int o = SG / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
            if (live && lane == 0 && j < nc) {
                float* c = C + (size_t) (c0 + j) * ldc + row;
                *c = beta != 0.0f ? alpha * v + beta * *c : alpha * v;
            }
        }
    });
}

// the tiled form when k, lda, ldb are even and A, B 4-byte aligned: a work-group of 8 sub-groups covers 4 rows x 16
// columns, sub-group s the pairs of k in its eighth [s k/16, (s+1) k/16) - each lane the pairs lane, lane + 16, ... of
// it in order, element 0 then 1 - and the eight partial sums are added in sub-group order through local memory: the
// same fixed order on every call
template <bool BF>
void small_n_gemm_tiled(sycl::queue& q, int m, int n, int k, float alpha, const uint16_t* A, int lda, const uint16_t* B,
                        int ldb, float beta, float* C, int ldc) {
    constexpr int SG = 16, NSG = 8, R = 4, NC = 16;
    const size_t row_groups = ((size_t) m + R - 1) / R, col_groups = ((size_t) n + NC - 1) / NC;
    const int pairs = k / 2, per = (pairs + NSG - 1) / NSG;
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> part(sycl::range<1>(NSG * R * NC), h);
        h.parallel_for(sycl::nd_range<2>({col_groups, row_groups * NSG * SG}, {1, NSG * SG}),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
            const int tid = (int) it.get_local_id(1), lane = tid % SG, sgi = tid / SG;
            const int row0 = (int) it.get_group(1) * R;
            const int c0 = (int) it.get_group(0) * NC;
            const int nc = sycl::min(NC, n - c0);
            float acc[R][NC];
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int j = 0; j < NC; ++j) acc[r][j] = 0.0f;
            const int p0 = sgi * per, p1 = sycl::min(pairs, p0 + per);
            const uint32_t* a32[R];
#pragma unroll
            for (int r = 0; r < R; ++r)
                a32[r] = reinterpret_cast<const uint32_t*>(A + (size_t) sycl::min(row0 + r, m - 1) * lda);
            for (int p = p0 + lane; p < p1; p += SG) {
                float a0[R], a1[R];
#pragma unroll
                for (int r = 0; r < R; ++r) {
                    const uint32_t w = a32[r][p];
                    a0[r] = to_f<BF>((uint16_t) (w & 0xffffu));
                    a1[r] = to_f<BF>((uint16_t) (w >> 16));
                }
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    if (j >= nc) break;
                    const uint32_t x = reinterpret_cast<const uint32_t*>(B + (size_t) (c0 + j) * ldb)[p];
                    const float b0 = to_f<BF>((uint16_t) (x & 0xffffu)), b1 = to_f<BF>((uint16_t) (x >> 16));
#pragma unroll
                    for (int r = 0; r < R; ++r) acc[r][j] = sycl::fma(a1[r], b1, sycl::fma(a0[r], b0, acc[r][j]));
                }
            }
            const sycl::sub_group sg = it.get_sub_group();
#pragma unroll
            for (int r = 0; r < R; ++r)
#pragma unroll
                for (int j = 0; j < NC; ++j) {
                    float v = acc[r][j];
#pragma unroll
                    for (int o = SG / 2; o > 0; o >>= 1) v += sycl::permute_group_by_xor(sg, v, o);
                    if (lane == 0) part[(sgi * R + r) * NC + j] = v;
                }
            sycl::group_barrier(it.get_group());
            if (tid < R * NC) {
                const int r = tid / NC, j = tid % NC;
                float v = part[r * NC + j];
                for (int s2 = 1; s2 < NSG; ++s2) v += part[(s2 * R + r) * NC + j];
                if (j < nc && row0 + r < m) {
                    float* c = C + (size_t) (c0 + j) * ldc + row0 + r;
                    *c = beta != 0.0f ? alpha * v + beta * *c : alpha * v;
                }
            }
        });
    });
}

}  // namespace

cublasStatus_t cublasCreate(cublasHandle_t* handle) {
    if (!handle) return blas_fail(CUBLAS_STATUS_INVALID_VALUE);
    *handle = new strata_cublas_handle();
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasDestroy(cublasHandle_t handle) {
    delete handle;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetStream(cublasHandle_t handle, cudaStream_t stream) {
    if (!handle) return blas_fail(CUBLAS_STATUS_INVALID_VALUE);
    handle->stream = stream;
    return CUBLAS_STATUS_SUCCESS;
}

cublasStatus_t cublasSetWorkspace(cublasHandle_t handle, void*, size_t) {
    return handle ? CUBLAS_STATUS_SUCCESS : blas_fail(CUBLAS_STATUS_INVALID_VALUE);  // oneMKL manages its own
}

cublasStatus_t cublasSetMathMode(cublasHandle_t handle, cublasMath_t) {
    return handle ? CUBLAS_STATUS_SUCCESS : blas_fail(CUBLAS_STATUS_INVALID_VALUE);  // 32F compute is the default
}

cublasStatus_t cublasGemmEx(cublasHandle_t handle, cublasOperation_t transa, cublasOperation_t transb, int m, int n,
                            int k, const void* alpha, const void* A, cudaDataType Atype, int lda, const void* B,
                            cudaDataType Btype, int ldb, const void* beta, void* C, cudaDataType Ctype, int ldc,
                            cublasComputeType_t computeType, cublasGemmAlgo_t) {
    if (!handle || !alpha || !beta || !A || !B || !C) return blas_fail(CUBLAS_STATUS_INVALID_VALUE);
    if (computeType != CUBLAS_COMPUTE_32F) return blas_fail(CUBLAS_STATUS_NOT_SUPPORTED);
    try {
        sycl::queue& q = queue_from_stream(handle->stream);
        const auto ta = convert(transa);
        const auto tb = convert(transb);
        const float a = *static_cast<const float*>(alpha);
        const float b = *static_cast<const float*>(beta);
        // a small output: the shim's own repeatable kernel (small_n_gemm); STRATA_SYCL_GEMM_ONEMKL_SMALL=1: oneMKL
        static const bool onemkl_small = std::getenv("STRATA_SYCL_GEMM_ONEMKL_SMALL") != nullptr;
        const bool small = !onemkl_small && (size_t) m * (size_t) n <= kSmallMN && ta == oneapi::mkl::transpose::trans &&
                           tb == oneapi::mkl::transpose::nontrans && Ctype == CUDA_R_32F &&
                           ((Atype == CUDA_R_16BF && Btype == CUDA_R_16BF) || (Atype == CUDA_R_16F && Btype == CUDA_R_16F));
        if (small) {
            // every operand is read where it is: one dot product per output, no tile reads past an operand's end
            const bool pairs = k % 2 == 0 && lda % 2 == 0 && ldb % 2 == 0 && reinterpret_cast<uintptr_t>(A) % 4 == 0 &&
                               reinterpret_cast<uintptr_t>(B) % 4 == 0;
            if (pairs && Atype == CUDA_R_16BF)
                small_n_gemm_tiled<true>(q, m, n, k, a, static_cast<const uint16_t*>(A), lda, static_cast<const uint16_t*>(B),
                                         ldb, b, static_cast<float*>(C), ldc);
            else if (pairs)
                small_n_gemm_tiled<false>(q, m, n, k, a, static_cast<const uint16_t*>(A), lda, static_cast<const uint16_t*>(B),
                                          ldb, b, static_cast<float*>(C), ldc);
            else if (Atype == CUDA_R_16BF)
                small_n_gemm<true>(q, m, n, k, a, static_cast<const uint16_t*>(A), lda, static_cast<const uint16_t*>(B), ldb,
                                   b, static_cast<float*>(C), ldc);
            else
                small_n_gemm<false>(q, m, n, k, a, static_cast<const uint16_t*>(A), lda, static_cast<const uint16_t*>(B), ldb,
                                    b, static_cast<float*>(C), ldc);
            return CUBLAS_STATUS_SUCCESS;
        }
        auto run = [&](const void* A_, const void* B_, int n_, void* C_) -> bool {
            if (Atype == CUDA_R_16BF && Btype == CUDA_R_16BF && Ctype == CUDA_R_32F) {
                oneapi::mkl::blas::column_major::gemm(
                    q, ta, tb, m, n_, k, a, reinterpret_cast<const oneapi::mkl::bfloat16*>(A_), lda,
                    reinterpret_cast<const oneapi::mkl::bfloat16*>(B_), ldb, b, static_cast<float*>(C_), ldc);
            } else if (Atype == CUDA_R_16F && Btype == CUDA_R_16F && Ctype == CUDA_R_32F) {
                oneapi::mkl::blas::column_major::gemm(q, ta, tb, m, n_, k, a, reinterpret_cast<const sycl::half*>(A_),
                                                      lda, reinterpret_cast<const sycl::half*>(B_), ldb, b,
                                                      static_cast<float*>(C_), ldc);
            } else if (Atype == CUDA_R_32F && Btype == CUDA_R_32F && Ctype == CUDA_R_32F) {
                oneapi::mkl::blas::column_major::gemm(q, ta, tb, m, n_, k, a, static_cast<const float*>(A_), lda,
                                                      static_cast<const float*>(B_), ldb, b, static_cast<float*>(C_), ldc);
            } else {
                return false;
            }
            return true;
        };
        // oneMKL's GEMM reads the last, partial tile of B from the wrong place when it lies more than 4 GiB into its
        // allocation (measured on an Arc A770, oneMKL 2026.1: a 1,845-column BF16 B placed past 4 GiB of a 10.5 GiB
        // allocation gave 215,040 wrong outputs - exactly its last 21 columns - and none below 4 GiB).  The prompt path
        // borrows the top of the ~10 GiB expert cache for its buffers, so its GEMMs met this and the prompt came out
        // non-finite.  Operands past 4 GiB go through a buffer of the queue's own (96 MiB, more when needed): A copied
        // in whole when it is there, B (and C's m rows, copied back; copied in first for a nonzero beta) in pieces of whole columns, one GEMM
        // per piece - every output element is still one dot product over k.
        const size_t es = Atype == CUDA_R_32F ? 4 : 2;
        const size_t a_bytes = (size_t) lda * (size_t) (transa == CUBLAS_OP_N ? k : m) * es;
        const size_t b_cols = (size_t) (transb == CUBLAS_OP_N ? n : k), b_col = (size_t) ldb * es;
        const size_t c_col = (size_t) ldc * sizeof(float);
        constexpr size_t k4G = 4ull << 30, kScratch = 96ull << 20;
        const bool far_a = device_offset_end(A, a_bytes) > k4G;
        const bool far_b = device_offset_end(B, b_cols * b_col) > k4G;
        const bool far_c = device_offset_end(C, (size_t) n * c_col) > k4G;
        static const bool no_stage = std::getenv("STRATA_SYCL_GEMM_NO_STAGE") != nullptr;   // the tests' negative control
        if (no_stage || (!far_a && !far_b && !far_c)) {
            if (!run(A, B, n, C)) return blas_fail(CUBLAS_STATUS_NOT_SUPPORTED);
            return CUBLAS_STATUS_SUCCESS;
        }
        // The staging buffer is the queue's own (a GEMM on another stream must not overwrite it while this one
        // reads it), grown when an operand needs more: A whole when it is far, B whole when it is far and
        // transposed (its columns of op(B) are rows of the stored matrix), and at least one column of B and C.
        const bool b_whole = far_b && transb != CUBLAS_OP_N;
        const size_t a_room = far_a ? (a_bytes + 255) & ~(size_t) 255 : 0;
        const size_t bw_room = b_whole ? (b_cols * b_col + 255) & ~(size_t) 255 : 0;
        const size_t per_col = (far_b && !b_whole ? b_col : 0) + (far_c ? c_col : 0);
        uint8_t* scratch = queue_scratch(q, std::max(kScratch, a_room + bw_room + per_col));
        if (scratch == nullptr) return blas_fail(CUBLAS_STATUS_ALLOC_FAILED);   // never the unprotected GEMM
        const size_t room = std::max(kScratch, a_room + bw_room + per_col);
        if (far_a) q.memcpy(scratch, A, a_bytes);
        if (b_whole) q.memcpy(scratch + a_room, B, b_cols * b_col);
        const void* Ause = far_a ? (const void*) scratch : A;
        const uint8_t* Bbase = b_whole ? scratch + a_room : static_cast<const uint8_t*>(B);
        // a piece of output columns [c0, c0 + nc): op(B)'s columns c0.. start c0 * ldb elements in for an untransposed
        // B, c0 elements in for a transposed one
        const size_t b_step = transb == CUBLAS_OP_N ? b_col : es;
        const size_t left = room - a_room - bw_room;
        const int per = per_col ? (int) std::min<size_t>((size_t) n, left / per_col) : n;
        for (int c0 = 0; c0 < n; c0 += per) {
            const int nc = std::min(per, n - c0);
            uint8_t* sb = scratch + a_room + bw_room;
            uint8_t* sc = sb + (far_b && !b_whole ? (size_t) nc * b_col : 0);
            const uint8_t* Bp = Bbase + (size_t) c0 * b_step;
            void* Cp = static_cast<uint8_t*>(C) + (size_t) c0 * c_col;
            if (far_b && !b_whole) q.memcpy(sb, Bp, (size_t) nc * b_col);
            // C's m rows of each column only: ldc > m leaves rows that are not this GEMM's (the prompt path's alpha
            // and beta projections share one buffer, interleaved by ldc), and a whole-column copy back wrote the
            // scratch's stale rows over the other projection's outputs
            if (far_c && b != 0.0f) q.ext_oneapi_memcpy2d(sc, c_col, Cp, c_col, (size_t) m * sizeof(float), (size_t) nc);
            if (!run(Ause, far_b && !b_whole ? (const void*) sb : (const void*) Bp, nc, far_c ? (void*) sc : Cp))
                return blas_fail(CUBLAS_STATUS_NOT_SUPPORTED);
            if (far_c) q.ext_oneapi_memcpy2d(Cp, c_col, sc, c_col, (size_t) m * sizeof(float), (size_t) nc);
        }
        return CUBLAS_STATUS_SUCCESS;
    } catch (const sycl::exception& e) {
        from_exception(e);
        return blas_fail(CUBLAS_STATUS_EXECUTION_FAILED);
    }
}
