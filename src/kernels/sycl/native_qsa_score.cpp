// src/kernels/sycl/native_qsa_score.cpp - SYCL port of src/kernels/cuda/native_qsa_score.cu; see
// include/strata/kernels/native_qsa_score.hpp.
//
// The CUDA scorer feeds raw F32 bits to sm_80+ TF32 `mma.sync` through `ldmatrix`; Intel has no equivalent
// instruction sequence, and the TF32 rounding is that hardware's.  As the HIP backend does (gfx1100), this keeps the
// entry point and the score contract with an ordered scalar F32 dot per indexer head (`fmaf_rn`), the per-head
// ReLU, the documented head-order addition, the optional block bias, the 1e9 incomplete-tail bias and the +0 mask,
// so scores are rounded as FP32 rather than TF32 inputs.  An XMX joint_matrix version is optimization work.
#include "strata/kernels/native_qsa_score.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "fp_exact.hpp"

#include <cuda_runtime.h>
#include <sycl/ext/intel/math.hpp>

#include <atomic>
#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace strata::kernels {
namespace {

namespace im = strata::kernels::fp_exact;
std::atomic<bool> enabled{false};
constexpr int D = 128, HEADS = 4, R = 4;

struct Span { const void* p; size_t n; };
void validate(Span s) {
    const auto p = reinterpret_cast<uintptr_t>(s.p);
    if (!p || p % 4 || s.n > UINTPTR_MAX - p) throw std::invalid_argument("native QSA score requires aligned bounded spans");
}
bool overlaps(Span a, Span b) {
    const auto x = reinterpret_cast<uintptr_t>(a.p), y = reinterpret_cast<uintptr_t>(b.p);
    return x < y + b.n && y < x + a.n;
}
}  // namespace

void native_qsa_score_set_enabled(bool value) { enabled.store(value, std::memory_order_relaxed); }
bool native_qsa_score_enabled() { return enabled.load(std::memory_order_relaxed); }

void native_qsa_score(const float* pooled, const float* query, const float* bias, const QsaShapes& s,
                      const int32_t* step, int64_t max_blocks, int64_t max_cells, float* cells, void* stream) {
    if (!stream || s.idx_dim != D || s.idx_n_head != HEADS || s.idx_block != R || s.idx_top_k != 2048 || max_cells < 1 ||
        max_cells > INT32_MAX - 3 || max_blocks != max_cells / R + 1)
        throw std::invalid_argument("native QSA score requires128dim/4heads/4cells/2048budget, exact capacities and explicit stream");
    const Span spans[] = {{pooled, size_t(max_blocks) * D * 4}, {query, HEADS * D * 4}, {step, kStepCount * 4},
                          {cells, size_t(max_cells) * 4}, {bias, bias ? size_t(max_blocks) * 4 : 0}};
    const int count = bias ? 5 : 4;
    for (int i = 0; i < count; ++i) validate(spans[i]);
    for (int i = 0; i < count; ++i)
        for (int j = i + 1; j < count; ++j)
            if (overlaps(spans[i], spans[j])) throw std::invalid_argument("native QSA score spans overlap");
    const int mc = int(max_cells);
    try {
        sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> head_score(sycl::range<1>(HEADS), h);
            h.parallel_for(sycl::nd_range<1>(size_t(max_blocks) * 32, 32), [=](sycl::nd_item<1> it) {
                const int n = step[kStepNKv], full = step[kStepNBid];
                if (n < 1 || n > mc || step[kStepPos] != n - 1 || full != n / R || step[kStepWidth] != (n < 2051 ? n : 2051))
                    return;   // invalid device counts suppress writes (uniform over the work-group)
                const int row = (int) it.get_group(0);
                if (row > full) return;
                const int head = (int) it.get_local_id(0);
                if (head < HEADS) {
                    float dot = 0.0f;
#pragma unroll
                    for (int d = 0; d < D; ++d) dot = im::fmaf_rn(pooled[size_t(row) * D + d], query[size_t(head) * D + d], dot);
                    head_score[head] = dot > 0.0f ? dot : 0.0f;
                }
                sycl::group_barrier(it.get_group());
                if (head == 0) {
                    float sum = im::fadd_rn(0.0f, head_score[0]);
                    sum = im::fadd_rn(sum, head_score[1]);
                    sum = im::fadd_rn(sum, head_score[2]);
                    sum = im::fadd_rn(sum, head_score[3]);
                    if (bias) sum = im::fadd_rn(sum, bias[row]);
                    sum = im::fadd_rn(sum, row == full && n % R ? 1e9f : 0.0f);
                    sum = im::fadd_rn(sum, 0.0f);   // the live causal mask is +0
                    for (int i = row * R; i < n && i < (row + 1) * R; ++i) cells[i] = sum;
                }
            });
        });
    } catch (const sycl::exception& e) {
        throw std::runtime_error(e.what());
    }
    const auto error = cudaGetLastError();
    if (error != cudaSuccess) throw std::runtime_error(cudaGetErrorString(error));
}

}  // namespace strata::kernels
