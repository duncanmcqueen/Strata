// src/kernels/sycl/bf16_gemv.cpp - SYCL port of src/kernels/cuda/bf16_gemv.cu.
// The CUDA file's header comment carries the dispatch contract (warp-per-row at
// n_out >= 64, naive below; tpr == 32 selects the warp path, anything else must be
// a power of two); the port keeps the shapes and the summation orders exactly.
// Both sides of every product are bf16-valued, so each product is exact in f32 and
// plain `acc += a * b` is the whole arithmetic - no division, no pinned rounding.
// Only what SYCL forces to change:
//   * warp shuffles            -> `sycl::shift_group_left` under a required 32-wide
//     sub-group (the runtime refuses devices without it);
//   * `__shared__` + `__syncthreads()` -> a local accessor + `group_barrier`.
#include "strata/kernels/bf16_gemv.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"

#include <cuda_runtime.h>

#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int WARPS = THREADS / 32;

bool check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

void sync_if_needed(sycl::queue& q, void* stream, const char* what) {
    if (stream != nullptr) return;
    q.wait();
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

/// One thread per output row, walking it contiguously.  Kept as the naive reference the split version is
/// checked against, and used directly for the shapes where the output width is already large.
void launch_naive(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                  sycl::queue& q) {
    q.parallel_for(sycl::range<1>(static_cast<size_t>(n_out)), [=](sycl::id<1> idx) {
        const int64_t o = static_cast<int64_t>(idx[0]);
        const uint16_t* row = w + o * n_in;
        float acc = 0.0f;
        for (int64_t i = 0; i < n_in; ++i)
            acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
        y[o] = acc;
    });
}

/// One SUB-GROUP per output row, lanes striding the reduction axis - the CUDA file's warp-per-row shape.
/// For a fixed `i` consecutive lanes touch consecutive addresses in this layout, so the load is coalesced.
void launch_warp(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                 sycl::queue& q) {
    const size_t grid = static_cast<size_t>((n_out + WARPS - 1) / WARPS);
    q.parallel_for(sycl::nd_range<1>(grid * THREADS, THREADS),
                   [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int64_t o = static_cast<int64_t>(it.get_group(0)) * WARPS + sg.get_group_id();
        if (o >= n_out) return;   // whole sub-group leaves together; the collectives below stay uniform
        const int lane = static_cast<int>(sg.get_local_linear_id());
        const uint16_t* row = w + o * n_in;
        float acc = 0.0f;
        for (int64_t i = lane; i < n_in; i += 32)
            acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
        for (int off = 16; off > 0; off >>= 1)
            acc += sycl::shift_group_left(sg, acc, static_cast<uint32_t>(off));
        if (lane == 0) y[o] = acc;
    });
}

/// `tpr` work-items cooperate on ONE row's reduction: each takes a strided slice and the work-group
/// reduces through local memory.  Used when the output width is too small to fill the machine.
void launch_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, int tpr,
                  sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 1> scratch(sycl::range<1>(static_cast<size_t>(tpr)), h);
        h.parallel_for(sycl::nd_range<1>(static_cast<size_t>(n_out) * tpr, static_cast<size_t>(tpr)),
                       [=](sycl::nd_item<1> it) {
            const int64_t o = static_cast<int64_t>(it.get_group(0));
            const int t = static_cast<int>(it.get_local_id(0));   // 0 .. tpr-1
            const uint16_t* row = w + o * n_in;
            float acc = 0.0f;
            for (int64_t i = t; i < n_in; i += tpr)
                acc += f32_from_bf16(x[i]) * f32_from_bf16(row[i]);
            // work-group reduction; `tpr` is at most a few hundred, so a tree in local is enough
            scratch[t] = acc;
            sycl::group_barrier(it.get_group());
            for (int off = tpr >> 1; off > 0; off >>= 1) {
                if (t < off) scratch[t] += scratch[t + off];
                sycl::group_barrier(it.get_group());
            }
            if (t == 0) y[o] = scratch[0];
        });
    });
}

}  // namespace

void bf16_gemv(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    // The dispatch is the CUDA file's, threshold and all: warp-per-row is coalesced and fills the machine
    // once there are enough rows; below it the naive kernel is at least not wasting warps.
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    if (n_out >= 64) {
        launch_warp(x, w, y, n_in, n_out, q);
        check_launch("bf16_gemv(warp)");
        sync_if_needed(q, stream, "bf16_gemv(warp)");
        return;
    }
    launch_naive(x, w, y, n_in, n_out, q);
    check_launch("bf16_gemv");
    sync_if_needed(q, stream, "bf16_gemv");
}

void bf16_gemv_split(const uint16_t* x, const uint16_t* w, float* y, int64_t n_in, int64_t n_out,
                     int threads_per_row, void* stream) {
    if (n_in <= 0 || n_out <= 0) return;
    // SUB-GROUP-PER-ROW when the caller asks for 32: that path needs no local memory and no barrier.  A
    // `tpr` that is not 32 and not a power of two is refused rather than quietly rounded, because the
    // tree reduction below needs one.
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    if (threads_per_row == 32) {
        launch_warp(x, w, y, n_in, n_out, q);
        check_launch("bf16_gemv_split(warp)");
        sync_if_needed(q, stream, "bf16_gemv_split(warp)");
        return;
    }
    if (threads_per_row <= 0 || (threads_per_row & (threads_per_row - 1)) != 0) {
        std::fprintf(stderr, "bf16_gemv_split: threads_per_row %d must be a power of two (32 selects the "
                             "warp-per-row path)\n", threads_per_row);
        std::exit(1);
    }
    launch_split(x, w, y, n_in, n_out, threads_per_row, q);
    check_launch("bf16_gemv_split");
    sync_if_needed(q, stream, "bf16_gemv_split");
}

}  // namespace strata::kernels
