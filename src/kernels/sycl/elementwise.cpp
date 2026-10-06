// src/kernels/sycl/elementwise.cpp - SYCL port of src/kernels/cuda/elementwise.cu
// (P2.S5's glue kernels).  The CUDA file and the shared header carry each kernel's
// contract; only what SYCL forces to change is noted here:
//
//   * `__fmul_rn` / `__fadd_rn`  -> `sycl::ext::intel::math::{fmul_rn,fadd_rn}`
//     (embedding_gather pins the product/add rounding exactly like the CUDA path).
//   * warp shuffles               -> `sycl::sub_group` shuffles under a required
//     32-wide sub-group (the runtime refuses devices without it).
//   * `__threadfence_system()`    -> `sycl::atomic_fence(seq_cst, system)`: the
//     doorbell's ordering guarantee toward the HOST reader, the whole point of the
//     fence (see the CUDA file's doorbell comment - it is not decoration).
//   * `__shared__` + `__syncthreads()` -> a local accessor + `group_barrier`.
//   * reads of mapped host memory the HOST updates (the doorbell flag, the CPU pool's rows) are uncached loads,
//     and the doorbell's own counter is an uncached store (mapped_host.hpp: on the A770 a running kernel does not
//     see a host store through a volatile or atomic load); `float4` copies are four scalar loads each.
#include "strata/kernels/elementwise.hpp"
#include "strata/kernels/dp4a.hpp"

#include "strata/kernels/bf16_bits.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
#include "fp_exact.hpp"
#include "mapped_host.hpp"

#include <cuda_runtime.h>

#include <sycl/ext/intel/math.hpp>

#include <cmath>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {

namespace intel_math = sycl::ext::intel::math;

void embedding_gather_item(int64_t i, const uint8_t* codes, const float* scales,
                           const float* offsets, int64_t n, int code_bits, int code_bias,
                           int group_elems, float* out) {
    if (i >= n) return;
    const int per_byte = 8 / code_bits;
    const unsigned mask = (1u << code_bits) - 1u;
    const int code = (codes[i / per_byte] >> ((i % per_byte) * code_bits)) & mask;
    const int64_t group = i / group_elems;
    const float product = intel_math::fmul_rn((float) (code + code_bias), scales[group]);
    out[i] = intel_math::fadd_rn(product, offsets ? offsets[group] : 0.0f);
}

/// `ggml_compute_softplus_f32`: `log1p(exp(x))`, with the large-x branch that avoids overflow.
/// Above 20 the function is `x` to within f32 anyway (the branch is not cosmetic).
float softplus_dev(float x) { return x > 20.0f ? x : sycl::log1p(sycl::exp(x)); }

void gdn_gate_item(int64_t i, const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t h_v) {
    if (i >= h_v) return;
    const int64_t t = i / h_v;      // `n_tokens` is the leading dim; the real call has one token
    const int64_t h = i % h_v;
    gate[i] = softplus_dev(alpha[i] + dt[h]) * ssm_a[h];
    (void) t;
}

void silu_item(int64_t i, float* x, int64_t n) {
    if (i >= n) return;
    // DOUBLE then cast, matching `ref/gdn.py`'s numpy (the reference is the oracle).
    const double v = (double) x[i];
    x[i] = (float) (v / (1.0 + sycl::exp(-v)));
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

bool check_launch(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s launch: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
    return true;
}

// A flat parallel_for over n items: the CUDA `grid_for(n), THREADS` shape.
template <typename F>
void launch_items(int64_t n, void* stream, F&& body) {
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    q.parallel_for(sycl::range<1>(static_cast<size_t>(n)), [=](sycl::id<1> idx) {
        body(static_cast<int64_t>(idx[0]));
    });
}

}  // namespace

void embedding_gather(const uint8_t* codes, const float* scales, const float* offsets,
                      int64_t n, int code_bits, int code_bias, int group_elems,
                      float* out, void* stream) {
    if (n <= 0) return;
    launch_items(n, stream, [=](int64_t i) {
        embedding_gather_item(i, codes, scales, offsets, n, code_bits, code_bias, group_elems, out);
    });
    check_launch("embedding_gather");
}

void gdn_gate(const float* alpha, const float* dt, const float* ssm_a, float* gate, int64_t n_tokens,
              int64_t h_v, void* stream) {
    if (n_tokens <= 0 || h_v <= 0) return;
    const int64_t n = n_tokens * h_v;
    launch_items(n, stream, [=](int64_t i) { gdn_gate_item(i, alpha, dt, ssm_a, gate, h_v); });
    check_launch("gdn_gate");
    sync_if_needed(sycl_runtime::queue_from_stream(stream), stream, "gdn_gate");
}

void scale_inplace(float* x, int64_t n, float s, void* stream) {
    if (n <= 0) return;
    launch_items(n, stream, [=](int64_t i) {
        if (i < n) x[i] *= s;
    });
    check_launch("scale_inplace");
    sync_if_needed(sycl_runtime::queue_from_stream(stream), stream, "scale_inplace");
}

void add_inplace(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    launch_items(n, stream, [=](int64_t i) {
        if (i < n) dst[i] += src[i];
    });
    check_launch("add_inplace");
    sync_if_needed(sycl_runtime::queue_from_stream(stream), stream, "add_inplace");
}

void f32_to_f16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    launch_items(n, stream, [=](int64_t i) {
        if (i < n) y[i] = f16_from_f32(x[i]);
    });
    check_launch("f32_to_f16_bulk");
    sync_if_needed(sycl_runtime::queue_from_stream(stream), stream, "f32_to_f16_bulk");
}

void f32_to_bf16_bulk(const float* x, uint16_t* y, int64_t n, void* stream) {
    if (n <= 0) return;
    launch_items(n, stream, [=](int64_t i) {
        if (i < n) y[i] = bf16_from_f32(x[i]);
    });
    check_launch("f32_to_bf16_bulk");
    sync_if_needed(sycl_runtime::queue_from_stream(stream), stream, "f32_to_bf16_bulk");
}

void silu_inplace(float* x, int64_t n, void* stream) {
    if (n <= 0) return;
    launch_items(n, stream, [=](int64_t i) { silu_item(i, x, n); });
    check_launch("silu_inplace");
    sync_if_needed(sycl_runtime::queue_from_stream(stream), stream, "silu_inplace");
}

/// THE DOORBELL.  One work-item, one INCREMENT of the memory (the value is never an
/// argument - a captured graph must ring the CURRENT value on every replay, not a
/// host literal frozen at capture time).  The system-scope fence before the
/// increment orders it after the payload writes it publishes, toward the CPU reader;
/// see the CUDA file's doorbell comment for what the missing fence cost.
void doorbell_ring(uint32_t* d_seq, void* stream) {
    if (d_seq == nullptr) return;
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    q.single_task([=] {
        sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        sycl_mapped::store(d_seq, sycl_mapped::load(d_seq) + 1u);
    });
    check_launch("doorbell_ring");
    sync_if_needed(q, stream, "doorbell_ring");
}

void doorbell_wait(const uint32_t* d_flag, const uint32_t* d_seq, void* stream) {
    if (d_flag == nullptr || d_seq == nullptr) return;
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    q.single_task([=] {
        const uint32_t want = sycl_mapped::load(d_seq);
        sycl_mapped::spin_until(d_flag, [=](uint32_t v) { return v == want; });
        sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
    });
    check_launch("doorbell_wait");
}

void copy_from_mapped(float* dst, const float* src, int64_t n, void* stream) {
    if (n <= 0) return;
    if ((n & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_from_mapped: n must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    launch_items(n / 4, stream, [=](int64_t i4) {   // 64-bit uncached loads (see copy_rows_from_mapped)
        const uint64_t* s = (const uint64_t*) (src + i4 * 4);
        uint64_t* d = (uint64_t*) (dst + i4 * 4);
        d[0] = sycl_mapped::load(s, 0);
        d[1] = sycl_mapped::load(s, 1);
    });
    check_launch("copy_from_mapped");
}

// the CPU rows of a verify window, skipping the rows the GPU plan computes itself (the
// pool writes +0.0 into those, so this writes +0.0 too): one work-group per row.
void copy_rows_from_mapped(float* dst, const float* src, int64_t rows, int64_t width, const int32_t* hit_rows,
                           const int32_t* count, void* stream) {
    if (rows <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "copy_rows_from_mapped: width must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    const int64_t width4 = width / 4;
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<int, 1> hit(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(rows) * 128), sycl::range<1>(128)),
                       [=](sycl::nd_item<1> it) {
            const int row = static_cast<int>(it.get_group(0));
            const int lane = static_cast<int>(it.get_local_id(0));
            if (lane == 0) {
                int hv = 0;
                const int c = *count;
                for (int i = 0; i < c; ++i) hv |= hit_rows[i] == row;
                hit[0] = hv;
            }
            sycl::group_barrier(it.get_group());
            // 64-bit uncached loads: each one is a PCIe round trip, so pairs of floats halve the time (0.35 -> 0.19 ms
            // for a 4-token window's rows on the A770)
            uint64_t* d = (uint64_t*) (dst + (int64_t) row * width);
            if (hit[0]) {
                for (int64_t i = lane; i < width / 2; i += 128) d[i] = 0;
            } else {
                const uint64_t* s = (const uint64_t*) (src + (int64_t) row * width);
                for (int64_t i = lane; i < width / 2; i += 128) d[i] = sycl_mapped::load(s, (size_t) i);
            }
        });
    });
    check_launch("copy_rows_from_mapped");
}

void scatter_rows_f32(const float* src, float* dst, const int32_t* rows, int64_t n, int64_t width, void* stream) {
    if (n <= 0) return;
    if ((width & 3) != 0 || ((uintptr_t) dst & 15) != 0 || ((uintptr_t) src & 15) != 0) {
        std::fprintf(stderr, "scatter_rows_f32: width must be a multiple of 4 and both pointers 16-byte aligned\n");
        std::exit(1);
    }
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(static_cast<size_t>(n) * 128), sycl::range<1>(128)),
        [=](sycl::nd_item<1> it) {
            const int64_t r = static_cast<int64_t>(it.get_group(0));
            const float* s = src + r * width;
            float* d = dst + (int64_t) rows[r] * width;
            for (int64_t i = it.get_local_id(0); i < width; i += 128) d[i] = s[i];
        });
    check_launch("scatter_rows_f32");
}

namespace {
inline uint32_t payload_sum(const float* x, int64_t n, const int32_t* ids, const float* weights, int64_t k) {
    uint32_t s = 0;
    for (int64_t j = 0; j < n; ++j) s += sycl::bit_cast<uint32_t>(((const volatile float*) x)[j]);
    for (int64_t j = 0; j < k; ++j) s += (uint32_t) ((const volatile int32_t*) ids)[j];
    for (int64_t j = 0; j < k; ++j) s += sycl::bit_cast<uint32_t>(((const volatile float*) weights)[j]);
    return s;
}
}  // namespace

bool doorbell_payload_ready(const uint32_t* h_seq, const float* h_x, int64_t n, const int32_t* h_ids,
                            const float* h_weights, int64_t k, uint32_t want) {
    const volatile uint32_t* w = h_seq + 1 + 2 * (want % 4);   // ring r's tag and checksum (elementwise.hpp)
    if (w[0] != want) return false;
    std::atomic_thread_fence(std::memory_order_acquire);
    return payload_sum(h_x, n, h_ids, h_weights, k) == w[1];
}

bool doorbell_wait_payload(const uint32_t* h_seq, const float* h_x, int64_t n, const int32_t* h_ids,
                           const float* h_weights, int64_t k, uint32_t want, int timeout_ms) {
    if (doorbell_payload_ready(h_seq, h_x, n, h_ids, h_weights, k, want)) return true;
    const auto t0 = std::chrono::steady_clock::now();
    while (!doorbell_payload_ready(h_seq, h_x, n, h_ids, h_weights, k, want)) {
        for (int i = 0; i < 64; ++i) __builtin_ia32_pause();
        if (std::chrono::steady_clock::now() - t0 > std::chrono::milliseconds(timeout_ms)) {
            const volatile uint32_t* w = h_seq + 1 + 2 * (want % 4);
            std::fprintf(stderr, "doorbell_wait_payload: ring %u: seq %u, tag %u, GPU checksum %08x, host checksum %08x "
                                 "(n %lld, k %lld)\n", want, h_seq[0], w[0], w[1], payload_sum(h_x, n, h_ids, h_weights, k),
                         (long long) n, (long long) k);
            return false;
        }
    }
    return true;
}

void doorbell_publish(const float* x, const int32_t* ids, const float* weights, int64_t n, int64_t k, float* x_out,
                      int32_t* ids_out, float* weights_out, uint32_t* d_seq, void* stream) {
    if (k > 1024) { std::fprintf(stderr, "doorbell_publish: k too large\n"); std::exit(1); }
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    // the payload with uncached stores (mapped_host.hpp: a plain store reaches the host only when the kernel ends, and
    // in a captured window the next kernel spins first), then its checksum and ring number, then the ring.  The host
    // still checks the payload against the checksum (doorbell_payload_ready): the fences do not keep the ring from
    // overtaking the last activation stores.
    // 64-bit stores where the layout allows: uncached stores drain at a fixed rate per store on the A770 (PCIe 3.0 x4:
    // ~12 million per second), so pairs of floats halve the payload's time (measured 0.93 -> 0.44 ms for 4 tokens)
    const bool pairs = n % 2 == 0 && ((uintptr_t) x % 8) == 0 && ((uintptr_t) x_out % 8) == 0;
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(1024), sycl::range<1>(1024)), [=](sycl::nd_item<1> it) {
        const int i = static_cast<int>(it.get_local_id(0));
        uint32_t part = 0;
        if (pairs) {
            for (int j = i; j < (int) n / 2; j += 1024) {
                const uint64_t v = ((const uint64_t*) x)[j];
                sycl_mapped::store((uint64_t*) x_out, v, (size_t) j);
                part += (uint32_t) v + (uint32_t) (v >> 32);
            }
        } else {
            for (int j = i; j < (int) n; j += 1024) {
                const float v = x[j];
                sycl_mapped::store(x_out, v, (size_t) j);
                part += sycl::bit_cast<uint32_t>(v);
            }
        }
        if (i < (int) k) {
            const int32_t id = ids[i];
            const float wv = weights[i];
            sycl_mapped::store(ids_out, id, (size_t) i);
            sycl_mapped::store(weights_out, wv, (size_t) i);
            part += (uint32_t) id + sycl::bit_cast<uint32_t>(wv);
        }
        const uint32_t sum = sycl::reduce_over_group(it.get_group(), part, sycl::plus<uint32_t>());
        sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
        sycl::group_barrier(it.get_group());
        if (i == 0) {
            const uint32_t next = sycl_mapped::load(d_seq) + 1u;
            // a slot per ring modulo 4: a split window publishes two rings before the host reads the first
            sycl_mapped::store(d_seq, sum, 2 + 2 * (next % 4));
            sycl_mapped::store(d_seq, next, 1 + 2 * (next % 4));
            sycl::atomic_fence(sycl::memory_order::seq_cst, sycl::memory_scope::system);
            sycl_mapped::store(d_seq, next);
        }
    });
    check_launch("doorbell_publish");
}

void copy_i32_from_mapped(int32_t* dst, const int32_t* src, int64_t n, void* stream) {
    if (n <= 0) return;
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    q.parallel_for(sycl::nd_range<1>(sycl::range<1>(128), sycl::range<1>(128)), [=](sycl::nd_item<1> it) {
        for (int64_t i = it.get_local_id(0); i < n; i += 128) dst[i] = sycl_mapped::load(src, (size_t) i);
    });
    check_launch("copy_i32_from_mapped");
}

/// One SUB-GROUP per row, reduced through sub-group shuffles - the CUDA file's
/// one-warp-per-row shape and its reasoning (rows are short and few; a block-per-row
/// tree would spend its time in barriers) carry over unchanged, as does the row
/// guard whose absence was the QSA bug (see the CUDA file).
void rms_norm_weighted(float* x, const float* w, int64_t rows, int64_t cols, float eps, void* stream) {
    if (rows <= 0 || cols <= 0) return;
    // 4 sub-groups per work-group, so a row count that is not a multiple of 4 wastes
    // at most 3 sub-groups rather than launching a work-group per row.
    const size_t groups = static_cast<size_t>((rows + 3) / 4);
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    q.parallel_for(
        sycl::nd_range<1>(sycl::range<1>(groups * 128), sycl::range<1>(128)),
        [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = static_cast<int>(sg.get_local_linear_id());
            const int64_t row = static_cast<int64_t>(it.get_group(0)) * 4 + sg.get_group_id();
            if (row >= rows) return;
            float* r = x + row * cols;
            float acc = 0.0f;
            for (int64_t c = lane; c < cols; c += 32) acc += r[c] * r[c];
            for (int off = 16; off > 0; off >>= 1)
                acc += sycl::shift_group_left(sg, acc, static_cast<uint32_t>(off));
            // The MEAN, not the sum; the reciprocal is broadcast from lane 0 so all
            // 32 lanes multiply by the same value (see the CUDA file).
            float inv = 0.0f;
            if (lane == 0) inv = sycl::rsqrt(acc / (float) cols + eps);
            inv = sycl::group_broadcast(sg, inv, 0);
            for (int64_t c = lane; c < cols; c += 32) r[c] = (w ? r[c] * w[c] : r[c]) * inv;
        });
    check_launch("rms_norm_weighted");
    sync_if_needed(q, stream, "rms_norm_weighted");
}

}  // namespace strata::kernels
