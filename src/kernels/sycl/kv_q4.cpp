// SYCL port of the CUDA implementation; layouts and reduction order are retained.
// src/kernels/cuda/kv_q4.cu - see include/strata/kernels/kv_q4.hpp. Q4_0 KV with Walsh-Hadamard rotation
// (from PR #21 by code-martin; KV-streaming integration and the deterministic group maximum added on merge).
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_stream.hpp"

#include "kv_helpers.hpp"
#include <cstdio>
#include <cstdlib>
#include <cuda_runtime.h>

namespace strata::kernels {
namespace {
using namespace sycl_kv;

void check(const char *what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_q4: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

// Fast Walsh-Hadamard Transform for N = 256: one warp per row, 8 values per lane in registers.
// Orthonormal (scale 1/sqrt(256) = 1/16), so it is its own inverse.
void fwht256_kernel_impl(sycl::nd_item<1> it, Grid grid, Grid block, const float *__restrict__ src,
                         float *__restrict__ dst, int64_t n_rows, float scale) {

    constexpr int warp_size = 32;
    constexpr int N = 256;
    constexpr int el_w = N / warp_size; // 8

    const int64_t r = (int64_t)group_x(it, grid) * block.y + local_y(it, block);
    if (r >= n_rows)
        return;

    const float *row_src = src + r * N;
    float *row_dst = dst + r * N;

    float reg[el_w];
    const int lane = local_x(it, block);

#pragma unroll
    for (int i = 0; i < el_w; ++i)
        reg[i] = row_src[i * warp_size + lane] * scale;

    // the low 5 index bits live across lanes
#pragma unroll
    for (int h = 1; h < warp_size; h *= 2) {
#pragma unroll
        for (int j = 0; j < el_w; ++j) {
            const float val = reg[j];
            const float val2 = xor_lane(it, val, h);
            reg[j] = (lane & h) == 0 ? val + val2 : val2 - val;
        }
    }
    // the high 3 bits across each lane's registers
#pragma unroll
    for (int h = warp_size; h < N; h *= 2) {
        const int step = h / warp_size;
#pragma unroll
        for (int j = 0; j < el_w; j += 2 * step) {
#pragma unroll
            for (int k = 0; k < step; ++k) {
                const float x = reg[j + k];
                const float y = reg[j + k + step];
                reg[j + k] = x + y;
                reg[j + k + step] = x - y;
            }
        }
    }
#pragma unroll
    for (int i = 0; i < el_w; ++i)
        row_dst[i * warp_size + lane] = reg[i];
}

void fwht256_kernel(Grid grid, Grid block, void *stream, const float *__restrict__ src,
                    float *__restrict__ dst, int64_t n_rows, float scale) {
    strata::sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(grid.size() * block.size(), block.size()),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           fwht256_kernel_impl(it, grid, block, src, dst, n_rows, scale);
                       });
    });
}

// One 32-value group in one warp (lane = element), ggml's q4_0: d = (the value of largest |x|) / -8,
// q = clamp(trunc(x / d + 8.5), 0, 15). Returns the scale bits; `byte` is lane t's packed byte for t < 16
// (element t in the low nibble, t + 16 in the high one). Ties in |x| resolve to the larger value in EVERY
// lane, so all lanes agree on d (a plain `a > amax` could leave lanes with opposite signs and one block two
// scales).
inline uint16_t q4_group(sycl::nd_item<1> it, float x, int lane, uint8_t &byte) {
    float amax = sycl::fabs(x), mval = x;
#pragma unroll
    for (int o = 16; o > 0; o >>= 1) {
        const float a = xor_lane(it, amax, o);
        const float v = xor_lane(it, mval, o);
        if (a > amax || (a == amax && v > mval)) {
            amax = a;
            mval = v;
        }
    }
    const float d = mval / -8.0f;
    const float id = d != 0.0f ? 1.0f / d : 0.0f;
    int q = (int)(x * id + 8.5f);
    const uint8_t qc = (uint8_t)(q < 0 ? 0 : (q > 15 ? 15 : q));
    const uint8_t qhi = down_lane(it, qc, 16);
    byte = (uint8_t)(qc | (qhi << 4));
    (void)lane;
    return f16_from_f32(d);
}

inline void q4_store(uint8_t *pool, long long row, int b, int lane, uint16_t d, uint8_t byte) {
    block_q4_0 *blk = reinterpret_cast<block_q4_0 *>(pool + row * (long long)sizeof(block_q4_0) * 8) + b;
    if (lane == 0)
        blk->d = d;
    if (lane < 16)
        blk->qs[lane] = byte;
}

// One block = one 32-value group of one KV head of K (blockIdx.z = 0) or V (1); 32 threads. KV streaming: the
// VRAM page only if the block is resident (table >= 0), the host copy always (identity layout) when there is
// one.
void kv_append_q4_kernel_impl(sycl::nd_item<1> it, Grid grid, Grid block, uint8_t *__restrict__ k_q4,
                              uint8_t *__restrict__ v_q4, const int32_t *__restrict__ table,
                              const int32_t *__restrict__ step, const float *__restrict__ kcur,
                              const float *__restrict__ vcur, int kv_heads, int head_dim, int page_size,
                              KvHostPools host) {

    const long long pos = (long long)(*(step + kStepPos));
    const int h = group_x(it, grid), b = group_y(it, grid), t = local_x(it, block);
    const bool is_v = group_z(it, grid) == 1;
    const float x = (is_v ? vcur : kcur)[h * head_dim + b * QK4_0 + t];
    uint8_t byte;
    const uint16_t d = q4_group(it, x, t, byte);
    const long long page = (long long)table[pos / page_size];
    if (page >= 0)
        q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, t, d, byte);
    if (host.k_q4 != nullptr)
        q4_store(is_v ? host.v_q4 : host.k_q4,
                 ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size), b, t, d, byte);
}

void kv_append_q4_kernel(Grid grid, Grid block, void *stream, uint8_t *__restrict__ k_q4,
                         uint8_t *__restrict__ v_q4, const int32_t *__restrict__ table,
                         const int32_t *__restrict__ step, const float *__restrict__ kcur,
                         const float *__restrict__ vcur, int kv_heads, int head_dim, int page_size,
                         KvHostPools host) {
    strata::sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(grid.size() * block.size(), block.size()),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           kv_append_q4_kernel_impl(it, grid, block, k_q4, v_q4, table, step, kcur, vcur,
                                                    kv_heads, head_dim, page_size, host);
                       });
    });
}

// The prompt path: grid (T, kv_heads, groups), K then V; also into the staging pool (identity layout) when
// given.
void kv_append_q4_batch_kernel_impl(sycl::nd_item<1> it, Grid grid, Grid block, uint8_t *__restrict__ k_q4,
                                    uint8_t *__restrict__ v_q4, const int32_t *__restrict__ table,
                                    int64_t pos0, const float *__restrict__ K, const float *__restrict__ V,
                                    int kv_heads, int head_dim, int page_size, int is_v_grid,
                                    KvHostPools host, KvHostPools stage) {

    const long long t = group_x(it, grid);
    const long long pos = pos0 + t;
    const int h = group_y(it, grid), b = group_z(it, grid), th = local_x(it, block);
    const bool is_v = is_v_grid != 0;
    const float x = (is_v ? V : K)[t * (kv_heads * head_dim) + h * head_dim + b * QK4_0 + th];
    uint8_t byte;
    const uint16_t d = q4_group(it, x, th, byte);
    const long long page = (long long)table[pos / page_size];
    const long long row_id = ((pos / page_size) * kv_heads + h) * page_size + (pos % page_size);
    if (page >= 0)
        q4_store(is_v ? v_q4 : k_q4, (page * kv_heads + h) * page_size + (pos % page_size), b, th, d, byte);
    if (host.k_q4 != nullptr)
        q4_store(is_v ? host.v_q4 : host.k_q4, row_id, b, th, d, byte);
    if (stage.k_q4 != nullptr)
        q4_store(is_v ? stage.v_q4 : stage.k_q4, row_id, b, th, d, byte);
}

void kv_append_q4_batch_kernel(Grid grid, Grid block, void *stream, uint8_t *__restrict__ k_q4,
                               uint8_t *__restrict__ v_q4, const int32_t *__restrict__ table, int64_t pos0,
                               const float *__restrict__ K, const float *__restrict__ V, int kv_heads,
                               int head_dim, int page_size, int is_v_grid, KvHostPools host,
                               KvHostPools stage) {
    strata::sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(grid.size() * block.size(), block.size()),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           kv_append_q4_batch_kernel_impl(it, grid, block, k_q4, v_q4, table, pos0, K, V,
                                                          kv_heads, head_dim, page_size, is_v_grid, host,
                                                          stage);
                       });
    });
}

// Gather step[kStepWidth] cells into FP16 scratch (the non-fused attention paths)
void kv_gather_q4_kernel_impl(sycl::nd_item<1> it, Grid grid, Grid block, const uint8_t *__restrict__ k_q4,
                              const uint8_t *__restrict__ v_q4, const int32_t *__restrict__ table,
                              const int32_t *__restrict__ ids, const int32_t *__restrict__ step, int kv_heads,
                              int head_dim, int page_size, uint16_t *__restrict__ k_scratch,
                              uint16_t *__restrict__ v_scratch) {

    const long long n_ids = (long long)(*(step + kStepWidth));
    const int blocks_per_head = head_dim / QK4_0;                    // 8
    const int bytes_per_head = blocks_per_head * sizeof(block_q4_0); // 144
    const long long total_blocks = n_ids * kv_heads * blocks_per_head;

    const long long blk_idx = (long long)group_x(it, grid) * block.y + local_y(it, block);
    if (blk_idx >= total_blocks)
        return;

    const int t = local_x(it, block);
    const long long id = blk_idx / (kv_heads * blocks_per_head);
    const int rem = (int)(blk_idx % (kv_heads * blocks_per_head));
    const int h = rem / blocks_per_head;
    const int b = rem % blocks_per_head;

    const int cell = ids[id];
    const long long page = (long long)table[cell / page_size];
    const long long row = (page * kv_heads + h) * page_size + (cell % page_size);

    const block_q4_0 *k_blk = reinterpret_cast<const block_q4_0 *>(k_q4 + row * bytes_per_head) + b;
    const block_q4_0 *v_blk = reinterpret_cast<const block_q4_0 *>(v_q4 + row * bytes_per_head) + b;
    const float kd = f32_from_f16(k_blk->d);
    const float vd = f32_from_f16(v_blk->d);
    const int j = t < 16 ? t : (t - 16);
    const uint8_t k_byte = k_blk->qs[j];
    const uint8_t v_byte = v_blk->qs[j];
    const int kq = (t < 16) ? ((k_byte & 0x0F) - 8) : ((k_byte >> 4) - 8);
    const int vq = (t < 16) ? ((v_byte & 0x0F) - 8) : ((v_byte >> 4) - 8);
    const long long dst_offset = ((id * kv_heads + h) * head_dim) + (b * QK4_0 + t);
    k_scratch[dst_offset] = f16_from_f32((float)kq * kd);
    v_scratch[dst_offset] = f16_from_f32((float)vq * vd);
}

void kv_gather_q4_kernel(Grid grid, Grid block, void *stream, const uint8_t *__restrict__ k_q4,
                         const uint8_t *__restrict__ v_q4, const int32_t *__restrict__ table,
                         const int32_t *__restrict__ ids, const int32_t *__restrict__ step, int kv_heads,
                         int head_dim, int page_size, uint16_t *__restrict__ k_scratch,
                         uint16_t *__restrict__ v_scratch) {
    strata::sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(grid.size() * block.size(), block.size()),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           kv_gather_q4_kernel_impl(it, grid, block, k_q4, v_q4, table, ids, step, kv_heads,
                                                    head_dim, page_size, k_scratch, v_scratch);
                       });
    });
}

void need_256(const QsaShapes &s, const char *what) {
    if (s.head_dim != 256) {
        std::fprintf(stderr, "%s: head_dim must be 256 (the Hadamard transform's size)\n", what);
        std::exit(1);
    }
}

} // namespace

void fwht256_cuda(const float *src, float *dst, int64_t n_rows, void *stream) {
    if (n_rows <= 0)
        return;
    const int rows_per_block = 4;
    const int64_t num_blocks = (n_rows + rows_per_block - 1) / rows_per_block;
    fwht256_kernel(Grid((unsigned)num_blocks), Grid(32, rows_per_block), (cudaStream_t)stream, src, dst,
                   n_rows, 1.0f / 16.0f);
    check("fwht256 launch");
}

void kv_append_q4_step(uint8_t *k_q4, uint8_t *v_q4, const int32_t *page_table, const int32_t *step,
                       const float *kcur, const float *vcur, const QsaShapes &s, void *stream,
                       const KvHostPools *host) {
    need_256(s, "kv_append_q4");
    const Grid grid((unsigned)s.n_head_kv, (unsigned)(s.head_dim / QK4_0), 2);
    kv_append_q4_kernel(grid, 32, (cudaStream_t)stream, k_q4, v_q4, page_table, step, kcur, vcur,
                        (int)s.n_head_kv, (int)s.head_dim, (int)s.page_size, host ? *host : KvHostPools{});
    check("kv_append_q4 launch");
}

void kv_append_q4(uint8_t *k_q4, uint8_t *v_q4, const int32_t *page_table, int64_t pos0, int64_t T,
                  const float *K, const float *V, const QsaShapes &s, void *stream, const KvHostPools *host,
                  const KvHostPools *stage) {
    if (T <= 0)
        return;
    need_256(s, "kv_append_q4");
    const Grid grid((unsigned)T, (unsigned)s.n_head_kv, (unsigned)(s.head_dim / QK4_0));
    cudaStream_t cs = (cudaStream_t)stream;
    const KvHostPools h = host ? *host : KvHostPools{}, st = stage ? *stage : KvHostPools{};
    for (int is_v = 0; is_v < 2; ++is_v)
        kv_append_q4_batch_kernel(grid, 32, cs, k_q4, v_q4, page_table, pos0, K, V, (int)s.n_head_kv,
                                  (int)s.head_dim, (int)s.page_size, is_v, h, st);
    check("kv_append_q4 batch launch");
}

void kv_gather_q4_step(const uint8_t *k_q4, const uint8_t *v_q4, const int32_t *page_table,
                       const int32_t *ids, const int32_t *step, int64_t max_ids, const QsaShapes &s,
                       uint16_t *k_scratch, uint16_t *v_scratch, void *stream) {
    if (max_ids <= 0)
        return;
    const int blocks_per_head = (int)(s.head_dim / QK4_0);
    const int64_t total_blocks = max_ids * s.n_head_kv * blocks_per_head;
    const int rows_per_block = 4;
    const unsigned num_blocks = (unsigned)((total_blocks + rows_per_block - 1) / rows_per_block);
    kv_gather_q4_kernel(Grid(num_blocks), Grid(32, rows_per_block), (cudaStream_t)stream, k_q4, v_q4,
                        page_table, ids, step, (int)s.n_head_kv, (int)s.head_dim, (int)s.page_size, k_scratch,
                        v_scratch);
    check("kv_gather_q4 launch");
}

} // namespace strata::kernels
