// SYCL port of the CUDA implementation; layouts and reduction order are retained.
// src/kernels/cuda/kv_stream.cu - see include/strata/kernels/kv_stream.hpp.
#include "strata/kernels/kv_stream.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"

#include "kv_helpers.hpp"
#include <cuda_runtime.h>

#include <algorithm>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {
namespace {
using namespace sycl_kv;

void check(const char *what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) {
        std::fprintf(stderr, "kv_stream: %s: %s\n", what, cudaGetErrorString(e));
        std::exit(1);
    }
}

constexpr int RT = 1024; // the resolve block

// The per-block byte runs of the (up to four) pool arrays: block b of array i is bytes [b * len, (b + 1) *
// len).
struct Runs {
    const uint8_t *src[4];
    uint8_t *dst[4];
    int len[4];
    int n;
};

Runs runs_of(const QsaAttnPools &slots, const KvHostPools &host, int fmt, const QsaShapes &s) {
    const int rows = (int)(s.n_head_kv * s.page_size);
    Runs r{};
    if (fmt == kKvQ4) {
        const int bytes = rows * (int)kv_q4_bytes_per_head((int)s.head_dim);
        r.src[0] = (const uint8_t *)host.k_q4;
        r.dst[0] = (uint8_t *)slots.k_q4;
        r.len[0] = bytes;
        r.src[1] = (const uint8_t *)host.v_q4;
        r.dst[1] = (uint8_t *)slots.v_q4;
        r.len[1] = bytes;
        r.n = 2;
    } else if (fmt == kKvInt8) {
        const int codes = rows * (int)s.head_dim, scales = rows * (int)(s.head_dim / KV_Q8_GROUP) * 2;
        r.src[0] = (const uint8_t *)host.k_q;
        r.dst[0] = (uint8_t *)slots.k_q;
        r.len[0] = codes;
        r.src[1] = (const uint8_t *)host.v_q;
        r.dst[1] = (uint8_t *)slots.v_q;
        r.len[1] = codes;
        r.src[2] = (const uint8_t *)host.k_scale;
        r.dst[2] = (uint8_t *)slots.k_scale;
        r.len[2] = scales;
        r.src[3] = (const uint8_t *)host.v_scale;
        r.dst[3] = (uint8_t *)slots.v_scale;
        r.len[3] = scales;
        r.n = 4;
    } else {
        const int bytes = rows * (int)s.head_dim * 2;
        r.src[0] = (const uint8_t *)host.k_pool;
        r.dst[0] = (uint8_t *)slots.k_pool;
        r.len[0] = bytes;
        r.src[1] = (const uint8_t *)host.v_pool;
        r.dst[1] = (uint8_t *)slots.v_pool;
        r.len[1] = bytes;
        r.n = 2;
    }
    return r;
}

// Block-wide exclusive prefix sum of one int per thread (RT threads); `total` is the sum over the block.
inline int block_scan(sycl::nd_item<1> it, int v, int *warp_sums, int &total) {
    const int lane = int(it.get_local_id(0)) & 31, w = int(it.get_local_id(0)) >> 5;
    int x = v;
    for (int o = 1; o < 32; o <<= 1) {
        const int y = up_lane(it, x, o);
        if (lane >= o)
            x += y;
    }
    if (lane == 31)
        warp_sums[w] = x;
    it.barrier(sycl::access::fence_space::global_and_local);
    if (w == 0) {
        int t = warp_sums[lane];
        for (int o = 1; o < 32; o <<= 1) {
            const int y = up_lane(it, t, o);
            if (lane >= o)
                t += y;
        }
        warp_sums[lane] = t;
    }
    it.barrier(sycl::access::fence_space::global_and_local);
    total = warp_sums[31];
    const int excl = x - v + (w > 0 ? warp_sums[w - 1] : 0);
    it.barrier(sycl::access::fence_space::global_and_local);
    return excl;
}

void resolve_kernel_impl(sycl::nd_item<1> it, [[maybe_unused]] Grid grid, Grid block, KvStreamMap m,
                         const int32_t *__restrict__ ids, const int32_t *__restrict__ steps, int n_q, int cap,
                         int page_size, int &s_nmiss, int &s_lookups, int &s_cut, int *ptr_warp_sums) {
    int *warp_sums = ptr_warp_sums;

    const int epoch = m.ctl[0] + 1;
    if (local_x(it, block) == 0) {
        s_nmiss = 0;
        s_lookups = 0;
    }
    it.barrier(sycl::access::fence_space::global_and_local);
    // 1. hits take this epoch and their reference bit; a missing block is claimed exactly once (-1 -> -2)
    int lookups = 0;
    for (int q = 0; q < n_q; ++q) {
        const int width = steps[q * kStepCount + kStepWidth];
        const int32_t *qi = ids + (long long)q * cap;
        for (int i = local_x(it, block); i < width; i += RT) {
            const int b = qi[i] / page_size;
            if (i > 0 && qi[i - 1] / page_size == b)
                continue; // ids are ascending: one lookup per block
            ++lookups;
            const int sl = m.page_table[b];
            if (sl >= 0) {
                m.slot_stamp[sl] = epoch;
                m.slot_ref[sl] = 1;
            } else if (sl == -1 && atomic_cas(&m.page_table[b], -1, -2) == -1) {
                m.miss_block[atomic_add(&s_nmiss, 1)] = b;
            }
        }
    }
    atomic_add(&s_lookups, lookups);
    it.barrier(sycl::access::fence_space::global_and_local);
    // 2. one victim per miss: a clock sweep from the hand. A slot this call uses (stamp == epoch) is never
    // taken;
    //    a referenced one loses its bit as the hand passes it and is taken on the next pass.
    const int need = s_nmiss, n = (int)m.n_slots;
    int hand = m.ctl[1], got = 0;
    for (int scanned = 0; got < need && scanned < 3 * n; scanned += RT) {
        const int j = (int)(((long long)hand + local_x(it, block)) % n);
        const bool mine = m.slot_stamp[j] == epoch;
        const bool cand = !mine && (m.slot_block[j] < 0 || m.slot_ref[j] == 0);
        int total = 0;
        const int rank = block_scan(it, cand ? 1 : 0, warp_sums, total);
        const int want = need - got;
        if (local_x(it, block) == 0)
            s_cut = RT;
        it.barrier(sycl::access::fence_space::global_and_local);
        if (cand && rank == want - 1)
            s_cut = local_x(it, block) + 1; // the hand stops just past the last slot taken
        it.barrier(sycl::access::fence_space::global_and_local);
        const int cut = s_cut;
        if (cand && rank < want) {
            m.miss_slot[got + rank] = j;
            m.slot_stamp[j] = epoch; // taken: a sweep that wraps around must not take it twice
        } else if (local_x(it, block) < cut && !mine) {
            m.slot_ref[j] = 0;
        }
        got += total < want ? total : want;
        hand = (int)(((long long)hand + cut) % n);
        it.barrier(sycl::access::fence_space::global_and_local);
    }
    // 3. re-point the table; the copy kernel fills the slots
    const int placed = got < need ? got : need;
    for (int k = local_x(it, block); k < need; k += RT) {
        const int b = m.miss_block[k];
        if (k >= placed) {
            m.page_table[b] = -1;
            continue;
        } // overflow: never happens with a legal n_slots
        const int sl = m.miss_slot[k];
        const int old = m.slot_block[sl];
        if (old >= 0)
            m.page_table[old] = -1;
        m.slot_block[sl] = b;
        m.slot_stamp[sl] = epoch;
        m.slot_ref[sl] = 1;
        m.page_table[b] = sl;
    }
    if (local_x(it, block) == 0) {
        m.ctl[0] = epoch;
        m.ctl[1] = hand;
        m.ctl[2] = placed;
        if (placed < need)
            m.ctl[3] = 1;
        unsigned long long *c = reinterpret_cast<unsigned long long *>(m.ctl + 4);
        c[0] += (unsigned long long)placed;
        c[1] += (unsigned long long)s_lookups;
        c[2] += 1ull;
    }
}

void resolve_kernel(Grid grid, Grid block, void *stream, KvStreamMap m, const int32_t *__restrict__ ids,
                    const int32_t *__restrict__ steps, int n_q, int cap, int page_size) {
    strata::sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler &h) {
        sycl::local_accessor<int, 1> l_s_nmiss(sycl::range<1>(1), h);
        sycl::local_accessor<int, 1> l_s_lookups(sycl::range<1>(1), h);
        sycl::local_accessor<int, 1> l_s_cut(sycl::range<1>(1), h);
        sycl::local_accessor<int, 1> l_warp_sums(sycl::range<1>((32)), h);
        h.parallel_for(sycl::nd_range<1>(grid.size() * block.size(), block.size()),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           resolve_kernel_impl(
                               it, grid, block, m, ids, steps, n_q, cap, page_size, l_s_nmiss[0],
                               l_s_lookups[0], l_s_cut[0],
                               l_warp_sums.template get_multi_ptr<sycl::access::decorated::no>().get());
                       });
    });
}

// One block per missed block (grid-stride): copy its runs from the host copy into its slot, 16 B per thread.
void copy_kernel_impl(sycl::nd_item<1> it, Grid grid, Grid block, KvStreamMap m, Runs r) {

    const int need = m.ctl[2];
    for (int k = group_x(it, grid); k < need; k += grid.x) {
        const long long b = m.miss_block[k], sl = m.miss_slot[k];
        for (int a = 0; a < r.n; ++a) {
            const uint4 *src = reinterpret_cast<const uint4 *>(r.src[a] + b * r.len[a]);
            uint4 *dst = reinterpret_cast<uint4 *>(r.dst[a] + sl * r.len[a]);
            for (int i = local_x(it, block); i < r.len[a] / 16; i += block.x)
                dst[i] = src[i];
        }
    }
}

void copy_kernel(Grid grid, Grid block, void *stream, KvStreamMap m, Runs r) {
    strata::sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(grid.size() * block.size(), block.size()),
                       [=](sycl::nd_item<1> it)
                           [[sycl::reqd_sub_group_size(32)]] { copy_kernel_impl(it, grid, block, m, r); });
    });
}

void reset_kernel_impl(sycl::nd_item<1> it, Grid grid, Grid block, KvStreamMap m) {

    const long long i0 = (long long)group_x(it, grid) * block.x + local_x(it, block),
                    st = (long long)grid.x * block.x;
    for (long long i = i0; i < m.n_blocks; i += st)
        m.page_table[i] = -1;
    for (long long i = i0; i < m.n_slots; i += st) {
        m.slot_block[i] = -1;
        m.slot_stamp[i] = -1;
        m.slot_ref[i] = 0;
    }
    if (i0 < kKvCtlInts)
        m.ctl[i0] = 0;
}

void reset_kernel(Grid grid, Grid block, void *stream, KvStreamMap m) {
    strata::sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(grid.size() * block.size(), block.size()),
                       [=](sycl::nd_item<1> it)
                           [[sycl::reqd_sub_group_size(32)]] { reset_kernel_impl(it, grid, block, m); });
    });
}

void ring_kernel_impl(sycl::nd_item<1> it, Grid grid, Grid block, int32_t *table, long long n_blocks,
                      long long n_slots) {

    for (long long i = (long long)group_x(it, grid) * block.x + local_x(it, block); i < n_blocks;
         i += (long long)grid.x * block.x)
        table[i] = (int32_t)(i % n_slots);
}

void ring_kernel(Grid grid, Grid block, void *stream, int32_t *table, long long n_blocks, long long n_slots) {
    strata::sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler &h) {
        h.parallel_for(sycl::nd_range<1>(grid.size() * block.size(), block.size()),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           ring_kernel_impl(it, grid, block, table, n_blocks, n_slots);
                       });
    });
}

} // namespace

uint64_t kv_block_bytes(const QsaShapes &s, int fmt) {
    const uint64_t rows = (uint64_t)(s.n_head_kv * s.page_size);
    if (fmt == kKvQ4)
        return rows * kv_q4_bytes_per_head((int)s.head_dim) * 2;
    return fmt == kKvInt8
               ? rows * (uint64_t)s.head_dim * 2 + rows * (uint64_t)(s.head_dim / KV_Q8_GROUP) * 2 * 2
               : rows * (uint64_t)s.head_dim * 2 * 2;
}

void kv_stream_reset(const KvStreamMap &m, void *stream) {
    reset_kernel(128, 256, (cudaStream_t)stream, m);
    check("reset");
}

void kv_stream_resolve(const KvStreamMap &m, const QsaAttnPools &slots, const KvHostPools &host, int fmt,
                       const int32_t *ids, const int32_t *steps, int64_t n_q, int64_t cap, const QsaShapes &s,
                       void *stream) {
    if (n_q <= 0)
        return;
    if (s.n_head_kv * s.page_size * (s.head_dim / KV_Q8_GROUP) * 2 % 16 != 0) {
        std::fprintf(stderr, "kv_stream: a block's scale run must be a multiple of 16 bytes\n");
        std::exit(1);
    }
    // one sweep step looks at RT consecutive slots `(hand + thread) % n_slots`; with fewer slots than RT two
    // threads see the same slot and may both take it for two different misses.  The engine never streams with
    // fewer than qsa_kv_resident_min() / page_size = 5,120 slots, so this is a guard, not a limit.
    if (m.n_slots < RT) {
        std::fprintf(
            stderr,
            "kv_stream: %lld slots is fewer than the resolve block (%d): the clock sweep would take a "
            "slot twice\n",
            (long long)m.n_slots, RT);
        std::exit(1);
    }
    resolve_kernel(1, RT, (cudaStream_t)stream, m, ids, steps, (int)n_q, (int)cap, (int)s.page_size);
    check("resolve");
    copy_kernel(96, 128, (cudaStream_t)stream, m, runs_of(slots, host, fmt, s));
    check("copy");
}

void kv_ring_table(int32_t *page_table, int64_t n_blocks, int64_t n_slots, void *stream) {
    ring_kernel(64, 256, (cudaStream_t)stream, page_table, n_blocks, n_slots);
    check("ring table");
}

void kv_ring_restore(const QsaAttnPools &slots, const KvHostPools &host, int fmt, int64_t b0, int64_t b1,
                     int64_t n_slots, const QsaShapes &s, void *stream) {
    const Runs r = runs_of(slots, host, fmt, s);
    for (int64_t b = b0; b < b1;) {
        const int64_t sl = b % n_slots, run = std::min<int64_t>(b1 - b, n_slots - sl); // up to the ring's end
        for (int a = 0; a < r.n; ++a)
            if (cudaMemcpyAsync(r.dst[a] + sl * r.len[a], r.src[a] + b * r.len[a], (size_t)(run * r.len[a]),
                                cudaMemcpyDefault, (cudaStream_t)stream) != cudaSuccess)
                check("ring restore");
        b += run;
    }
}

void kv_stage_from_host(const QsaAttnPools &stage, const KvHostPools &host, int fmt, int64_t n_blocks,
                        const QsaShapes &s, void *stream) {
    if (n_blocks <= 0)
        return;
    const Runs r = runs_of(stage, host, fmt, s);
    for (int a = 0; a < r.n; ++a)
        if (cudaMemcpyAsync(r.dst[a], r.src[a], (size_t)(n_blocks * r.len[a]), cudaMemcpyDefault,
                            (cudaStream_t)stream) != cudaSuccess)
            check("stage");
}

KvStreamCounters kv_stream_counters(const KvStreamMap &m) {
    int32_t c[kKvCtlInts] = {};
    KvStreamCounters r;
    if (m.ctl == nullptr || cudaMemcpy(c, m.ctl, sizeof(c), cudaMemcpyDeviceToHost) != cudaSuccess)
        return r;
    const unsigned long long *u = reinterpret_cast<const unsigned long long *>(c + 4);
    r.misses = u[0];
    r.lookups = u[1];
    r.calls = u[2];
    r.overflow = c[3] != 0;
    return r;
}

} // namespace strata::kernels
