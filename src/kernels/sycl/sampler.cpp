// src/kernels/sycl/sampler.cpp - SYCL port of src/kernels/cuda/sampler.cu (P2.S2: the
// sampler chain, in llama.cpp's order).
//
//     penalties -> top_k -> top_p -> min_p -> temperature -> pick
//
// The CUDA file's header and per-kernel comments carry the contract - THE ORDER IS THE
// WHOLE CONTENT, the tie rule (larger value, then the LOWER index), the `n_vocab` "no
// candidate" sentinel - and are not repeated here; the arithmetic below is line-for-line
// the same.  Only what SYCL forces to change:
//
//   * warp shuffles             -> `sycl::sub_group` under a required 32-wide sub-group
//     (the runtime refuses devices without it): `__shfl_down_sync` -> shift_group_left,
//     `__shfl_xor_sync` -> permute_group_by_xor, `__shfl_sync(x, 0)` -> group_broadcast.
//   * `__shared__` (static and dynamic) -> local accessors; `__syncthreads()` ->
//     group_barrier(work_group), `__syncwarp()` -> group_barrier(sub_group).
//   * shared-memory `atomicOr`  -> sycl::atomic_ref (relaxed, work-group scope) on the
//     local accessor's words; the bitmap fill is order-free, so the OR is all it needs.
//   * `int2` candidate lists    -> `SplitCand { int id; int bits; }` below (the same 8
//     bytes: the id and the value's bits); `__int_as_float` / `__float_as_int` ->
//     sycl::bit_cast, `__umulhi` -> the 64-bit shift the parity test's host Philox
//     spells out.
//   * `logf` -> `sycl::log`, double `exp` -> `sycl::exp` (fp64 through NEO's emulation,
//     as quantize_act and silu already rely on); no libm float imports in device code.
//   * IGC's default f32 division with a VARIABLE divisor is not correctly rounded (see
//     docs/INTEL_SYCL.md), and the CUDA reference's `logit / penalty_repeat` and
//     `1.0f / temperature` are IEEE - both go through
//     strata::kernels::fp_exact::fdiv_rn.  Every other division is fp64 (correctly
//     rounded under emulation) or by a compile-time constant.
//   * `sampler_greedy_cluster_kernel` (thread-block clusters, sm_90+) has no SYCL
//     equivalent: the backend reports a cc 8.0 sentinel, so `sample_greedy_cluster`
//     returns false exactly as the HIP build does and the one-block argmax runs -
//     the same token, by the cluster kernel's own contract.  STRATA_ARGMAX_MULTI is
//     honoured trivially (it only gated the cluster launch).
#include "strata/kernels/sampler.hpp"
#include "strata/core/coupled_draft.hpp"

#include "strata/sycl_runtime/queue_bridge.hpp"
#include "fp_exact.hpp"
#include "mapped_host.hpp"

#include <cstddef>

#include <cuda_runtime.h>

#include <sycl/ext/intel/math.hpp>

#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <mutex>
#include <vector>

namespace strata::kernels {
namespace {

namespace intel_math = sycl::ext::intel::math;

// The CUDA file's `int2` candidate: (token id, the logit's bits).
struct SplitCand {
    int id;
    int bits;
};

constexpr float kNegInfF = sycl::bit_cast<float>(0xff800000u);
constexpr float kPosInfF = sycl::bit_cast<float>(0x7f800000u);

// ---- Philox 4x32-10, counter-based, exactly the CUDA file's (see its comment for why
// counter-based matters).  Integer arithmetic throughout; the final scale is by 2^-24.
uint32_t umulhi32(uint32_t a, uint32_t b) { return (uint32_t) (((uint64_t) a * b) >> 32); }

void philox4x32_round(uint32_t& c0, uint32_t& c1, uint32_t& c2, uint32_t& c3, uint32_t k0, uint32_t k1) {
    const uint32_t hi0 = umulhi32(0x9E3779B9u, c0);
    const uint32_t hi1 = umulhi32(0xBB67AE85u, c2);
    const uint32_t lo0 = 0x9E3779B9u * c0;
    const uint32_t lo1 = 0xBB67AE85u * c2;
    const uint32_t n0 = hi1 ^ c1 ^ k0;
    const uint32_t n1 = lo1;
    const uint32_t n2 = hi0 ^ c3 ^ k1;
    const uint32_t n3 = lo0;
    c0 = n0;
    c1 = n1;
    c2 = n2;
    c3 = n3;
}

float philox_uniform(uint64_t seed, uint64_t counter) {
    uint32_t c0 = (uint32_t) counter, c1 = (uint32_t) (counter >> 32);
    uint32_t c2 = (uint32_t) seed, c3 = (uint32_t) (seed >> 32);
    for (int i = 0; i < 10; ++i) philox4x32_round(c0, c1, c2, c3, (uint32_t) i, 0u);
    // 24 bits of mantissa, so the value is uniform in [0,1) with no rounding to 1.0
    return (float) (c0 >> 8) * (1.0f / 16777216.0f);
}

// ---- the penalties, transcribed from `llama_sampler_penalties_apply` (see the CUDA
// file: the repeat penalty MULTIPLIES non-positive logits and DIVIDES positive ones;
// the presence penalty is a boolean).
int history_count(const int* h, int n, int v) {
    int c = 0;
    for (int i = 0; i < n; ++i)
        if (h[i] == v) ++c;
    return c;
}

float apply_penalties(float logit, int count, const SamplerParams& p) {
    if (count <= 0) return logit;
    if (logit <= 0.0f) logit *= p.penalty_repeat;
    else               logit = intel_math::fdiv_rn(logit, p.penalty_repeat);   // IEEE, as nvcc's `/`
    logit -= (float) count * p.penalty_freq + (count > 0 ? 1.0f : 0.0f) * p.penalty_present;
    return logit;
}

// `1.0f / p.temperature`, IEEE-rounded as in the CUDA file (fdiv_rn; see the header).
float inv_temperature(const SamplerParams& p) {
    return p.temperature > 0.0f ? intel_math::fdiv_rn(1.0f, p.temperature) : 0.0f;
}

// The penalty window is the last `last_n` entries of this row's history (the TAIL).
struct PenaltyWindow {
    const int* hrow;
    int hlen;
};

PenaltyWindow penalty_window(const int* history, int history_len, int last_n, int t) {
    PenaltyWindow w{nullptr, 0};
    w.hrow = history ? history + (size_t) t * history_len : nullptr;
    if (w.hrow) {
        w.hlen = last_n < history_len ? last_n : history_len;
        if (w.hlen < 0) w.hlen = 0;
        w.hrow += history_len - w.hlen;
    }
    return w;
}

using LocalAtomic = sycl::atomic_ref<unsigned int, sycl::memory_order::relaxed, sycl::memory_scope::work_group,
                                     sycl::access::address_space::generic_space>;

// The full-vocabulary membership bitmap of the window, as in `sampler_greedy_kernel`:
// the gate needs a NON-EMPTY WINDOW (see the CUDA file).  `bits` has at least
// `bits_words` words.  The barriers are work-group-uniform: the gate is.
bool penal_bits_fill(sycl::nd_item<1> it, unsigned* bits, int bits_words, const PenaltyWindow& w, int n_vocab) {
    const bool use_bits = w.hrow != nullptr && w.hlen > 0 && bits_words > 0;
    if (use_bits) {
        const int tid = (int) it.get_local_linear_id();
        const int nth = (int) it.get_local_range(0);
        for (int i = tid; i < bits_words; i += nth) bits[i] = 0u;
        sycl::group_barrier(it.get_group());
        for (int i = tid; i < w.hlen; i += nth)
            if (w.hrow[i] >= 0 && w.hrow[i] < n_vocab)   // an id outside the vocabulary is never a candidate
                LocalAtomic(bits[w.hrow[i] >> 5]).fetch_or(1u << (w.hrow[i] & 31));
        sycl::group_barrier(it.get_group());
    }
    return use_bits;
}

int penal_hit_count(const unsigned* bits, const PenaltyWindow& w, bool use_bits, int v) {
    if (!use_bits || !(bits[v >> 5] & (1u << (v & 31)))) return 0;
    return history_count(w.hrow, w.hlen, v);
}

// top_k 1..64 as given; 0 ("off") and anything wider keep 64; never more than the vocabulary
constexpr int kSelMax = 64;

int sampled_k(int top_k, int n_vocab) {
    int k = (top_k > 0 && top_k < kSelMax) ? top_k : kSelMax;
    return k > n_vocab ? n_vocab : k;
}

// (bv, bi) <- the first of (bv, bi) and (ov, oi) in the selection order
void take_first(float& bv, int& bi, float ov, int oi) {
    if (ov > bv || (ov == bv && oi < bi)) { bv = ov; bi = oi; }
}

// The `__shfl_down_sync` butterfly of the CUDA reductions: only lane 0's result is used.
void sg_first_down(sycl::sub_group sg, float& bv, int& bi) {
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = sycl::shift_group_left(sg, bv, (uint32_t) off);
        const int oi = sycl::shift_group_left(sg, bi, (uint32_t) off);
        take_first(bv, bi, ov, oi);
    }
}

// The XOR butterfly of `warp_first`: the first of the sub-group's 32 candidates, left in
// EVERY lane (see the CUDA file for why that is exact).
void sg_first_xor(sycl::sub_group sg, float& bv, int& bi) {
    for (int off = 16; off > 0; off >>= 1) {
        const float ov = sycl::permute_group_by_xor(sg, bv, (uint32_t) off);
        const int oi = sycl::permute_group_by_xor(sg, bi, (uint32_t) off);
        take_first(bv, bi, ov, oi);
    }
}

/// `sampler_kernel`'s tail on ONE SUB-GROUP, arithmetic unchanged (see the CUDA file's
/// `sampled_tail_warp` comment: the lanes share the `exp`s, lane 0 alone runs the two
/// ORDERED sums and the two cumulative scans, every double is the one the old chain
/// computed).  `sel_ids` / `sel_logit` are the top_k list in selection order, in local
/// memory; lane 0 writes `out[t]` (and, kProb, the pick's probability to `*prob_out`).
template <bool kProb = false>
void sampled_tail_warp(sycl::sub_group sg, const int* sel_ids, const float* sel_logit, int k,
                       const SamplerParams& p, int t, int* out, double* ex, float* prob_out = nullptr) {
    const int lane = (int) sg.get_local_linear_id();
    const float inv_t = inv_temperature(p);
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = sycl::fmax(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        for (int i = lane; i < k; i += 32) ex[i] = sycl::exp((double) sel_logit[i] - (double) mx);
        sycl::group_barrier(sg);
        double sum = 0.0;
        if (lane == 0)
            for (int i = 0; i < k; ++i) sum += ex[i];
        sum = sycl::group_broadcast(sg, sum, 0u);
        sycl::group_barrier(sg);                 // lane 0 has read every `ex` before it is overwritten
        for (int i = lane; i < k; i += 32) ex[i] = ex[i] / sum;
        sycl::group_barrier(sg);
        int cut = k;
        if (lane == 0) {
            double cum = 0.0;
            for (int i = 0; i < k; ++i) {
                cum += ex[i];
                if (cum >= (double) p.top_p) { cut = i + 1; break; }
            }
        }
        cut = sycl::group_broadcast(sg, cut, 0u);
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
        sycl::group_barrier(sg);                 // `ex` is written again below
    }
    // min_p on top_p's survivors, as in `sampler_kernel` (every lane, the same float arithmetic)
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + sycl::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    // temperature, then one Philox draw
    float smx = sel_logit[0] * inv_t;
    for (int i = 1; i < n_keep; ++i) smx = sycl::fmax(smx, sel_logit[i] * inv_t);
    for (int i = lane; i < n_keep; i += 32) ex[i] = sycl::exp((double) (sel_logit[i] * inv_t) - (double) smx);
    sycl::group_barrier(sg);
    double sum = 0.0;
    if (lane == 0)
        for (int i = 0; i < n_keep; ++i) sum += ex[i];
    sum = sycl::group_broadcast(sg, sum, 0u);
    sycl::group_barrier(sg);
    for (int i = lane; i < n_keep; i += 32) ex[i] = ex[i] / sum;
    sycl::group_barrier(sg);
    if (lane == 0) {
        const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
        double cum = 0.0;
        int pi = n_keep > 0 ? n_keep - 1 : 0;
        int pick = sel_ids[pi];
        for (int i = 0; i < n_keep; ++i) {
            cum += ex[i];
            if ((double) u < cum) { pick = sel_ids[i]; pi = i; break; }
        }
        out[t] = pick;
        if constexpr (kProb) *prob_out = n_keep > 0 ? (float) ex[pi] : 1.0f;
    }
}

/// **THE GREEDY ARGMAX, ONE WORK-GROUP PER TOKEN** - `sampler_greedy_kernel` unchanged,
/// launch shape and tie rule included (see the CUDA file for both).
void greedy_body(sycl::nd_item<1> it, const float* logits, int n_vocab, const int* history, int history_len,
                 const SamplerParams p, int plen, int* out, unsigned* penal_bits, float* sv, int* si) {
    const int t = (int) it.get_group(0);
    const float* l = logits + (size_t) t * n_vocab;
    const PenaltyWindow w = penalty_window(history, history_len, plen, t);
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = penal_bits_fill(it, penal_bits, bits_words, w, n_vocab);

    const sycl::sub_group sg = it.get_sub_group();
    const int lane = (int) sg.get_local_linear_id();
    const int tid = (int) it.get_local_linear_id();
    const int warp = tid >> 5;
    const int nth = (int) it.get_local_range(0);

    // `n_vocab` is the "no candidate" index: it loses every comparison to a real one.
    float bv = kNegInfF;
    int best = n_vocab;
    for (int v = tid; v < n_vocab; v += nth) {
        const float s = apply_penalties(l[v], penal_hit_count(penal_bits, w, use_bits, v), p);
        if (s > bv) { bv = s; best = v; }
    }
    sg_first_down(sg, bv, best);
    if (lane == 0) { sv[warp] = bv; si[warp] = best; }
    sycl::group_barrier(it.get_group());
    if (warp == 0) {
        const int nw = (nth + 31) >> 5;
        float wv = lane < nw ? sv[lane] : kNegInfF;
        int wi = lane < nw ? si[lane] : n_vocab;
        sg_first_down(sg, wv, wi);
        // A tie between two `-inf` candidates leaves `wi == n_vocab`, and the serial version answered 0.
        if (lane == 0) out[t] = (wi < n_vocab) ? wi : 0;
    }
}

/// **THE SAMPLED PATH, ONE WORK-GROUP PER TOKEN, AS THE REFERENCE** - `sampler_kernel`
/// (STRATA_OLD_SAMPLER=1), with its O(k^2 x n_vocab) `taken` sweep and the tail computed
/// redundantly by every work-item.  See the CUDA file for the semantics.
void old_body(sycl::nd_item<1> it, const float* logits, int n_vocab, int n_tokens, const int* history,
              int history_len, const SamplerParams p, int* out, unsigned* penal_bits, int* sel_ids,
              float* sel_logit, float* sv, int* si) {
    const int t = (int) it.get_group(0);
    if (t >= n_tokens) return;
    const float* l = logits + (size_t) t * n_vocab;
    // Temperature is needed by BOTH stages below; the chain still APPLIES it after the truncation filters.
    const float inv_t = inv_temperature(p);
    const PenaltyWindow w = penalty_window(history, history_len, p.penalty_last_n, t);
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = penal_bits_fill(it, penal_bits, bits_words, w, n_vocab);

    const sycl::sub_group sg = it.get_sub_group();
    const int lane = (int) sg.get_local_linear_id();
    const int tid = (int) it.get_local_linear_id();
    const int warp = tid >> 5;
    const int nth = (int) it.get_local_range(0);

    const int k = sampled_k(p.top_k, n_vocab);
    for (int i = 0; i < k; ++i) {
        float bv = kNegInfF;
        int best = n_vocab;
        for (int v = tid; v < n_vocab; v += nth) {
            bool taken = false;
            for (int j = 0; j < i; ++j)
                if (sel_ids[j] == v) { taken = true; break; }
            if (taken) continue;
            const float s = apply_penalties(l[v], penal_hit_count(penal_bits, w, use_bits, v), p);
            if (s > bv) { bv = s; best = v; }
        }
        sg_first_down(sg, bv, best);
        if (lane == 0) { sv[warp] = bv; si[warp] = best; }
        sycl::group_barrier(it.get_group());
        if (warp == 0) {
            const int nw = (nth + 31) >> 5;
            float wv = lane < nw ? sv[lane] : kNegInfF;
            int wi = lane < nw ? si[lane] : n_vocab;
            sg_first_down(sg, wv, wi);
            if (lane == 0) { sel_ids[i] = (wi < n_vocab) ? wi : 0; sel_logit[i] = wv; }
        }
        sycl::group_barrier(it.get_group());
    }

    // top_p over the top_k list, then min_p, then temperature and one Philox draw - llama.cpp's order
    // (issue #53), every work-item computing the same chain redundantly (the old kernel's shape).
    int n_keep = k;
    float mx = sel_logit[0];
    for (int i = 1; i < k; ++i) mx = sycl::fmax(mx, sel_logit[i]);
    if (p.top_p < 1.0f) {
        double sum = 0.0;
        for (int i = 0; i < k; ++i) sum += sycl::exp((double) sel_logit[i] - (double) mx);
        double cum = 0.0;
        int cut = k;
        for (int i = 0; i < k; ++i) {
            cum += sycl::exp((double) sel_logit[i] - (double) mx) / sum;
            if (cum >= (double) p.top_p) { cut = i + 1; break; }
        }
        if (cut < p.min_keep) cut = p.min_keep < k ? p.min_keep : k;
        n_keep = cut;
    }
    if (p.min_p > 0.0f) {
        const float thresh = sel_logit[0] + sycl::log(p.min_p);
        for (int i = 0; i < n_keep; ++i)
            if (sel_logit[i] < thresh) { n_keep = i; break; }
    }
    // temperature only: the penalties were applied once, before the selection (issue #53)
    auto scaled = [&](int i) { return sel_logit[i] * inv_t; };
    float smx = scaled(0);
    for (int i = 1; i < n_keep; ++i) smx = sycl::fmax(smx, scaled(i));
    double sum = 0.0;
    for (int i = 0; i < n_keep; ++i) sum += sycl::exp((double) scaled(i) - (double) smx);
    const float u = philox_uniform(p.seed, p.counter + (uint64_t) t);
    double cum = 0.0;
    int pick = sel_ids[n_keep - 1];
    for (int i = 0; i < n_keep; ++i) {
        cum += sycl::exp((double) scaled(i) - (double) smx) / sum;
        if ((double) u < cum) { pick = sel_ids[i]; break; }
    }
    if (tid == 0) out[t] = pick;
}

/// **THE ONE-BLOCK SAMPLED PATH** - `sampler_one_block_kernel`: the threshold selection
/// (`s < prev_v || (s == prev_v && v > prev_i)`) instead of the `taken` sweep, then
/// sub-group 0 runs the tail.  Semantics identical to the reference; see the CUDA file.
void one_block_body(sycl::nd_item<1> it, const float* logits, int n_vocab, const int* history, int history_len,
                    const SamplerParams p, int* out, unsigned* penal_bits, int* sel_ids, float* sel_logit,
                    double* ex, float* sv, int* si) {
    const int t = (int) it.get_group(0);
    const float* l = logits + (size_t) t * n_vocab;
    const PenaltyWindow w = penalty_window(history, history_len, p.penalty_last_n, t);
    const int bits_words = (int) ((n_vocab + 31) / 32);
    const bool use_bits = penal_bits_fill(it, penal_bits, bits_words, w, n_vocab);

    const sycl::sub_group sg = it.get_sub_group();
    const int lane = (int) sg.get_local_linear_id();
    const int tid = (int) it.get_local_linear_id();
    const int warp = tid >> 5;
    const int nth = (int) it.get_local_range(0);

    const int k = sampled_k(p.top_k, n_vocab);
    float prev_v = kPosInfF;   // +inf and id -1: round 0 takes every logit
    int prev_i = -1;
    for (int i = 0; i < k; ++i) {
        float bv = kNegInfF;
        int best = n_vocab;
        for (int v = tid; v < n_vocab; v += nth) {
            const float s = apply_penalties(l[v], penal_hit_count(penal_bits, w, use_bits, v), p);
            if ((s < prev_v || (s == prev_v && v > prev_i)) && s > bv) { bv = s; best = v; }
        }
        sg_first_down(sg, bv, best);
        if (lane == 0) { sv[warp] = bv; si[warp] = best; }
        sycl::group_barrier(it.get_group());
        if (warp == 0) {
            const int nw = (nth + 31) >> 5;
            float wv = lane < nw ? sv[lane] : kNegInfF;
            int wi = lane < nw ? si[lane] : n_vocab;
            sg_first_down(sg, wv, wi);
            if (lane == 0) { sel_ids[i] = (wi < n_vocab) ? wi : 0; sel_logit[i] = wv; }
        }
        sycl::group_barrier(it.get_group());
        // An empty round leaves (-inf, 0): nothing comes after it, as nothing was left untaken.
        prev_v = sel_logit[i];
        prev_i = sel_ids[i];
    }
    if (warp != 0) return;
    sampled_tail_warp(sg, sel_ids, sel_logit, k, p, t, out, ex);
}

// ---- THE SPLIT top_k (the CUDA file's comments carry the exactness argument: the first
// k of a union are within the first k of each part, and a merge of ordered lists is
// ordered, the selection order being a strict total order).
constexpr int kSplitPerLane = 32;                               // logits per lane, in registers
constexpr int kSplitWarpSpan = 32 * kSplitPerLane;              // 1,024 logits per sub-group
constexpr int kSplitWarps = 4;
constexpr int kSplitBlockSpan = kSplitWarps * kSplitWarpSpan;   // 4,096 logits per work-group
constexpr int kSplitMaxBlocks = 64;                             // lists the merge holds: n_vocab <= 262,144
constexpr int kSplitMaxRows = 64;                               // rows per split launch (the scratch's bound)

/// Merge `nl` (<= 64) lists of `k` candidates - list L at `lists[L * stride]`, each in
/// the selection order and padded with sentinels - into their first `k`: `sink(i, value,
/// id)` runs in every lane for i = 0..k-1 with the same pair.  Lane owns lists `lane`
/// and `lane + 32`; a round takes the first of all heads and advances the list it came
/// from.  An id is in one list at most, so exactly one head matches.
template <typename Sink>
void warp_merge_lists(sycl::sub_group sg, const SplitCand* lists, int nl, int stride, int k, int n_vocab,
                      Sink&& sink) {
    const int lane = (int) sg.get_local_linear_id();
    float hv[2];
    int hi[2], pos[2];
    for (int m = 0; m < 2; ++m) {
        const int L = lane + 32 * m;
        pos[m] = 0;
        hv[m] = kNegInfF;
        hi[m] = n_vocab;
        if (L < nl) {
            const SplitCand c = lists[(size_t) L * stride];
            hi[m] = c.id;
            hv[m] = sycl::bit_cast<float>(c.bits);
        }
    }
    for (int i = 0; i < k; ++i) {
        float bv = hv[0];
        int bi = hi[0];
        take_first(bv, bi, hv[1], hi[1]);
        sg_first_xor(sg, bv, bi);
        sink(i, bv, bi);
        if (bi < n_vocab) {
            for (int m = 0; m < 2; ++m) {
                if (hi[m] != bi) continue;
                if (++pos[m] < k) {
                    const SplitCand c = lists[(size_t) (lane + 32 * m) * stride + pos[m]];
                    hi[m] = c.id;
                    hv[m] = sycl::bit_cast<float>(c.bits);
                } else {
                    hi[m] = n_vocab;
                    hv[m] = kNegInfF;
                }
            }
        }
    }
}

/// **SPLIT STAGE 1: THE top_k OF EACH 4,096-LOGIT BLOCK** - `sampler_split_part_kernel`.
/// Grid (blocks per row, rows), 128 work-items.  Each sub-group holds its 1,024 penalised
/// logits in registers and runs `k` argmax rounds with the one-block threshold; sub-group
/// 0 merges the four lists into the block's list in `cand`.
void split_part_body(sycl::nd_item<2> it, const float* logits, int n_vocab, const int* history, int history_len,
                     const SamplerParams p, int k, int n_blocks, SplitCand* cand, unsigned* bits, SplitCand* wl) {
    const int t = (int) it.get_group(1);
    const float* l = logits + (size_t) t * n_vocab;
    const sycl::sub_group sg = it.get_sub_group();
    const int lane = (int) sg.get_local_linear_id();
    const int tid = (int) it.get_local_linear_id();
    const int warp = tid >> 5;
    const int nth = (int) it.get_local_range(0);
    const int blo = (int) it.get_group(0) * kSplitBlockSpan;

    const PenaltyWindow w = penalty_window(history, history_len, p.penalty_last_n, t);
    // The membership bitmap of THIS WORK-GROUP'S 4,096 logits (512 bytes, not the
    // vocabulary's 31 KB): the same test, and a hit pays the same exact count.
    const bool use_bits = w.hrow != nullptr && w.hlen > 0;
    if (use_bits) {
        for (int i = tid; i < kSplitBlockSpan / 32; i += nth) bits[i] = 0u;
        sycl::group_barrier(it.get_group());
        for (int i = tid; i < w.hlen; i += nth) {
            const int h = w.hrow[i];
            if (h >= 0 && h < n_vocab && h >= blo && h - blo < kSplitBlockSpan)
                LocalAtomic(bits[(h - blo) >> 5]).fetch_or(1u << ((h - blo) & 31));
        }
        sycl::group_barrier(it.get_group());
    }

    // This sub-group's logits, penalised (a zero count returns the logit unchanged, so
    // only the bitmap's hits go through it).  Past the vocabulary: -inf.
    const int lo = blo + warp * kSplitWarpSpan;
    float s[kSplitPerLane];
    for (int j = 0; j < kSplitPerLane; ++j) {
        const int v = lo + 32 * j + lane;
        s[j] = v < n_vocab ? l[v] : kNegInfF;
    }
    if (use_bits) {
        for (int j = 0; j < kSplitPerLane; ++j) {
            const int v = lo + 32 * j + lane, b = v - blo;
            if (v < n_vocab && (bits[b >> 5] & (1u << (b & 31))))
                s[j] = apply_penalties(s[j], history_count(w.hrow, w.hlen, v), p);
        }
    }

    SplitCand* wlw = wl + warp * kSelMax;
    float prev_v = kPosInfF;   // +inf and id -1: round 0 takes every logit
    int prev_i = -1;
    int i = 0;
    for (; i < k; ++i) {
        // two chains (even and odd j), each walked in ascending id with a strict `>`
        float b0 = kNegInfF, b1 = kNegInfF;
        int i0 = n_vocab, i1 = n_vocab;
        for (int j = 0; j < kSplitPerLane; j += 2) {
            const int v0 = lo + 32 * j + lane, v1 = v0 + 32;
            const float x0 = s[j], x1 = s[j + 1];
            if ((x0 < prev_v || (x0 == prev_v && v0 > prev_i)) && x0 > b0) { b0 = x0; i0 = v0; }
            if ((x1 < prev_v || (x1 == prev_v && v1 > prev_i)) && x1 > b1) { b1 = x1; i1 = v1; }
        }
        take_first(b0, i0, b1, i1);
        sg_first_xor(sg, b0, i0);
        if (i0 >= n_vocab) break;                // the same in every lane: nothing left in these 1,024 logits
        if (lane == 0) wlw[i] = SplitCand{i0, sycl::bit_cast<int>(b0)};
        prev_v = b0;
        prev_i = i0;
    }
    for (int r = i + lane; r < k; r += 32) wlw[r] = SplitCand{n_vocab, (int) 0xff800000u};   // sentinels
    sycl::group_barrier(it.get_group());
    if (warp == 0) {
        SplitCand* dst = cand + ((size_t) t * n_blocks + it.get_group(0)) * k;
        warp_merge_lists(sg, wl, kSplitWarps, kSelMax, k, n_vocab, [&](int r, float v, int id) {
            if (lane == 0) dst[r] = SplitCand{id, sycl::bit_cast<int>(v)};
        });
    }
}

/// **SPLIT STAGE 2: ONE SUB-GROUP PER ROW MERGES THE BLOCK LISTS, THEN RUNS THE TAIL** -
/// `sampler_split_merge_kernel` unchanged.
void split_merge_body(sycl::nd_item<1> it, const SplitCand* cand, int n_blocks, int n_vocab, const SamplerParams p,
                      int k, int* out, SplitCand* lists, int* sel_ids, float* sel_logit, double* ex) {
    const int t = (int) it.get_group(0);
    const sycl::sub_group sg = it.get_sub_group();
    const int lane = (int) sg.get_local_linear_id();
    const SplitCand* src = cand + (size_t) t * n_blocks * k;
    for (int e = lane; e < n_blocks * k; e += 32) lists[e] = src[e];
    sycl::group_barrier(sg);
    warp_merge_lists(sg, lists, n_blocks, k, k, n_vocab, [&](int i, float v, int id) {
        if (lane == 0) { sel_ids[i] = id < n_vocab ? id : 0; sel_logit[i] = v; }
    });
    sycl::group_barrier(sg);
    sampled_tail_warp(sg, sel_ids, sel_logit, k, p, t, out, ex);
}

// ---- launch helpers -------------------------------------------------------------

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

void launch_greedy(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, unsigned shmem_words, int* out, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        // the penalty bitmap (1 word when penalties are off: the kernel never reads it then)
        sycl::local_accessor<unsigned int, 1> bits(sycl::range<1>(shmem_words ? shmem_words : 1), h);
        sycl::local_accessor<float, 1> sv(sycl::range<1>(32), h);
        sycl::local_accessor<int, 1> si(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * 1024, 1024),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           greedy_body(it, logits, n_vocab, history, history_len, p, p.penalty_last_n, out,
                                       &bits[0], &sv[0], &si[0]);
                       });
    });
}

void launch_old(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                const SamplerParams& p, unsigned shmem_words, int* out, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<unsigned int, 1> bits(sycl::range<1>(shmem_words ? shmem_words : 1), h);
        sycl::local_accessor<int, 1> sel_ids(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<float, 1> sel_logit(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<float, 1> sv(sycl::range<1>(32), h);
        sycl::local_accessor<int, 1> si(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * 1024, 1024),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           old_body(it, logits, n_vocab, n_tokens, history, history_len, p, out, &bits[0],
                                    &sel_ids[0], &sel_logit[0], &sv[0], &si[0]);
                       });
    });
}

void launch_one_block(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                      const SamplerParams& p, unsigned shmem_words, int* out, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<unsigned int, 1> bits(sycl::range<1>(shmem_words ? shmem_words : 1), h);
        sycl::local_accessor<int, 1> sel_ids(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<float, 1> sel_logit(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<double, 1> ex(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<float, 1> sv(sycl::range<1>(32), h);
        sycl::local_accessor<int, 1> si(sycl::range<1>(32), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * 1024, 1024),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           one_block_body(it, logits, n_vocab, history, history_len, p, out, &bits[0], &sel_ids[0],
                                          &sel_logit[0], &ex[0], &sv[0], &si[0]);
                       });
    });
}

void launch_split(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                  const SamplerParams& p, int k, int n_blocks, SplitCand* scratch, int* out, sycl::queue& q) {
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<unsigned int, 1> bits(sycl::range<1>(kSplitBlockSpan / 32), h);
        sycl::local_accessor<SplitCand, 1> wl(sycl::range<1>(kSplitWarps * kSelMax), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) n_blocks * (kSplitWarps * 32), (size_t) n_tokens),
                                         sycl::range<2>(kSplitWarps * 32, 1)),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
                           split_part_body(it, logits, n_vocab, history, history_len, p, k, n_blocks, scratch,
                                           &bits[0], &wl[0]);
                       });
    });
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<SplitCand, 1> lists(sycl::range<1>(kSplitMaxBlocks * kSelMax), h);
        sycl::local_accessor<int, 1> sel_ids(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<float, 1> sel_logit(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<double, 1> ex(sycl::range<1>(kSelMax), h);
        h.parallel_for(sycl::nd_range<1>((size_t) n_tokens * 32, 32),
                       [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                           split_merge_body(it, scratch, n_blocks, n_vocab, p, k, out, &lists[0], &sel_ids[0],
                                            &sel_logit[0], &ex[0]);
                       });
    });
}

// ---- host-side path selection, as in the CUDA file ------------------------------

enum class SampledPath { Split, OneBlock, Old };

bool env_flag(const char* name) {
    const char* e = std::getenv(name);
    return e != nullptr && *e != '\0' && std::strcmp(e, "0") != 0;
}

SampledPath sampled_path() {
    static const SampledPath path = env_flag("STRATA_OLD_SAMPLER")         ? SampledPath::Old
                                    : env_flag("STRATA_SAMPLER_ONE_BLOCK") ? SampledPath::OneBlock
                                                                           : SampledPath::Split;
    return path;
}

// A stream being captured into a graph must not reach `split_scratch` (cudaMalloc): it
// gets the one-block kernel, which needs no memory of its own.  The legacy stream cannot be captured.
bool stream_capturing(void* stream) {
    if (stream == nullptr) return false;
    cudaStreamCaptureStatus st = cudaStreamCaptureStatusNone;
    if (cudaStreamIsCapturing((cudaStream_t) stream, &st) != cudaSuccess) {
        (void) cudaGetLastError();
        return true;
    }
    return st != cudaStreamCaptureStatusNone;
}

// The split's block lists, one buffer per (device, stream), grown on demand, retired
// rather than freed - the CUDA file's policy and its reasons, unchanged.
SplitCand* split_scratch(void* stream, size_t entries) {
    struct Slot {
        int device;
        void* stream;
        SplitCand* ptr;
        size_t entries;
        size_t failed;                       // the smallest size cudaMalloc refused (0: none)
    };
    static std::mutex mu;
    static std::vector<Slot> slots;
    static std::vector<SplitCand*> retired;
    int device = 0;
    if (cudaGetDevice(&device) != cudaSuccess) {
        (void) cudaGetLastError();
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mu);
    Slot* slot = nullptr;
    for (Slot& s : slots)
        if (s.device == device && s.stream == stream) slot = &s;
    if (slot == nullptr) {
        slots.push_back({device, stream, nullptr, 0, 0});
        slot = &slots.back();
    }
    if (slot->entries >= entries) return slot->ptr;
    if (slot->failed != 0 && entries >= slot->failed) return nullptr;
    constexpr size_t kCap = (size_t) kSplitMaxRows * kSplitMaxBlocks * kSelMax;
    size_t want = 2 * slot->entries < kCap ? 2 * slot->entries : kCap;
    if (want < entries) want = entries;
    SplitCand* ptr = nullptr;
    if (cudaMalloc(&ptr, want * sizeof(SplitCand)) != cudaSuccess) {
        (void) cudaGetLastError();
        want = entries;
        if (cudaMalloc(&ptr, want * sizeof(SplitCand)) != cudaSuccess) {
            (void) cudaGetLastError();
            slot->failed = entries;
            return nullptr;
        }
    }
    if (slot->ptr != nullptr) retired.push_back(slot->ptr);
    slot->ptr = ptr;
    slot->entries = want;
    return ptr;
}

int coupled_blocks(int nv) { return (nv + kSplitBlockSpan - 1) / kSplitBlockSpan; }
int coupled_kpart(int nv) { return nv < kSelMax ? nv : kSelMax; }

}  // namespace

bool sample_greedy_cluster(const float* logits, int n_tokens, int n_vocab, int* out, void* stream) {
    // Thread-block clusters are sm_90+ CUDA; the SYCL backend reports a cc 8.0 sentinel, so
    // the one-block argmax always runs - the same token (see the header and the CUDA file).
    (void) logits;
    (void) n_tokens;
    (void) n_vocab;
    (void) out;
    (void) stream;
    return false;
}

void sample_tokens(const float* logits, int n_tokens, int n_vocab, const int* history, int history_len,
                   const SamplerParams& p, int* out, void* stream) {
    if (n_tokens <= 0 || n_vocab <= 0) return;
    if (p.penalty_last_n > 0 && (history == nullptr || history_len <= 0)) {
        std::fprintf(stderr, "sample_tokens: penalty_last_n %d needs a history (got %p, len %d)\n",
                     p.penalty_last_n, (const void*) history, history_len);
        std::exit(1);
    }
    const unsigned shmem_words = (history != nullptr && history_len > 0 && p.penalty_last_n > 0)
                                     ? (unsigned) ((n_vocab + 31) / 32)   // the penalty bitmap
                                     : 0;
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    if (p.greedy || p.temperature <= 0.0f) {
        // One work-group per token, 1,024 work-items over the vocabulary (the cluster
        // kernel is sm_90+ only; see `sample_greedy_cluster`).  See `sampler_greedy_kernel`.
        launch_greedy(logits, n_tokens, n_vocab, history, history_len, p, shmem_words, out, q);
    } else if (sampled_path() == SampledPath::Old) {
        launch_old(logits, n_tokens, n_vocab, history, history_len, p, shmem_words, out, q);
    } else {
        // The split top_k by default; the one-block kernel when asked for, or when the
        // split cannot run (see the CUDA file for the four conditions).
        const int k = sampled_k(p.top_k, n_vocab);
        const int n_blocks = (n_vocab + kSplitBlockSpan - 1) / kSplitBlockSpan;
        SplitCand* scratch = nullptr;
        if (sampled_path() == SampledPath::Split && n_blocks <= kSplitMaxBlocks && n_tokens <= kSplitMaxRows &&
            !stream_capturing(stream))
            // sized for 16 rows and 64 entries at least, so a verify window or a wider top_k does not regrow it
            scratch = split_scratch(stream, (size_t) (n_tokens > 16 ? n_tokens : 16) * n_blocks * kSelMax);
        if (scratch != nullptr) {
            launch_split(logits, n_tokens, n_vocab, history, history_len, p, k, n_blocks, scratch, out, q);
        } else {
            launch_one_block(logits, n_tokens, n_vocab, history, history_len, p, shmem_words, out, q);
        }
    }
    check_launch("sample_tokens");
    sync_if_needed(q, stream, "sample_tokens");
}

size_t coupled_draft_scratch_bytes(int nv) {
    if (nv <= 0 || coupled_blocks(nv) > kSplitMaxBlocks) return 0;
    return (size_t) coupled_blocks(nv) * (size_t) kSelMax * sizeof(SplitCand);
}

void coupled_draft_stage(const SamplerParams* mapped_params, const int32_t* mapped_hist, SamplerParams* params,
                         int32_t* ring, int cap, void* stream) {
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    q.parallel_for(sycl::nd_range<1>(256, 256), [=](sycl::nd_item<1> it) {
        const int tid = (int) it.get_local_linear_id();
        // mapped host memory the host rewrites per request: uncached loads (mapped_host.hpp)
        const int* s = (const int*) mapped_params;
        int* d = (int*) params;
        for (int i = tid; i < (int) (sizeof(SamplerParams) / sizeof(int)); i += 256) d[i] = sycl_mapped::load(s, (size_t) i);
        static_assert(offsetof(SamplerParams, penalty_last_n) % sizeof(int) == 0 &&
                      sizeof(SamplerParams::penalty_last_n) == sizeof(int), "penalty_last_n is one int");
        const int last_n = sycl_mapped::load(s, offsetof(SamplerParams, penalty_last_n) / sizeof(int));
        const int h = strata::core::coupled_hist_len(last_n, cap);
        for (int i = cap - h + tid; i < cap; i += 256) ring[i] = sycl_mapped::load(mapped_hist, (size_t) i);
    });
    check_launch("coupled_draft_stage");
}

void coupled_draft_sample(float* logits, int nv, const int32_t* sub_to_id, const int32_t* id_to_sub, int id_vocab,
                          const SamplerParams* params, int32_t* ring, int cap, int j, const int32_t* step_rec,
                          void* scratch, int32_t* out_id, float* out_prob, void* stream) {
    const int n_blocks = coupled_blocks(nv), kpart = coupled_kpart(nv);
    if (nv <= 0 || n_blocks > kSplitMaxBlocks || scratch == nullptr) {
        std::fprintf(stderr, "coupled_draft_sample: %d logits need scratch and at most %d blocks\n", nv,
                     kSplitMaxBlocks);
        std::exit(1);
    }
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    // the penalties, in place (see the CUDA file: each distinct token is penalised once)
    q.submit([&](sycl::handler& h) {
        const int words = (nv + 31) / 32;
        sycl::local_accessor<unsigned int, 1> seen(sycl::range<1>((size_t) words), h);
        h.parallel_for(sycl::nd_range<1>(1024, 1024), [=](sycl::nd_item<1> it) {
            const SamplerParams p = *params;
            const int hh = strata::core::coupled_hist_len(p.penalty_last_n, cap);
            if (hh <= 0) return;
            const int* hrow = ring + strata::core::coupled_hist_start(cap, j, hh);
            unsigned* sn = &seen[0];
            const int tid = (int) it.get_local_linear_id();
            for (int i = tid; i < words; i += 1024) sn[i] = 0u;
            sycl::group_barrier(it.get_group());
            for (int i = tid; i < hh; i += 1024) {
                const int v = hrow[i];
                if (v < 0 || v >= id_vocab) continue;
                const int s = id_to_sub != nullptr ? id_to_sub[v] : v;
                if (s < 0 || s >= nv) continue;
                const unsigned bit = 1u << (s & 31);
                if (LocalAtomic(sn[s >> 5]).fetch_or(bit) & bit) continue;
                logits[s] = apply_penalties(logits[s], history_count(hrow, hh, v), p);
            }
        });
    });
    check_launch("coupled_penalize");
    // the selection of the split sampler, unchanged (penalties already applied above)
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<unsigned int, 1> bits(sycl::range<1>(kSplitBlockSpan / 32), h);
        sycl::local_accessor<SplitCand, 1> wl(sycl::range<1>(kSplitWarps * kSelMax), h);
        h.parallel_for(sycl::nd_range<2>(sycl::range<2>((size_t) n_blocks * (kSplitWarps * 32), 1),
                                         sycl::range<2>(kSplitWarps * 32, 1)),
                       [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(32)]] {
                           split_part_body(it, logits, nv, nullptr, 0, SamplerParams{}, kpart, n_blocks,
                                           (SplitCand*) scratch, &bits[0], &wl[0]);
                       });
    });
    check_launch("coupled_draft split part");
    q.submit([&](sycl::handler& h) {
        sycl::local_accessor<SplitCand, 1> lists(sycl::range<1>(kSplitMaxBlocks * kSelMax), h);
        sycl::local_accessor<int, 1> sel_ids(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<float, 1> sel_logit(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<double, 1> ex(sycl::range<1>(kSelMax), h);
        sycl::local_accessor<int, 1> pick(sycl::range<1>(1), h);
        sycl::local_accessor<float, 1> prob(sycl::range<1>(1), h);
        h.parallel_for(sycl::nd_range<1>(32, 32), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
            const sycl::sub_group sg = it.get_sub_group();
            const int lane = (int) sg.get_local_linear_id();
            SamplerParams p = *params;
            p.counter = strata::core::coupled_draft_counter((int64_t) step_rec[0]);
            const int k = sampled_k(p.top_k, nv);    // <= kpart: the first k of a union lie in the first k of each list
            for (int e = lane; e < n_blocks * kpart; e += 32) lists[e] = ((const SplitCand*) scratch)[e];
            sycl::group_barrier(sg);
            warp_merge_lists(sg, &lists[0], n_blocks, kpart, k, nv, [&](int i, float v, int id) {
                if (lane == 0) { sel_ids[i] = id < nv ? id : 0; sel_logit[i] = v; }
            });
            sycl::group_barrier(sg);
            if (p.greedy || p.temperature <= 0.0f) {   // never launched for greedy requests; the argmax, defensively
                if (lane == 0) { pick[0] = sel_ids[0]; prob[0] = 1.0f; }
            } else {
                sampled_tail_warp<true>(sg, &sel_ids[0], &sel_logit[0], k, p, 0, &pick[0], &ex[0], &prob[0]);
            }
            sycl::group_barrier(sg);
            if (lane == 0) {
                const int s = pick[0];
                const int id = sub_to_id != nullptr ? sub_to_id[s] : s;
                *out_id = id;
                *out_prob = prob[0];
                ring[cap + j] = id;
            }
        });
    });
    check_launch("coupled_draft merge");
}

}  // namespace strata::kernels
