// include/strata/kernels/qsa_select.hpp - plan v0.3 P5/P7: the QSA indexer's block scores and top-k selection,
// for many queries at once and at long context.
//
// The per-token path scores every pooled block with one FP64 block per row (qsa_index_kernel) and selects with a
// single-block kernel that makes 32 bit-serial passes over every CELL (topk_kernel): ~0.1-0.2 ms per query at 32K,
// once per QSA layer and token - 1-2 ms of a decode token and most of a 32K prompt's time.  Here:
//
//   qsa_block_scores : one warp per (query, block), FP32; relu per indexer head, summed; block n_bid (the
//                      incomplete tail) scores the `dead` key and gets +1e9 when it has cells - which is exactly
//                      what the per-token path reads there, because pooled[n_bid] holds `dead` at that time (in
//                      a chunk it may already hold a block completed later, so it is not read).
//   qsa_block_topk   : one block per query; a 4-pass radix select over the query's n_bid + 1 blocks, each
//                      weighted by its cell count, then the cells emitted in ascending order with ties to the
//                      lowest index - the same selection as topk_kernel, over a quarter of the elements.  On sm_90+
//                      a call of up to 16 queries runs it on a cluster of 8 CTAs per query: the same ids.
//
// Queries carry their own step record (pos, n_kv, n_bid, width) as everywhere else in QSA.
#pragma once

#include "strata/kernels/qsa.hpp"

#include <cstdint>

namespace strata::kernels {

/// scores [nq, max_blocks]; q_idx [nq, idx_n_head, idx_dim] (normed and rotated); steps [nq, kStepCount].
void qsa_block_scores(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                      int64_t max_blocks, const QsaShapes& s, float* scores, void* stream,
                      int64_t active_blocks = -1);   ///< perf-review C-1: > 0 launches only this many blocks (the
                                                     ///< batch's largest n_bid + 1; not for a captured graph)

/// The same scores on tensor cores (3xTF32, FP32-level accuracy but another summation order: not bitwise; the tail
/// block n_bid is the warp kernel's arithmetic). For the prompt path; false (nothing launched) on another geometry.
/// On AMD it is the gfx12 (RDNA4) WMMA kernel (a three-way bf16 split, six products); false on any other AMD target.
/// On CUDA below sm_80 (no TF32 tensor cores) it is an FP32 tiled kernel, also not bitwise, and only with STRATA_SELECT_SIMT=1 (otherwise false).
bool qsa_block_scores_tc(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps, int64_t nq,
                         int64_t max_blocks, const QsaShapes& s, float* scores, void* stream, int64_t active_blocks);

/// ids [nq, cap] (cells, ascending); `cap` >= the largest selection width.
void qsa_block_topk(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                    const QsaShapes& s, int32_t* ids, void* stream,
                    int64_t active_blocks = -1);   ///< > 0: no query of the call has more than this many blocks (n_bid + 1;
                                                   ///< the same contract as qsa_block_scores's). The register kernel is
                                                   ///< chosen by this, not by the capacity max_blocks (a long
                                                   ///< --max-context otherwise sends every short prompt to the slow one).
                                                   ///< CUDA uses the bound only on sm_75; HIP keeps its existing policy.
                                                   ///< Omit it for captured graphs whose context can grow after capture.
/// The same ids on a thread-block cluster of 8 CTAs per query (sm_90+, CUDA; S19). qsa_block_topk takes it for calls
/// of up to 16 queries unless STRATA_QSA_CLUSTER=0. False (nothing launched) where it cannot run: HIP, a card or a
/// build below sm_90, or a capacity whose keys do not fit one cluster's shared memory. Capturable.
bool qsa_block_topk_cluster(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                            const QsaShapes& s, int32_t* ids, void* stream);
/// The original kernel (256 threads, one shared histogram, keys read from memory on every radix pass), for tests and
/// STRATA_TOPK_OLD=1: the same ids.
void qsa_block_topk_ref(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                        const QsaShapes& s, int32_t* ids, void* stream);

// ---- W1: exact hierarchical top-k (workspace-taking, spec section 3) ---------------------------------
//
// The register selector covers at most 1024 * 33 blocks; this parallel hierarchical selector has no such bound.
// It is pure arithmetic in the host view: allocate the workspace before any graph capture, keep it alive for the
// captured execution's lifetime, and pass it per owner/stream.
constexpr int64_t kQsaTopkTileBlocks = 1024;  ///< score blocks per hierarchical tile
constexpr int64_t kQsaTopkHistBins = 256;     ///< radix digits per pass

/// Workspace bytes a hierarchical selection of `nq` queries with a physical stride of `max_blocks` needs.
/// Checked 64-bit arithmetic is the caller's; this is a plain formula shared by every backend.
inline uint64_t qsa_topk_workspace_bytes(int64_t nq, int64_t max_blocks) {
    if (nq <= 0 || max_blocks <= 0) return 0;
    const int64_t tiles = (max_blocks + kQsaTopkTileBlocks - 1) / kQsaTopkTileBlocks;
    const uint64_t hist = (uint64_t) nq * (uint64_t) tiles * (uint64_t) kQsaTopkHistBins;
    const uint64_t qtile = (uint64_t) nq * (uint64_t) tiles * 4u;  // gt/eq totals, admit, output start
    const uint64_t state = (uint64_t) nq * 8u;                     // per-query prefix/above/thr/eq_budget/flags
    return (hist + qtile + state) * sizeof(int32_t);
}

/// Exact hierarchical top-k using caller-owned workspace (`workspace_bytes` >= qsa_topk_workspace_bytes).
/// Returns false (nothing launched, no output written) when the geometry/workspace is unsupported, so the caller
/// can use the reference selector.  SYCL backend only; other backends do not implement it.
bool qsa_block_topk_hier(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                         const QsaShapes& s, int32_t* ids, void* stream, void* workspace,
                         uint64_t workspace_bytes);

/// W2: query-tiled shared-key FP32 scorer, bitwise-identical to qsa_block_scores.  Test/bench entry point;
/// SYCL backend only.  Returns false (nothing launched) on unsupported geometry.
bool qsa_block_scores_tiled(const float* pooled, const float* dead, const float* q_idx, const int32_t* steps,
                            int64_t nq, int64_t max_blocks, const QsaShapes& s, float* scores, void* stream,
                            int64_t active_blocks);

/// Device-aware top-k dispatch (section 1A) with caller-owned workspace.  On SYCL it consults the device profile
/// selectors (`STRATA_SYCL_TOPK=reference|legacy|hierarchical|auto`); `auto` uses the established legacy
/// register kernel for shapes within its capacity and the reference selector beyond it until a long-context
/// winner is qualified for this exact device.  A null workspace disables the hierarchical path.
void qsa_block_topk_ws(const float* scores, const int32_t* steps, int64_t nq, int64_t max_blocks, int64_t cap,
                       const QsaShapes& s, int32_t* ids, void* stream, void* workspace, uint64_t workspace_bytes,
                       int64_t active_blocks);

}  // namespace strata::kernels
