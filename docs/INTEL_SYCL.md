# Intel Arc (SYCL) build

An opt-in third GPU backend, `STRATA_ENABLE_SYCL`, built with Intel's oneAPI DPC++
compiler for Level Zero.  The host sources and tests are unchanged: they speak the
CUDA runtime subset declared in `include/strata/sycl_compat/` and implemented by
`src/sycl_runtime/` on SYCL USM, in-order queues and `sycl_ext_oneapi_graph`.  The
device code is a real rewrite (SYCL cannot compile CUDA kernel syntax), one file per
`src/kernels/cuda/*.cu` under `src/kernels/sycl/`, behind the same public headers.
The plan and the per-call mapping are in [docs/INTEL_SYCL_PORTING_GUIDE.md](INTEL_SYCL_PORTING_GUIDE.md);
the HIP backend (`docs/AMD_HIP.md`) is the structural model.

**Status: the engine runs the Coder model (Qwen3.8-Flash-Next IQ1_M, native pack) end to end on an Arc A770, with
the MTP draft layer and setup's resident RAM mode, at 15-16 tok/s decode (11-12 tok/s with an 8.5K-token prompt)** (see [Model run and speed](#model-run-and-speed)).  What exists, each
verified on an Arc A770:

| Piece | Where | Verified by |
| --- | --- | --- |
| CUDA runtime subset (~90 calls: memory, streams, events, graphs, host USM, errors, device info) | `src/sycl_runtime/` | `sycl_smoke` |
| cuBLAS gemm subset on oneMKL (staging past 4 GiB; its own repeatable kernel for small outputs) | `src/sycl_runtime/blas.cpp` | `sycl_gemm_4g` (bitwise, a double reference, a negative control) |
| NEOX partial RoPE + native (analytic) RoPE | `src/kernels/sycl/rope.cpp`, `native_rope.cpp` | `sycl_rope_parity` — the CUDA/HIP test, unchanged |
| 2-bit sub-expert dequantizer | `src/kernels/sycl/dequant_s2.cpp` | `sycl_dequant_s2_parity` — bit-exact over 200000 blocks |
| Q8_0 / Q8_K activation quantizers | `src/kernels/sycl/quantize_act.cpp` | `sycl_quantize_act_parity` — byte-exact vs ggml |
| bf16 dequantizer (K-quants; i-quants through `iq_kernels`) | `src/kernels/sycl/dequant_bf16.cpp` | `sycl_elementwise_parity` (NaN section) |
| Layer glue: silu, scale/add, gdn gate, rms_norm, embedding gather, f32↔f16/bf16 | `src/kernels/sycl/elementwise.cpp` | `sycl_elementwise_parity` |
| bf16 GEMV (naive / warp / split) | `src/kernels/sycl/bf16_gemv.cpp` | `sycl_bf16_gemv_parity` |
| 2-bit expert GEMV, plain and Q8-hit path (+ quads/fast) | `src/kernels/sycl/s2_gemv.cpp`, `s2_gemv_q8.cpp` | `sycl_s2_gemv_parity`, `sycl_s2_gemv_q8_parity` |
| Shared-expert GEMV family (Q4_K/Q8_K/Q8_0, split, async) | `src/kernels/sycl/s_gemv.cpp` | `sycl_s_gemv_parity`, `sycl_s_gemv_q8k_parity` |
| Shared expert block (swiglu, gate, moe_combine, mmvf) | `src/kernels/sycl/shared_expert.cpp` | `sycl_shared_expert_parity` |
| MoE router top-10 | `src/kernels/sycl/router_top10.cpp` | `sycl_router_top10_parity` |
| Sampler (greedy / one-block / split top-k, coupled draft) | `src/kernels/sycl/sampler.cpp` | `sycl_sampler_parity` |
| Gated delta-net recurrence, convolution, L2/output norms and beta gate | `src/kernels/sycl/gdn.cpp` | `sycl_gdn_parity` — unchanged CUDA/HIP test |
| Gated residual in BF16, FP32 and native modes; fused single/multi-token reads | `src/kernels/sycl/gr.cpp`, `native_gr_norm.cpp`, `native_gr_postops.cpp`, `fused_gr.cpp` | `sycl_gr_parity` — unchanged, including eight-token bitwise parity and changing graph replay |
| QSA cache append/gather, reference indexer, exact top-k, attention and output gates; native scalar/batch indexer | `src/kernels/sycl/qsa.cpp`, `native_qsa_indexer.cpp` | `sycl_qsa_parity` — unchanged, including changing-position graph replay and RoPE scaling |
| Grouped 2-bit experts: per-hit, device-count, multi-token, weight-sharing and CPU-order paths; resident grouping and hit selection | `src/kernels/sycl/s2_expert_grouped.cpp` | `sycl_s2_expert_grouped_parity` — unchanged; `sycl_grouped_cpu_order` adds CPU arithmetic and routing/graph checks |
| Chunked decode attention: FP16, INT8, Q4 and hybrid K8/V4, single and batched | `src/kernels/sycl/qsa_decode_attn.cpp` | `sycl_kv_stream_parity` — unchanged; `sycl_kv_decode` adds an independent FP64 oracle and changing graph replay |
| KV streaming: clock eviction, host-to-slot copies, ring restore and prompt staging | `src/kernels/sycl/kv_stream.cpp` | `sycl_kv_stream_parity` — unchanged, all three formats; `sycl_kv_decode` adds captured eviction and staging checks |
| INT8 and Q4 KV append/gather; 256-point Walsh-Hadamard transform | `src/kernels/sycl/kv_q8.cpp`, `kv_q4.cpp` | `sycl_kv_q8_parity`, `sycl_kv_q4_parity` — unchanged, including byte-exact quantization checks |
| QSA selection: scalar and weight-sharing block scores, reference and register radix top-k | `src/kernels/sycl/qsa_select.cpp` | `sycl_qsa_topk_active_parity` and `sycl_qsa_select_bench` — unchanged; `sycl_qsa_select` adds CPU sorting, FP64 scores and growing-context graph replay |
| QSA prompt attention: scratch-free online softmax for FP16, INT8, Q4 and hybrid K8/V4 | `src/kernels/sycl/qsa_prompt_attn.cpp` | `sycl_qsa_prompt_attn_parity` — unchanged; `sycl_kv_decode` adds FP64 and changing graph/refusal checks |
| I-quant and UD-Q4_K_XL kernels: dequantizers (f16/f32, embedding rows, interleaved gate/up), q8_1 quantizer, MMVQ (per-column and decode-once), grouped native experts (group stride, fused SwiGLU + q8_1) | `src/kernels/sycl/iq_kernels.cpp` | `sycl_iq_parity` — unchanged, against gguf-py on generated fixtures; `sycl_iq_multi_parity`, `sycl_native_grouped_parity` — unchanged, bitwise |
| Native MMVQ: Q2_0, Q4_0, Q5_0, Q8_0, Q3_K, Q4_K, Q5_K, Q6_K, IQ4_NL, IQ4_XS, ncols 1–8, exact and upstream layouts | `src/kernels/sycl/native_mmvq.cpp` | `sycl_mmvq_multi_parity` — unchanged, bitwise with its negative control; `sycl_iq_parity` |
| PLE block (BF16/native key and value, legacy and native post-ops, the native T-token batch, history advance) | `src/kernels/sycl/ple.cpp`, `native_ple_postops.cpp` | `sycl_ple_block`: every stage against an FP64 transcription of `ple.hpp` (≤1.3e-7 relative with a BF16 key, ≤6.3e-3 with the Q8_0 key's q8_1 activations), the batch bitwise equal to single tokens, capture replay bitwise. The unchanged `ple_parity` needs captured model fixtures and is built but not registered without them |
| Control vector | `src/kernels/sycl/cvec.cpp` | `sycl_cvec_parity` — unchanged, including bitwise equality with the fused read's folded write |
| Engine-only kernels: verify window (GDN conv/ab/recurrence multi-token, commit, flag waits, mapped copies, MTP helpers), fused GDN, native router, MoE combine, QSA norm/gate/score, GDN step and preprocessing, short-context FlashAttention | `src/kernels/sycl/verify_kernels.cpp`, `fused_gdn.cpp`, `native_*.cpp` | `sycl_engine_kernels`: verify kernels bitwise equal to T single-token kernels; router ids exact against a host top-10; MoE combine and QSA score bitwise against their rounding contracts on the host; FlashAttention ≤1.9e-7 and GDN ≤1e-7 against FP64; the flag wait held and released by a host thread |
| Prefill kernels (hyper-connection, GDN conv/recurrence, routing, blob dequant, SwiGLU, RoPE, KV append) | `src/prefill/sycl/kernels.cpp` | `sycl_prefill_native_batch` (the HIP test, no HIP code in it); the model run is still to come |
| Engine host: device, pinned arena (host USM), core, engine, prefill GEMM (oneMKL), `strata generate` | `src/core/sycl/device.cpp`, `pinned.cu` (as C++), the shared host sources | builds; `sycl_file_expert_source_test`, `sycl_expert_layout_test`, `sycl_expert_cache_*`, `sycl_native_dense_ple_key_test`, `sycl_pinned_shared_test`, `sycl_ple_reader_selftest` pass |

The temporary `native_mmvq` link stubs are gone from `shared_expert.cpp`: the real
family lives in `native_mmvq.cpp`, so `native_mmvq_supported()` now reports the
CUDA list and `shared_expert_multi` runs.

The engine targets (`strata_core`, `strata_engine`, `strata_prefill`, `strata`) build on SYCL from the same host
source lists as CUDA/HIP (`STRATA_CORE_HOST_SOURCES`, `STRATA_ENGINE_SOURCES` in CMakeLists.txt).  The CUDA-only MMQ
and fused prompt experts are not built; prefill takes its dequantize + oneMKL GEMM path.  The pinned arena is one
Level Zero host USM allocation (`PageBacking::PinnedBySycl`): SYCL cannot register existing memory, and the kernels
read mapped expert blobs.  A kernel is done when its existing parity test passes on this backend; kernels no
existing test calls are checked by the supplementary tests above.

Kernel-level measurements on the A770:
- Int8 dot products: IGC turns `strata_dp4a`'s four byte products into the hardware `dp4a` instruction (16 per
  16-dot loop in the generated assembly), so the quantized kernels do not pay for SYCL's missing intrinsic.
- Launch cost: a 2,560-float kernel costs 3.3-3.4 µs launched directly and 1.3-1.9 µs replayed from a SYCL graph.
- `gpu_stamp` (the opt-in `STRATA_VERIFY_PROFILE`): the A770 has no device-wide clock, only a per-EU sub-group counter
  that is not comparable between kernels.  Without a device clock the stamp reads the HOST's clock: a host thread
  keeps writing steady_clock nanoseconds into a host USM word and the stamp kernel reads it uncached
  (`src/kernels/sycl/gpu_stamp.cpp`; a 200 ms sleep reads as 200.3 ms).  The thread keeps one core busy, so only
  the profile mode starts it.  The one-shot `strata generate` now prints the stage table too ("verify GPU stages").

## Model run and speed

Measured on this machine: Arc A770 16 GB on a chipset PCIe 3.0 x4 slot (1.9 GB/s host to device, measured by the
engine's probe), Ryzen 5 5600 (6 cores, AVX2), 31 GB RAM, the Coder pack with `--mmap-experts` (the experts read in
place from the GGUF), `--expert-cache auto` (5,425 VRAM slots, 10.3 GiB), a 23-token prompt, 128 new tokens, greedy.

| Configuration | Decode | Notes |
| --- | --- | --- |
| first run, `--spec 4`, suffix drafts | 0.78 tok/s | before the kernel work below |
| + fp_exact, 16-wide MMVQ | 2.70 tok/s | |
| + the fast router | 3.35-3.43 tok/s | |
| `--spec 2` | 4.39 tok/s | suffix drafts accept 8%: a smaller window costs less |
| + 16-wide grouped experts | 4.58 tok/s | |
| setup's config: `--spec 4 --spec-min-p 0.5 --mtp` | 6.91-7.20 tok/s | MTP drafts accept 84-89%, 3.4 tokens per round |
| same + `--spec-split` | 6.49-8.25 tok/s | mean of 4 runs 7.44 vs 7.04 (3 runs) without; setup passes it for SYCL |
| + `--resident-budget-gib 10` | 9.52 tok/s | the experts the GPU does not hold, hottest first, copied into RAM |
| + `--resident-experts` (setup's low-RAM choice for one GPU) | 9.99 tok/s | 14.98 GiB in RAM; at least 3.2 GB stayed available |
| + the adaptive tier capped on a slow link | 12.06-12.31 tok/s | 8 swaps per adaptation instead of 96 (below) |
| + batched routing, 16-wide GR reads, the payload check | 13.18-13.38 tok/s | 14.3 GiB resident; deterministic over repeated runs |
| + 64-bit uncached transfers | 15.33 tok/s | 14.1 GiB resident; 12.60 when only 10.9 GiB fit (the RAM other programs leave varies) |
| + the long-prompt fixes (below) | 15.15-15.85 tok/s | 15.2-15.4 GiB resident; 8,572-token prompt: prefill 114-129 tok/s, decode 11.1-11.5 tok/s |
| + Q6_K dense weights reordered (below) | 15.47-15.57 tok/s | fixed placement (`--resident-budget-gib 14 --adapt-swaps 0`): 15.04-15.22 before; GPU wait 55.6 -> 41-43 ms per round |
| + 7 expert-pool workers (setup, SMT CPUs) | 15.62-15.73 tok/s | 14.71-14.79 with the default 5 in the same session; 8,572-token prompt: prefill 111-130 tok/s, decode 11.5-11.8 tok/s |
| + the pool's finer row split, faster IQ3_S/IQ2_S decode (below) | 15.50-16.14 tok/s (6 runs) | pool 108-120 ms per round, from 120-135; the GPU side of the split window is now as long |
| + setup's draft vocabulary (`mtp/rt/draft_vocab.bin`) | 15.79-16.34 tok/s (one run 12.87: a stall outside the rounds, with 22 GB of this PC in swap) | MTP drafting 18.2 -> 11.8-14.1 ms per round (the 106K-token draft head instead of the 248K-token one) |

Run to run, decode varies by ±12% (the draft acceptance and the page cache vary), and between sessions by a few
percent more (the RAM other programs leave decides the resident size).  The reply is now the same on every run (below,
"Repeatable replies").  Prefill runs at about 3.3 tok/s for this short prompt.
About half of its 7.5 s is streaming experts from the host (3.8 s, 1,609 experts).

What each step fixed:
- **Exact float operations.** `sycl::ext::intel::math::fadd_rn`/`fmul_rn` are emulated in software on DG2 (16x
  slower than an add).  `src/kernels/sycl/fp_exact.hpp` makes them plain operators and every kernel library
  builds with `-ffp-contract=off`.  IGC only fuses within a statement, and the code never writes `a * b + c` in
  one statement, so the rounding of each operation is the same.  `fdiv_rn`/`fsqrt_rn` stay (IGC's defaults are not
  correctly rounded).
- **Sub-groups of 16.** DG2 has 128 registers per thread, so a SIMD32 kernel has 32 per lane. The CUDA-shaped
  decode-once kernels spilled 2-5.5 KB per thread (IGC dumps: `IGC_ShaderDumpEnable=1`).  Native MMVQ, the i-quant
  MMVQ and the grouped experts now run 16-wide.  Each work-item carries the CUDA warp's lanes l and l + 16 (or
  32 / W of them), each with its own accumulator, and the butterfly is replayed in the CUDA order
  (`warp_sum_w`).  The results are bitwise the 32-lane kernels' (unchanged parity tests, plus a byte hash of
  old and new outputs).  Q6_K head at 6 columns: 29 -> 6.5 ms.  Grouped experts at the Coder's geometry
  (28 experts, 36 entries): 644 -> 399 µs per layer call (IQ2_S/IQ4_NL), 1,399 -> 611 µs (IQ3_S/Q2_0).
- **The router.** The Coder has 256 experts; the native router serves only 512, so decode used the
  reference router, whose FP64 is emulated here (410-760 µs per call, a fifth of the GPU time).  The HIP
  file's fast kernel (a tree sum that brackets the serial one, selection in one sub-group) is now the SYCL
  default too: 130 µs, bitwise the portable kernel's (`sycl_router_fast`, 82k rows).
- **The RAM tier.** The model is 58 GB on disk and this PC has 31 GB of RAM, so with `--mmap-experts` only
  16.9 of the 27.6 GB expert shard stayed in the page cache.  Each round faulted about 1 GB from NVMe, and the
  CPU pool waited on the disk (`jobs` 187 ms against 70 ms of compute per round).  Setup's resident mode
  copies the experts the GPU does not hold into RAM once: 49 blob reads from the file in a whole run.
- **The adaptive tier on a slow link.** Every 4 rounds the engine swaps up to 96 hot experts into VRAM.  At
  1.9 GB/s that is ~180 MB the next window waits for (51 ms per round).  With 96, 32, 16, 8 and 0 swaps, decode
  ran at 9.29, 9.92, 9.82, 10.04-10.32 and 9.93 tok/s.  The engine now caps the swaps when its PCIe probe
  measures a slow link: 96 x (GB/s / 25), at least 8 (`--adapt-swaps N` overrides).  That is 8 here, and it
  leaves fast links at 96.  This is a property of the link, not of SYCL.
- **MTP and the split window.** These are the engine's own features.  With them the CPU expert pool is the
  largest part of a round (230-290 ms of 400-480 ms): 10-13 experts per layer miss the VRAM cache and run on
  the 6-core CPU at about 18 GB/s.  The 1.9 GB/s link makes uploading them slower than computing them on the CPU
  (the engine picks pcie_frac 0.05).

- **Batched routing for any expert count.** The verify window batched the router (one GEMV, one top-k for the
  window's rows) only for the native 512-expert router.  The Coder's 256 experts went per token: 4 GEMVs and 4
  router launches per layer at window size 4, 0.5 ms of each GDN layer's GPU segment.  The batch now covers any
  expert count whenever the per-token GEMV is the same kernel (`bf16_gemv_fp32_mmvf`), with `router_top10` over
  the window's rows: the GDN segment went from 3.33 to 2.82 ms, with byte-identical output.

- **16-wide GR reads.** The multi-token GR down and up kernels spilled at SIMD32 (1,632 and 3,584 bytes per
  thread).  On 16-wide sub-groups, each work-item carries CUDA lanes l and l + 16 and the 32-lane butterfly is
  replayed (`warp_sum16` in `fused_gr.cpp`): 0.279 -> 0.185 ms per 4-token read, outputs byte-identical to the old
  kernels (a direct old/new comparison, plus gr_parity's multi/single bitwise check).

### A race in the doorbell, and its fix

Repeated runs did not always print the same tokens (2 of 9 differed, diverging at token 6 or 29).  A diagnostic
checksum showed why: the ring can become visible to the host before the last activation stores of the same
kernel (14 of 2,832 rings in one run), despite the system-scope fences, so the CPU pool occasionally computed on a
partly stale payload.  `doorbell_publish` now also writes, next to the sequence number, the exact checksum of the
payload (the wrapping sum of the bit patterns of the activations, ids and weights) and the ring it belongs to.
The host waits until both match its view (`doorbell_wait_payload`, elementwise.hpp) before the pool reads, in the
verify window and both token-graph loops.  The sequence word gets 64 bytes for this (layer.cpp's doorbell).  Five
runs that diverged before, and three MTP runs, now print the same tokens; speed is unchanged.  CUDA/HIP keep
their ordering and skip the check.

### Long context

The numbers above are short-context (a 23-token prompt, ~150 positions used of a 4096-token window).  With an
8,572-token prompt (a source file plus a question) and a 16K window, setup's configuration with a 6 GiB RAM budget
answered correctly: prefill 114-129 tok/s (time to first token 80-89 s), decode 11.1-11.5 tok/s.  Getting there
fixed four problems that only long prompts reached:
- **oneMKL past 4 GiB.**  The prompt path carves its buffers from borrowed expert-cache slots - the top of one
  ~10 GiB allocation.  oneMKL 2026.1's BF16 GEMM reads the last, partial tile of a B operand from the wrong place when
  it lies more than 4 GiB into its allocation: a 1,845-column B past 4 GiB gave exactly its last 21 columns wrong
  (215,040 outputs), none below 4 GiB.  In the engine those columns read stale expert bytes, layer 0 came out
  non-finite, the router picked experts 0-9 for every token, and the reply was "!!!..." or a crash.  The BLAS shim
  now knows every device allocation (`device_offset_end`) and stages a GEMM's operands that lie past 4 GiB through a
  96 MiB buffer of its own, in pieces of whole columns (`sycl_runtime/blas.cpp`; `sycl_gemm_4g` checks it bitwise
  and fails with `STRATA_SYCL_GEMM_NO_STAGE=1`).  Borrowing then works as on CUDA/HIP: 8192-token chunks and the
  full cache.  (`STRATA_SYCL_PREFILL_OWN=1` keeps own buffers instead, at most 2048-token chunks, the reservation
  below exact.)
- **The own buffers' VRAM reservation** (the opt-in mode).  The expert cache's auto size reserved a linear estimate, which left
  an 8192-token chunk's blob ring out; the over-committed card hung a verify window in 2 of 3 long runs.  SYCL now
  reserves `Prefill::bytes_needed` exactly (6 of 6 long runs then completed).
- **Cross-stream waits in the prompt path** were device-side barriers: the engine sat in a semaphore wait while
  experts streamed over the slow link (past the xe driver's 5 s job timeout: the engine was reset mid-prompt) or two
  queues waited on each other (a deadlock).  Outside a capture `cudaStreamWaitEvent` now waits on the host (the
  event's work is already submitted); captures keep the barrier.
- **The doorbell check's slot.**  A split window publishes two rings before the host reads the first; the payload's
  tag and checksum are kept per ring (modulo 4), not in one slot.

### Repeatable replies

Repeated runs of the same prompt used to reply in different (equally sensible) words, 4 of 17 short runs one way
and 13 the other, even with a fixed expert placement.  Hashing the verify window's buffers stage by stage found the
first difference in the GDN state the prompt path leaves behind, and in it the `ssm_alpha`/`ssm_beta` projections:
oneMKL 2026.1's BF16 GEMM is not repeatable when its output is small against k.  Back-to-back calls with the same
inputs gave different low bits for m <= 512 at k 2560 or 10240 and many column counts (m 48, k 2560: n 1-3, 96, 128;
m 16, k 10240: every n up to 256), never for m >= 1024 - its partial sums over k meet in a varying order.  (A
benchmark whose inputs have few mantissa bits cannot show this: every order then gives the same sum.)  The BLAS shim
now computes an output of m x n <= 65536 with its own kernel: each output one fixed-order dot product, k split over a
work-group's 8 sub-groups and their sums added in sub-group order (`sycl_gemm_4g` checks it against a double
reference; `STRATA_SYCL_GEMM_ONEMKL_SMALL=1` gives those GEMMs back to oneMKL).  It costs nothing measurable on the
prompt path: 0.018 ms for the 48 x 1 x 2560 projection (oneMKL 0.019), and an 8,572-token prompt ran at 111-130
tok/s.  Four runs with a fixed placement, and four with setup's (resident sizes 15.18-15.46 GiB), each printed the
same 128 tokens; two 8,572-token runs printed the same reply.

The same search found a bug in the staging above: it copied C back in whole columns of ldc, so a GEMM writing m of
ldc rows (the alpha and beta projections share one buffer, interleaved) wrote the scratch's stale rows over the
other's outputs.  It copies C's m rows now (`sycl_gemm_4g` has the case, and fails with the old copy).

The xe driver kills a GPU job that runs longer than its job timeout (`job_timeout_ms`, 5000; at most 10000 via
`/sys/class/drm/cardN/device/tile0/gt0/engines/ccs/job_timeout_ms`, as root).  A verify window is one job whose spin
waits last as long as the CPU pool: with the experts on disk instead of in RAM (`--mmap-experts` after a long prompt
evicted the page cache) a first window ran past it and the engine was reset.  The RAM tier keeps the pool short.

### The per-layer handoff over the link

Measured with `STRATA_VERIFY_LAYER_MS=1` and a standalone model of the handoff (one graph of 48 layers and a host
thread that answers each doorbell), the GPU segment between the host's flag and the next doorbell is, per layer:
- **Publishing the activations: ~0.21 ms per token row** (0.25, 0.47 and 0.87 ms for 1, 2 and 4 rows).  The
  doorbell's 4-byte uncached stores drain at about 12 million per second, and its system fence waits for them.
- **Reading the CPU experts' rows back: ~0.33 ms** at window size 4 (uncached 4-byte loads, about 470 MB/s).
- The flag wait itself: ~20 µs.

That was about 1.2 ms per layer with 4-byte accesses (55-60 ms of a 4-token window), set by kernel-driven traffic
to host memory on this link; with 64-bit accesses it is about 0.65 ms.  What was tried (the benchmarks built with
the libraries' flags, see the first item):
- Benchmarks must be built with the kernel libraries' ocloc options (`-ze-opt-greater-than-4GB-buffer-required`):
  without it IGC may use stateful addressing, the per-access cache hint is lost, and an uncached access silently
  becomes a cached one.  Several "fast" variants first measured that way were stale.
- **64-bit uncached accesses keep the hint and halve both transfers** (each access is one PCIe transaction, at a
  fixed rate): the doorbell writes the activations as float pairs (0.93 -> 0.44 ms for 4 tokens) and the row
  copies read pairs (0.35 -> 0.19 ms); a GDN layer's segment went from 2.82 to 2.07 ms with byte-identical output.
  `sycl::vec` stores through the cache-control `annotated_ptr` lose the hint (`store.ugm.d32x4...wb.wb` in the ISA)
  and their data reaches the host only when the graph ends.
- More work-items do not help: stores from 32 work-groups (with fences) and a flat copy kernel (one work-item per
  float) were correct and as slow; stores from a separate kernel without a fence in the ringing thread arrived
  after the ring (14 of 48 payloads stale).
- Memcpy nodes: device-to-device copies after kernels are ordered correctly (48 of 48 layers).  But a
  **host-to-device copy after a spin wait read the host buffer before the wait released** (1 of 2 layers wrong,
  46 of 48), and a device-to-host copy followed by a ring kernel lets the ring overtake the data (an engine run with
  it crashed on stale expert ids).  Copies of host data the CPU writes mid-graph stay kernels.
- The host pulling the activations with its own copy queue deadlocks: the copy cannot run while the graph's spin
  kernel holds the GPU.
- ESIMD's L3 writeback fence hung the device.

The CUDA design of these transfers stays.  The remaining levers are the link (a CPU-attached x16 slot) and
fewer bytes per handoff.

Where the time went at 12 tok/s (before the items below): a round (~274 ms, 3.4 tokens) is ~120 ms of waiting on the GPU, ~120 ms in the CPU
pool (18 experts per layer on 6 AVX2 cores) and ~25 ms of MTP drafting.  Two cautions about the in-engine profilers.
`STRATA_VERIFY_PROFILE` is exact in isolation (a stamp pair around a graph of GEMVs read 0.255 ms for a measured
0.256 ms).  In the engine it charged 2.4 ms per layer to the QKV GEMV, while skipping that GEMV saved only 0.33 ms
per layer of GPU wait.  The second caution is in the tools list below.  The A/B of round time is the reliable
measure.  In isolation the VRAM-hit experts are the largest kernel.  Their remaining cost is 64-bit address arithmetic (DG2 has no 64-bit integer add: the
IQ4_NL down kernel has 200 `addc` and 799 `mov` for 89 loads), plus 2-byte loads from the 18-byte IQ4_NL/Q2_0
blocks.  A weight reorder at cache upload, as llama.cpp's SYCL backend does, is the next step there.
Also left: the fused GR reads (about 50 ms per window).

- **Q6_K dense weights reordered.**  The dense GEMVs' Q6_K matrices (attn_qkv in 22 layers, ssm_out in 29, attn_gate,
  attn_q, the 248K-row output head: 1.7 GB) ran at 81-115 GB/s: a 210-byte block leaves its 32-bit words 2-byte
  aligned, so each was two 16-bit loads.  They are now uploaded as llama.cpp's SYCL backend does it, the matrix's ql,
  qh, scales and d each one array (`native_q6_k_reorder`, registered per device pointer; native_mmvq.hpp), and the
  GEMV reads every word with one aligned load in the same lane order: bitwise the same outputs (`sycl_q6k_reorder`,
  1-8 columns and the prompt path's dequantize).  qkv at 1 / 4 columns: 0.190 -> 0.110 ms / 0.255 -> 0.180 ms; the
  head: 4.28 -> 2.41 ms / 5.56 -> 4.10 ms.  The GPU's share of a round fell from 55.6 to 41-43 ms and the reply
  stayed the same.  The shared expert's small Q6_K matrices keep the GGUF layout (their kernels read blocks), and
  `STRATA_SYCL_Q6K_GGUF=1` keeps every matrix in it.
- **The CPU pool.**  Now the largest part of a round: at 15.7 tok/s a round (3.4 tokens, ~214 ms) is ~120 ms of pool,
  ~60 ms of GPU wait (including the adaptive tier's uploads) and the MTP draft.  The pool runs i-quant gate/up rows
  (IQ3_XXS, IQ3_S, IQ2_S) at ~11 GB/s and IQ4_NL/Q2_0 down rows at ~17 GB/s, while the same 5 threads stream 36 GB/s
  from RAM: it is bound by the AVX2 decode, not the memory.  One more worker than the 6 physical cores, on an SMT
  sibling (`--pool-workers 7 --pool-affinity auto`), measured 15.62-15.73 tok/s against 14.71-14.79 (8, 9, 10
  workers: 15.8-16.2 in single runs, 11: 13.35); setup passes physical cores + 1 on SMT CPUs for SYCL.  The RAM copy
  is now allocated 2 MiB aligned and marked for transparent huge pages (`STRATA_RESIDENT_THP=0` leaves it unmarked):
  with the page cache filling RAM the kernel's default defrag gave it none, and huge pages read ~10% faster, but here
  it still got few (234 MB of 13 GiB) - free RAM is the limit.
  Two pool changes (`src/kernels/cpu/`, shared with CUDA/HIP; every row is still computed whole by one thread with
  the same instructions' results, so the reply did not change):
  - **The row split.**  A phase (gate/up, then down) split its rows into 3 equal ranges per thread, but rows cost
    more the more tokens their expert holds, and an SMT sibling runs slower: the threads were busy only 76% (gate/up)
    and 70% (down) of each phase.  With 12 ranges per thread, 90% and 89%; the pool's time per round fell from 141 to
    123 ms (6 workers; 134 -> 124 with 7).  `STRATA_POOL_TASKS_PER_THREAD` overrides it.
  - **The i-quant decode.**  Most CPU experts hold one token of a window (19.6 distinct of 24.3 routed per layer),
    and one token runs ggml-cpu's `vec_dot`; two or more run the multi-token kernels (`iq_avx2.cpp`).  One thread,
    640 x 2560 gate+up rows from RAM, at 2 / 4 tokens: IQ3_S 1.80 / 1.57 -> 2.71 / 2.16 GB/s (its 9-bit grid
    indices built in one vector and gathered, instead of per-lane scalar arithmetic), IQ2_S 2.80 / 2.37 -> 3.46 /
    2.59 GB/s (ggml's sign and scale decode, once per half and block), IQ3_XXS 3.14 / 2.45 -> 3.19 / 2.56 (the
    cheaper sign vector).  Bitwise the same sums (checked on random rows).  Gathers measured slower for IQ3_XXS and
    IQ2_S on this Zen 3; ggml's single-token dots stay (faster than these kernels at one token for IQ3_S and IQ2_S).
  - Six independent threads running ggml's IQ3_XXS dot reach 17.8 GB/s here (22.7 for IQ2_S); 7 or 12 do not go
    faster.  Huge pages made no difference to them.

  The earlier rows of the table ran without `mtp/rt/draft_vocab.bin`, which setup copies from `data/` (that MTP
  folder was built by hand); with it the drafter's head covers 106,299 tokens (212.9 MiB) and the reordered Q6_K
  head is gathered array by array (`mtp.cpp`): the same drafts accepted (92 of 111) and the same reply as the GGUF
  layout's gather, 11.8 against 14.3 ms of drafting per round.
  `STRATA_VERIFY_PROFILE` with 7 pool workers hung a window (the profiler's host-clock thread is one more spinning
  thread; the GPU was reset by the driver and recovered); with 5 workers it runs.  Its in-engine stage times are not
  reliable while the pool saturates the CPU (the clock word is written by a thread that then gets less time).

  With the pool at ~115 ms the GPU side of the split window is about as long, so a faster pool alone no longer
  shortens a round: `--adapt-swaps 0` against the default 8 measured 16.14 / 15.84 against 15.71 / 15.85 tok/s.

Hardware notes: on this board the A770 is behind the chipset at PCIe 3.0 x4.  A CPU-attached x16 slot would make
the expert uploads and prefill streaming about 8x faster, and the engine would then put more misses on the GPU.

Tools:
- `STRATA_VERIFY_LAYER_MS=1`: per window size, the host-clock GPU segment before each doorbell and the pool time,
  by layer kind ("verify layers" in the run summary).  This is the reliable in-engine timing.
- `STRATA_SYCL_DEBUG_SKIP=name1,name2` (debugging only, the results are wrong): while a graph is captured, a caller
  whose symbol contains one of the names gets a scratch queue, so its kernels stay out of the graph.  The change
  in the layer segment is that caller's cost in the real graph (router: 0.50 ms per GDN layer; fused GR read:
  0.37 ms).
- `STRATA_SYCL_Q6K_GGUF=1`: the Q6_K dense matrices in the GGUF block layout (the A/B arm of the reorder).
- `STRATA_SYCL_GEMM_ONEMKL_SMALL=1`: small-output GEMMs on oneMKL again (not repeatable, see above).
- `STRATA_POOL_TASKS_PER_THREAD=N`: the CPU pool's row ranges per thread in a native phase (default 12).
- `STRATA_SYCL_QSTATS=1`: submissions per stream, kind (kernel, memcpy, graph, host function; `@c` = recorded into
  a graph) and call site, printed at exit.  Two queues do not share the A770: while one queue runs a kernel,
  even one work-group, a kernel on another queue cost 300-380 µs instead of 25 µs (measured).  So decode should
  keep its GPU work on one queue.  In this run it does: during a window, the other streams carry only
  copy-engine memcpys (which did not slow a kernel stream) and the MTP drafter, which runs between windows.
- `STRATA_SYCL_TRACE=1`: every launch records its index with an uncached store; on a hang the engine prints
  the launches around the last one the GPU reached.  `STRATA_SYCL_TRACE=profile` also samples that index every
  ~20 µs and ranks callers by time at exit.  The ranking is only exact on one in-order queue; the last launch of a
  round also collects the idle time until the next round.  With the MTP drafter on, a traced decode hung twice
  in a flag wait before the doorbell payload check existed; with the check, a traced MTP run completes.
- Intel's PTI (`libpti_view`) times kernels outside graphs, but a decode run under it hangs in the first spin
  wait.

## Remaining kernel batches

Batch 2 is complete: GDN, GR, QSA core and grouped 2-bit experts are ported;
their existing parity tests pass on the A770.
GR covers BF16, FP32 and native activation modes at the real model geometry
(2560/4/320), final mixers without injection, and captured graphs. Its fused
multi-token path shares weight loads across up to eight tokens using a 1280-float
activation tile (40 KiB of local memory at eight tokens); all outputs are
byte-identical to separate fused single-token calls in the existing fixture.
The native BF16 MMVF projection already lives in `shared_expert.cpp` and is reused.

QSA core retains byte-exact cache copies and deterministic top-k tie selection,
reference pooling in double precision, and native pooling with F16-rounded keys.
The native scalar and batch indexers share the original float reduction and
rotation order. Graph replay reads positions/counts from device buffers. Both sparse and dense
QSA selftests pass on the A770; native batch and sequential indexer states are
bit-identical for no scaling, linear scaling and YaRN in the existing fixture.
The runtime now exposes recorded graph node counts, stable handles and supported
command types through the installed SYCL graph API; kernel launch-parameter
inspection remains unsupported. Event recording accepts CUDA's default stream
argument. Both changes let the existing QSA test run unchanged.

The SYCL fused read uses the plain variant. CUDA's split/staged `cp.async` variants
and the opt-in `STRATA_GR_V3` implementation are not selected. Optional device
profiling stamps are refused until the verification timestamp helper is ported.
No performance claim is made from these correctness tests.

The full suite passes 43 of 44 tests (A770, 2026-10-04, 109 s): all 25 SYCL kernel
parity tests (the 21 earlier ones plus `iq`, `iq_multi`, `native_grouped` and
`mmvq_multi`), the i-quant fixture generator, the unchanged selection harness, all three
supplementary tests (CPU-order/routing, KV/decode/prompt graph replay and QSA selection
oracles), the runtime smoke test and the host-only tests pass. The existing `expert_multi_test` failure is the CPU's missing AVX-512
extensions.

Grouped experts retain the selectable old/new kernels, activation alignment
fallbacks, device-count guards and weight sharing across routed entries. The
unchanged parity test checks byte-identical outputs and scratch in both dispatches,
FP16 and FP32 scale contracts, and mapped-host weight blobs. A supplementary
SYCL test checks CPU-order gate/up projections against a scalar transcription of
the CPU reduction, final down projections against the double dot-product oracle,
and captured/direct results with changed inputs. It also checks single-warp and
128-entry hit selection, including invalid IDs and an empty captured selection.

The native GR and QSA helpers moved forward from Batch 3 because the current
parity tests exercise them. The existing parity sources remain unchanged.
`qsa.cu` implements its core FP16 append/gather operations locally. The grouped expert
source uses `verify_kernels.hpp` for the window bound, not a device implementation.

Batch 2b is complete: decode and prompt attention, KV streaming and QSA selection
are ported, including INT8/Q4 storage helpers needed by the unchanged streaming
parity test. On the A770, streamed and fully resident attention are byte-identical through 40,000 appends
and 306 query batches per format; ring-restored attention is also byte-identical.
The supplementary test checks FP64 decode results (maximum absolute error below
1.6e-7 in its fixture), single/batched equality, changing graph replay, empty and
nonresident selections, and hybrid K8/V4. It also captures the streaming resolver
and copy kernels together, exercising hits, eviction and empty selections in all
three formats, and checks prompt staging against the host bytes.
The resolve block retains 1,024 threads and uses work-group barriers that fence
both local and global memory, including the slot stamps used by the clock sweep.

QSA selection is ported. The unchanged active-bound test passes all 16 cases,
and the unchanged selection harness verifies register/reference IDs. The
supplementary test compares all IDs to a stable CPU sort across 57 changing-context
replays, including ties, NaNs, infinities, signed zeros and partial blocks, up to
262,144 cells. Captured graphs omit the live bound; a valid explicit bound can
choose the register kernel without changing the score-row stride. The scorer's
multi-query and bounded paths are byte-identical in the fixture, with maximum
absolute error below 2.9e-6 against FP64 for scores without the tail boost.
`STRATA_TOPK_OLD` and `STRATA_SCORES_MULTI` retain their debug dispatch controls.
CUDA TF32/AMD WMMA scoring and CUDA thread-block clusters are unavailable on this
backend: their capability probes launch nothing and callers retain scalar scores
and the one-work-group top-k paths. No selection performance claim is made.

Prompt attention keeps one work-group per (query, KV head), walks selected cells
in 32-cell tiles and retains the online softmax state on the device. Each tile's
value sum is formed separately and merged once, avoiding a long serial FP32 sum
across every selected cell. It requires no external scratch or merge kernel.
The SYCL implementation uses full FP32 subgroup dot products instead of CUDA
matrix instructions. Intel matrix acceleration remains an optimization to do;
passing parity does not establish prompt performance or end-to-end model quality.
The unchanged parity test compares against decode attention and FP64 in FP16,
INT8 and Q4 pools at short, sparse and transition contexts; the 32K-context
fixture uses 2,048 queries. The supplementary
fixture adds hybrid K8/V4, empty and missing selections, an entirely masked first
tile followed by valid cells, changing graph replay, and calls that must refuse
unsupported geometry or incomplete pools without writing output.
`STRATA_PROMPT_ATTN_Q4=0` returns false so callers can use decode attention.
CUDA-specific V1/async pipeline controls do not select variants in this port.

Batch 3 has started: the i-quant kernels and native MMVQ are ported. On the A770 (2026-10-04):

- `sycl_iq_parity`: the dequantizers of all ten fixture formats (IQ2_XXS, IQ2_XS, IQ2_S,
  IQ3_XXS, IQ3_S, IQ1_M, IQ4_NL, IQ4_XS, Q2_0, Q3_K) match gguf-py exactly (relative
  error 0.00e+00); native MMVQ is within 4.6e-3 to 5.7e-3 relative of the float product
  (the test's bound is 2e-2), and every column of a 2- to 8-column call is bitwise equal
  to a single-column call.
- `sycl_iq_multi_parity`: the decode-once MMVQ and grouped kernels are bitwise equal to
  the per-column ones for every format, ncols 1–8 and 11.
- `sycl_native_grouped_parity`: the group-strided launches and the fused SwiGLU + q8_1
  pass are bitwise equal to the v1 launches for every gate/up and down pair.
- `sycl_mmvq_multi_parity`: exact multi-column calls are bitwise equal to single-column
  calls, and the upstream layout differs where its negative control requires.

Bitwise agreement between kernels needed every float product and sum of the dots pinned
through `fmul_rn`/`fadd_rn`/`fsub_rn`, the same lesson as the grouped 2-bit port. On SYCL,
native MMVQ's single-column call is its multi-column kernel with one column (the CUDA file
states that its ncols = 1 kernels are that kernel's exact layout). The second q8_1 quantizer
reuses `quantize_q8_1_rows`, which is the same llama.cpp kernel. No performance claim is
made.

Batch 3 is complete: PLE, the control vector, the native and fused engine kernels and the verify window are ported
(see the table). The CUDA cp.async key-head GDN recurrence (`gdn_rec_kh_kernel`, taken on sm_80+ cards that hold
its 64 blocks) is not: prefill uses the software-pipelined column kernel, which CUDA keeps bitwise equal to it.
CUDA's raw-bit TF32 QSA scorer becomes a scalar FP32 dot, as on HIP.

## Setup

Pin the toolchain in `tools/sycl/versions.txt`; work on Linux with the `xe` driver.

```sh
. tools/sycl/env.sh    # compiler/MKL paths, ONEAPI_DEVICE_SELECTOR, allocation env
```

`env.sh` pins the inference card (`ONEAPI_DEVICE_SELECTOR=level_zero:<n>`; check
`sycl-ls` — the order is not OpenVINO's), allows >4 GB device allocations
(`UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1`), enables free-memory queries
(`ZES_ENABLE_SYSMAN=1`) and turns on FP64 emulation (`NEO_FP64_EMULATION=1`).
Raise the locked-memory limit as the CUDA build expects.

## Toolchain pitfalls (each one measured on this machine)

- **DG2/BMG have no FP64 hardware.**  Two ported kernels use double arithmetic
  because their CUDA reference does and the parity contract is byte-exact
  (`quantize_q8_0`'s scale division, `silu`).  The AOT image is built with
  `-cl-fp64-gen-emu` (cmake/sycl_backend.cmake) and the driver must report the
  fp64 aspect at runtime (`NEO_FP64_EMULATION=1`, env.sh) or launch is refused.
- **icpx's default `-ffp-model=fast` is not IEEE on the HOST either**: its
  vectorized loops came out 1 ulp off a plain `x[i]*s` on ~45% of elements,
  which fails the parity tests' host oracles (g++ and the GPU agree with each
  other).  The build sets `-ffp-model=precise`.
- **IGC's default f32 division is not correctly rounded** (1 ulp off on ~28% of
  variable divisors; division by a constant is fine).  Where the CUDA reference
  relies on IEEE division (`quantize_q8_K`'s `iscale` and `d`, the hit-path
  quantizer's `1/s`) the SYCL kernels call
  `sycl::ext::intel::math::fdiv_rn` instead of `/`.
- **libm float calls in device code (`logf`, `powf`, …) become imported device
  symbols**, and the oneMKL link adds `-fsycl-allow-device-image-dependencies`,
  so the import is resolved at runtime — where the AOT image fails with "No
  device image found for external symbol logf".  Device code uses the `sycl::`
  builtins (SPIR-V instructions, no imports); `rope_scaling.hpp` maps its
  `logf/cosf/sinf` through the `STRATA_ROPE_*F` macros for this reason.
- **Sub-group collectives on `double` work under the FP64 emulation**
  (`sycl::shift_group_left`, `group_broadcast`) — the sampler's and router's
  double reductions rely on it.
- **Grouped dot products need explicit rounding for dispatch parity.** The first
  SYCL grouped port passed the double host oracle but its old/new intermediate
  buffers differed. Pinning both chunk products and their accumulation through
  `sycl::ext::intel::math::fmul_rn/fadd_rn` makes both layouts byte-identical on
  the A770. The CPU-order path retains explicit FMA and its eight-lane reduction.
- The runtime's `cudaDeviceProp::l2CacheSize` uses SYCL's reported global-memory
  cache size as the existing microbenchmark's working-set sizing estimate; it
  does not establish the Intel GPU's cache hierarchy or benchmark performance.
- Two AOT warnings are informational, not failures: "EU fusion is disabled …
  SIMD32 mode specified by intel_reqd_sub_group_size(32)" on every
  `reqd_sub_group_size(32)` kernel, and "Stack call has been detected" for
  fp64-emulated or large per-lane-array kernels.  Separately, IGC reports
  register spills at SIMD32 for the warp-per-row split GEMV kernels
  (16 accumulators + 4 code words in flight) — correct, but a known
  performance item for when benchmarking starts.

## Build

```sh
cmake -S . -B build-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_ENABLE_SYCL=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_SYCL_AOT_DEVICES=dg2-g10 -DSTRATA_BUILD_TESTS=ON
cmake --build build-sycl -j
ctest --test-dir build-sycl --output-on-failure
```

`STRATA_SYCL_AOT_DEVICES` takes the ocloc names of the cards you own
(`dg2-g10` = A770, `bmg-g21` = B580); a `spir64` JIT image is always built as the
fallback.  The backends are mutually exclusive at configure time.

`strata_sycl_kernels` and `strata_sycl_runtime` are shared libraries, and device code
is split per kernel with `STRATA_SYCL_LINK_JOBS` (default 8) parallel ocloc jobs. While the
kernels were a static library, every test executable re-ran the AOT compile of every kernel:
after native MMVQ, one test link took 41 minutes with ocloc on one core and produced a
219 MB binary. Now the AOT compile happens once, when the kernel library links (the full
tree built in 12.6 minutes on this 12-core host, the test executables are about 400 KB),
and `CMAKE_LINK_DEPENDS_NO_SHARED` keeps a runtime rebuild (14 seconds) from relinking it.
The runtime must stay shared too: one copy of its device, stream and error state serves
the kernel library and every executable.

## Rules the backend enforces

- A GPU without a 32-wide sub-group is excluded at enumeration: the kernels assume
  32-lane warps, so the backend refuses to start rather than run wrong.
- `cudaDeviceGetAttribute` reports a fixed compute-capability **8.0 sentinel**:
  high enough for every portable `cc >= 75/80` host branch, below 9.0 so the sm_90
  thread-block-cluster paths are never selected.
- Graph capture maps to `sycl_ext_oneapi_graph`; anything the installed DPC++ cannot
  record (host tasks, already-recording queues) fails with
  `cudaErrorStreamCaptureUnsupported` so the engine's non-captured fallbacks run.
  The three graph rules of `include/strata/core/graph.hpp` hold unchanged.
- `cudaHostRegister` fails loudly: SYCL has no host-memory registration, so the
  pinned arena must come from `sycl::malloc_host` (the future `pinned.cpp` port),
  never from a silent no-op.
- `cudaMemcpyPeerAsync` (cross-card copies) is refused: out of scope for v1.
- **Host memory the host updates is read uncached.** On the A770 a running kernel never saw a host store to host
  USM through a plain, volatile, device- or system-scope atomic load, or after a system-scope acquire fence
  (2,000,000 polls each); the first flag wait spun until the driver reset the device. Loads with the
  sycl_ext_intel_cache_controls "uncached in L1 and L3" hint see the store, and a spin loop also needs a fence so
  the compiler keeps the load inside it (`src/kernels/sycl/mapped_host.hpp`). The doorbell, the flag waits and every
  copy "from mapped" use them; weights the host wrote before launch are read normally.
- **Device stores the host polls are written uncached.** A plain or volatile store to host USM, even followed
  by a system-scope fence, reached the host only when the kernel ended (482 ms into a 482 ms kernel); the uncached
  write hint reached it at the launch latency. Inside a captured window the next kernel may be a spin wait on the
  host's answer, so `doorbell_publish` and `mtp_select` write their payload and sequence number uncached
  (`sycl_mapped::store`). Publishing 15,360 floats costs about 50 µs, bound by the x4 link.
- **Every verify window size is captured up front.** A window graph first captured after many replays of other
  sizes never started on Level Zero (the host waited on its first doorbell; draining the queue first did not
  help). `src/core/verify.cpp` captures sizes 1..max_t at startup on SYCL; the cause is not known.
- Streams created without `cudaStreamNonBlocking` are blocking, as in CUDA: a synchronous
  `cudaMemcpy`/`cudaMemset` first waits for their submitted work (streams being captured
  are skipped). Without this, `sycl_qsa_parity`'s sequential indexer appends raced their
  input upload and failed 5 of 15 runs once launches got faster; with it, 0 of 20.
  Kernels and async copies on the null stream are not yet ordered against blocking streams.
