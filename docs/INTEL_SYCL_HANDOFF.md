# Intel Arc SYCL port: agent handoff

Updated 2026-10-05: every kernel is ported, the engine runs the Coder model end to end on the A770 (15 tok/s decode
with setup's configuration, 11 tok/s at 8.5K context), and setup supports `--backend sycl`.  Measurements and the speed history are in
[INTEL_SYCL.md](INTEL_SYCL.md#model-run-and-speed); the sections below the next-steps list are the earlier batches'
record.

## Objective and current checkpoint

Continue the Intel Arc SYCL backend port of Strata. The user has repeatedly authorized continuing the implementation and tests. This handoff records the work completed through the QSA prompt-attention port; the next implementation batch is Batch 3.

- Batch 1: complete; runtime/backend foundation and initial kernel family ports.
- Batch 2: complete; GDN, GR, QSA core and grouped 2-bit experts.
- Batch 2b: complete for correctness; QSA decode/prompt attention, selection and KV streaming, plus their INT8/Q4 storage dependencies.
- Batch 3: in progress. Done: i-quant kernels (`iq_kernels.cpp`) and native MMVQ (`native_mmvq.cpp`). Pending: PLE, remaining native kernels and verification.
- Engine integration: pending. This is a tested kernel backend, not yet a working Intel inference engine.
- Matrix acceleration and performance/model-quality validation: pending, separate from correctness parity.

## Workspace and instructions

Work from the repository root (the directory with `CMakeLists.txt`).

Read `AGENTS.md`, [INTEL_SYCL.md](INTEL_SYCL.md), and [INTEL_SYCL_PORTING_GUIDE.md](INTEL_SYCL_PORTING_GUIDE.md). Preserve the public kernel interfaces in `include/strata/kernels/`: SYCL implementations live in `src/kernels/sycl/`, while existing CUDA/HIP parity sources are linked unchanged against them. Passing the existing parity test is the port acceptance gate.

The port is committed on the `intel-sycl` branch of the fork (based on upstream `99f3dbd`; upstream `main` has since been rewritten, so a rebase is needed before any upstream PR).

- Current developer instructions prohibit spawning sub-agents unless the user or applicable repository/skill instructions explicitly authorize delegation.

## Toolchain and commands

Source the environment in **every** build/test shell:

```bash
cd strata   # the repository root
. tools/sycl/env.sh
```

The existing build is `build-sycl`. It uses oneAPI DPC++ 2026.1.1, oneMKL, Level Zero, AOT `dg2-g10` plus a `spir64` JIT fallback. Exact versions are in `tools/sycl/versions.txt`.

The test/inference card is Arc A770, `level_zero:1`. Arc B580, `level_zero:0`, drives the display; keep it out of test runs. `env.sh` sets the compiler/library paths, device selector, relaxed allocation limit, Sysman free-memory queries and FP64 emulation. Both cards lack native FP64; the build also enables AOT FP64 emulation. Kernels require subgroup size 32.

To configure from scratch if needed:

```bash
cmake -S . -B build-sycl -G Ninja -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_C_COMPILER=icx -DCMAKE_CXX_COMPILER=icpx \
  -DSTRATA_ENABLE_SYCL=ON -DSTRATA_ENABLE_CUDA=OFF -DSTRATA_ENABLE_HIP=OFF \
  -DSTRATA_SYCL_AOT_DEVICES=dg2-g10 -DSTRATA_BUILD_TESTS=ON
cmake --build build-sycl -j 4
ctest --test-dir build-sycl --output-on-failure
git diff --check
```

Use targeted builds/tests during implementation. Avoid concurrent Ninja invocations on this shared build tree. The streaming test takes about 70 seconds; provide progress updates while longer checks run.

## Completed runtime and build foundation

- Opt-in, mutually exclusive SYCL backend in CMake, oneMKL linkage, precise floating-point compilation and AOT/JIT images.
- CUDA-shaped runtime compatibility headers and implementation over SYCL USM, in-order queues, events and experimental command graphs.
- Stream-to-queue bridge: `include/strata/sycl_runtime/queue_bridge.hpp`.
- Runtime smoke checks device discovery, subgroup requirements, memory, streams/events, timing and graph capture/replay.
- QSA required default-stream support for `cudaEventRecord` and graph node inspection. Graph counts, stable node handles and supported node types now use the installed SYCL graph API. Kernel launch-parameter inspection remains unsupported.
- Added `cudaDeviceProp::l2CacheSize` from SYCL's global-memory cache-size query solely as the original benchmark's sizing estimate; this is not a measurement of Intel cache hierarchy.
- `cudaHostRegister` and cross-card peer copies remain explicitly unsupported. Future pinned-arena integration must allocate host USM rather than silently pretend registration succeeded.
- The runtime compute-capability value is an 8.0 compatibility sentinel, not a hardware claim. Do not use it as proof that CUDA-specific acceleration exists on Intel.

## Completed kernel inventory

All paths below are under `src/kernels/sycl/`.

| Family | Implementations and behavior |
| --- | --- |
| RoPE | `rope.cpp`, `native_rope.cpp`; partial NEOX and analytic native paths, scaling support |
| Dequantization and activations | `dequant_s2.cpp`, `dequant_bf16.cpp`, `quantize_act.cpp`, `elementwise.cpp`; exact quantization contracts, glue operations and conversions |
| GEMV and shared experts | `bf16_gemv.cpp`, `s2_gemv.cpp`, `s2_gemv_q8.cpp`, `s_gemv.cpp`, `shared_expert.cpp`; fast/quads S2 entry points are included in the ported family rather than separate filenames |
| Routing and sampling | `router_top10.cpp`, `sampler.cpp` |
| GDN | `gdn.cpp`; recurrence, convolution, norms and gates |
| GR | `gr.cpp`, `native_gr_norm.cpp`, `native_gr_postops.cpp`, `fused_gr.cpp`; BF16/FP32/native modes and single/multi-token fused reads |
| QSA core | `qsa.cpp`, `native_qsa_indexer.cpp`; cache append/gather, reference and native indexers, exact top-k, attention and output gates |
| Grouped S2 experts | `s2_expert_grouped.cpp`; old/new per-hit and grouped paths, device counts, multi-token weight sharing, CPU-order path, grouping and hit selection |
| KV storage | `kv_q8.cpp`, `kv_q4.cpp`; append/gather, host/staging writes, Q4 Walsh-Hadamard transform |
| Decode attention | `qsa_decode_attn.cpp`; single/batched chunked attention, FP16/INT8/Q4/hybrid K8V4, missing-page masking |
| KV streaming | `kv_stream.cpp`; clock eviction/resolution, slot copies, counters/reset, ring mapping/restore and prompt staging |
| QSA selection | `qsa_select.cpp`; scalar and multi-query shared-weight scoring, reference and register radix selection, active-bound dispatch |
| Prompt attention | `qsa_prompt_attn.cpp`; scratch-free FP32 online softmax, one work-group per query/KV head, 32-cell tiles, all four KV formats |
| I-quants | `iq_kernels.cpp`; IQ1_M/IQ2_*/IQ3_*/IQ4_*/Q2_0 and UD-Q4_K_XL (Q4_K/Q5_K/Q5_0/Q5_1/Q8_0) dots, dequantizers, embedding rows, q8_1 quantizer, per-column and decode-once MMVQ, grouped native experts with group stride and fused SwiGLU + q8_1; `dequant_bf16.cpp` now routes i-quants here |
| Native MMVQ | `native_mmvq.cpp`; ten native formats, ncols 1-8, exact and upstream layouts, `native_quantize_q8_1`; the shared-expert stubs are removed |

The native BF16 MMVF projection is already implemented inside `shared_expert.cpp`. Audit symbols before treating every missing CUDA filename as an unimplemented interface.

## Implementation details to preserve

### GR and QSA core

GR parity covers the real 2560/4/320 geometry, all three activation modes, final mixers without injection, graphs and eight-token weight sharing. The multi-token fused read shares a 1280-float tile per token, 40 KiB local memory at eight tokens. Outputs match separate fused calls byte-for-byte.

Only the plain fused GR variant is implemented. CUDA split/staged `cp.async` and `STRATA_GR_V3` variants are not selected. Optional profiling timestamp stamps are refused until the verification helper exists.

QSA core preserves byte-exact cache copies, deterministic tie ordering, double reference pooling and F16-rounded native pooling. Native scalar and batch indexer states match byte-for-byte for no scaling, linear scaling and YaRN. Replay positions/counts come from device buffers.

### Grouped experts

The unchanged grouped parity fixture covers selectable old/new dispatches, outputs and intermediate scratch, FP16/FP32 scales, alignment fallbacks and mapped-host weights. The first implementation passed the double oracle but old/new intermediates differed because compiler contraction differed between layouts. Explicit `fmul_rn` for both chunk scale products and `fadd_rn` for accumulation fixed it. Preserve this rounding.

`tests/sycl/grouped_cpu_order.cpp` reuses the original fixture without editing it and adds an independent transcription of the CPU reduction: eight FMA lanes, paired low/high products and horizontal sums, plus correction. Gate/up trace words match exactly. Down projections use the original double oracle. Capture/direct results are identical across changed inputs. Routing covers single-warp and 128-entry selection, invalid IDs, shuffled mappings and empty replay.

### Streaming and decode attention

The unchanged streaming test executes 40,000 appends and 306 query batches per format. Streamed/resident outputs and ring-restored attention are byte-identical for FP16, INT8 and Q4; map inversion and counters are checked.

The resolver retains its 1024-thread work-group and the minimum-slot guard. Work-group barriers fence both local and global memory because later clock-sweep steps consume global slot stamps. Do not replace them with local-only fences.

`kv_helpers.hpp` contains private grid/vector/subgroup/atomic helpers reused by KV and selection ports. CUDA x-fast indexing is flattened into SYCL 1D work-groups; subgroup size 32 is explicit.

### Selection

The register radix kernel uses 1024 threads, 33 keys per thread, per-warp histograms and two scans. Larger contexts use the reference radix kernel. A valid explicit active-block bound may choose the register kernel without changing the score-row stride; graph callers omit the bound so context can grow safely. Invalid/nonpositive bounds retain the capacity rule.

`STRATA_TOPK_OLD` and `STRATA_SCORES_MULTI` retain debug dispatch controls. The supplemental sorting/scoring test also passed with `STRATA_TOPK_OLD=1 STRATA_SCORES_MULTI=0`.

CUDA TF32/AMD WMMA block scoring and CUDA thread-block-cluster selection are **not implemented** on SYCL. Capability probes launch nothing and report unavailable; scalar scores and one-work-group top-k remain available. Do not count these accelerated variants as completed.

### Prompt attention: most recent change

This is a separate SYCL implementation, not a wrapper around decode attention. It keeps online softmax state in local memory and numerator accumulators in registers, needs no external scratch, and uses full FP32 subgroup arithmetic instead of CUDA MMA instructions.

The first version accumulated directly into the running numerator for every cell and narrowly failed the original FP16 accuracy bound at 32K/256 queries: maximum FP64 error 5.83e-6 versus decode 1.41e-6. Forming a separate numerator sum for each 32-cell tile and merging once per tile reduced that error to 1.85e-6 and passed. Preserve tile-local accumulation.

A subgroup barrier ensures all lanes finish reading previous softmax state before lane 0 updates it. Work-group barriers separate row loads, scores, probabilities and accumulation. Empty/all-missing selections produce zero; an entirely missing first tile can recover when later tiles contain resident cells.

`STRATA_PROMPT_ATTN_Q4=0` returns false so callers can fall back to decode attention. Unsupported geometry, null required buffers and incomplete pools return false without writing output; zero queries return true without launching. CUDA V1/async pipeline controls do not select SYCL variants. Intel matrix acceleration remains pending.

## Numerical/toolchain lessons

- Batch 3 i-quants: every float product/sum in the dots, the lane reductions and the q8_1 block sum goes through `fmul_rn`/`fadd_rn`/`fsub_rn`; the per-column, decode-once and grouped kernels are compared bitwise. `byte_perm` uses CUDA's 3-bit selector (Q2_0 depends on it); codebook tables are read as their element type, not cast to wider words (no alignment guarantee for host `static const` tables in SYCL).
- Native MMVQ's single-column call is its multi-column kernel with NCOLS = 1 (the CUDA file documents the equivalence). This keeps exact multi-column calls bitwise equal by construction; do not reintroduce separate ncols = 1 kernels without rerunning `sycl_iq_parity` and `sycl_mmvq_multi_parity`.
- Build cost: `strata_sycl_kernels` and `strata_sycl_runtime` are SHARED, device code is split per kernel and ocloc runs `STRATA_SYCL_LINK_JOBS` (8) jobs per link. As a static library, each test link re-ran AOT for every kernel (41 minutes, 219 MB per executable after native MMVQ). Now the library link is about 13 minutes, done once; tests link in seconds. `CMAKE_LINK_DEPENDS_NO_SHARED` stops runtime edits from relinking the kernel library. Keep the runtime shared, or its state is duplicated.
- Runtime shim: blocking streams (created without `cudaStreamNonBlocking`) are now ordered before synchronous `cudaMemcpy`/`cudaMemset`, as CUDA's legacy default stream requires. Its absence made `sycl_qsa_parity`'s batch-vs-sequential indexer check fail 5 of 15 runs (the sequential loop's next upload overwrote a pending kernel's input); 0 of 20 after the fix. Kernels and async copies on the null stream are still not ordered against blocking streams.
- Long builds in this environment: background shells die with the session; run multi-minute builds as `systemd-run --user --collect --unit=<name> bash -c '...'` and poll a log.

- Build with `-ffp-model=precise`; the compiler's fast mode changed host oracle arithmetic.
- Use `sycl::` math builtins in device code. Imported `logf`/`powf` symbols caused AOT runtime resolution failures.
- IGC variable-divisor FP32 division is not correctly rounded. Use Intel `fdiv_rn` when parity requires IEEE rounding.
- Preserve explicit double arithmetic where exact reference pooling or scale calculation requires it; do not accidentally call float intrinsics on doubles.
- Use the existing `f16_bits.hpp` conversion helpers where bit contracts matter.
- Preserve source reduction order and explicit FMA/rounding when tests compare bits. Ordinary tolerance passing does not establish dispatch/intermediate parity.
- AOT warnings about SIMD32 EU fusion, stack calls and register spills are known; successful parity is correctness evidence, not a performance claim.

## Latest validation checkpoint

Full build succeeded (12.6 minutes for the tree after the shared-library change). Full CTest on the A770: **43/44 pass**, 109 seconds.

- All **25** SYCL kernel parity tests pass with original test sources unchanged: the 21 earlier ones plus `sycl_iq_parity` (gguf-py fixtures from `tools/iq_fixture.py`, generated into the build directory by the `sycl_iq_parity_fixtures` test), `sycl_iq_multi_parity`, `sycl_native_grouped_parity` and `sycl_mmvq_multi_parity`.
- `sycl_iq_parity`: dequant relative error 0.00e+00 for all ten formats; MMVQ 4.6e-3 to 5.7e-3 relative (bound 2e-2); multi-column bitwise.
- The unchanged QSA selection harness passes.
- Three supplementary tests pass: grouped CPU-order/routing, KV/decode/prompt graphs and independent QSA selection oracles.
- Runtime smoke passes.
- `git diff --check` passes.
- The sole failure is the pre-existing `expert_multi_test`: this CPU lacks AVX512F/BW/VL/VNNI/VBMI. Do not weaken, skip or repair unrelated tests to conceal this hardware limitation.

Prompt CTest uses the original source with arguments `32768 2048 1`: 2048 queries at 32K, plus that source's short and transition cases. Against its FP64 oracle, the new prompt maximum errors were 1.98e-6 for INT8, 2.05e-6 for FP16 and 5.4e-6 for Q4 at 32K. All original accuracy gates passed. Timings printed by the fixture are not an end-to-end performance result.

The expanded `tests/sycl/kv_decode.cpp` compares both prompt and decode to an independent FP64 oracle for four formats, checks direct/captured results, changing selections/page tables, all-missing and initially missing tiles, and refusal without output writes. Latest prompt errors were below 1.5e-7 in that smaller fixture.

`tests/sycl/qsa_select.cpp` verifies IDs against stable CPU sorting for 57 replay cases including NaNs, infinities, signed zeros, ties, partial blocks and up to 262144 cells. Scoring paths match byte-for-byte; maximum error against FP64 is 2.87e-6 in its fixture. The original active-bound parity test passes all 16 cases.

Useful existing targets:

```bash
cmake --build build-sycl --target sycl_qsa_prompt_attn_parity sycl_kv_decode -j 4
ctest --test-dir build-sycl -R 'sycl_(qsa_prompt_attn_parity|kv_decode)$' --output-on-failure
cmake --build build-sycl --target sycl_qsa_select sycl_qsa_topk_active_parity sycl_qsa_select_bench -j 4
ctest --test-dir build-sycl -R 'sycl_qsa_(select|topk_active_parity|select_bench)$' --output-on-failure
```

Latest logs in `/tmp` (useful only while this machine/session retains them):

- `/tmp/strata-prompt-full-build.log`
- `/tmp/strata-prompt-full-test.log`
- `/tmp/strata-select-full-test.log`
- `/tmp/strata-kv-full-test.log`

CTest detail is in `build-sycl/Testing/Temporary/LastTest.log`; a later test run overwrites it. Do not depend on temporary generator scripts as project infrastructure. The checked-in/workspace C++ sources are authoritative.

## Next agent: recommended work order

Batches 1-3 and engine integration are done.  Model: `~/models/strata-coder/` (the two IQ1_M shards, `pack/`, and
`mtp/rt` built by setup's MTP steps).  Run command: see INTEL_SYCL.md.  Use `level_zero:1` (the A770); the B580 runs the
display.  Long runs go through `systemd-run --user` units (background shells die with the session).

1. Where a round goes now (setup's configuration with its draft vocabulary, 15.8-16.3 tok/s, ~3.4 tokens per round, ~215 ms): ~115 ms in the
   CPU expert pool and about as long on the GPU side of the split window, overlapped; ~60-70 ms of it the host waits on
   the GPU.  Both sides have to shrink now.  GPU: the VRAM-hit experts' 18-byte IQ4_NL/Q2_0 blocks still force 2-byte
   loads (the Q6_K reorder's treatment at expert-cache upload is the next step), the fused GR reads (~50 ms per
   window), and ~0.65 ms per layer of link-bound host transfers.  CPU: one-token experts (80% of them) run ggml-cpu's
   single-token i-quant dots at 17.8-22.7 GB/s over 6 threads; the multi-token kernels (`iq_avx2.cpp`) are faster
   only from two tokens.
2. Build benchmarks with the libraries' ocloc options (`tools/sycl` flags, `-ze-opt-greater-than-4GB-buffer-required`):
   without them uncached hints are silently dropped.  Give them inputs with full mantissas (normal random values):
   with a few mantissa bits every summation order gives the same sum, and an order that varies cannot show.
3. oneMKL's partial-tile reads past 4 GiB of an allocation (INTEL_SYCL.md, "Long context") are worked around in the
   BLAS shim; report it upstream, and drop the staging once a fixed oneMKL passes `sycl_gemm_4g` with
   `STRATA_SYCL_GEMM_NO_STAGE=1`.  Likewise oneMKL's small-output GEMMs are not repeatable (INTEL_SYCL.md,
   "Repeatable replies"); the shim's own kernel computes them.
4. The verify-window late-capture hang (`capture_all` in `verify.cpp`): find the cause and remove the workaround.
5. `ple_parity` still needs model fixtures (`bench/micro/ple_in.bin`/`ple_out.bin`); they can now be captured from
   the Coder run.
6. Known host-only test failures on this machine: `sycl_platform_memory_test` (`ulimit -l` is 8 MB) and
   `expert_multi_test` (needs AVX-512; the Ryzen 5600 has AVX2).
