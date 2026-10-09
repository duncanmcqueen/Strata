# Strata Intel performance — W0–W7 progress

Date: 2026-10-07. Machine: B70 (bmg, id `0xE223`) + integrated Intel Arc (unknown arch, id `0x7D55`).
**A770 (dg2) hardware is not present.** A770 status is "synthetic + build-verified; runtime qualification pending";
it is **not** reported as tested just because B70 tests pass.

Legend: **IMPLEMENTED** · **TESTED** (unit/parity) · **MEASURED** (device runs) · **QUALIFIED/DEFAULT** (in auto
dispatch for that device) · **EXPERIMENTAL** (opt-in only) · **REJECTED** (evidence) · **PENDING** (blocked).

## Architecture-aware dispatch (section 1A)

| item | B70 | A770 (dg2) | unknown arch |
| --- | --- | --- | --- |
| Device profile (`include/strata/sycl_runtime/device_profile.hpp`, `src/sycl_runtime/device_profile.cpp`) | IMPLEMENTED, MEASURED (classifies `bmg`, id 0xE223) | IMPLEMENTED (classifies `dg2`), synthetic-tested | IMPLEMENTED, MEASURED (integrated Arc → `unknown`) |
| Pure selectors + synthetic table test `sycl_device_profile_test` (13 groups) | TESTED | TESTED | TESTED |
| Central policy: reject → override → qualified record → safe fallback | IMPLEMENTED, TESTED | TESTED | TESTED |
| No name-substring gate; ID/capability based | IMPLEMENTED | IMPLEMENTED | IMPLEMENTED |
| Dispatch follows the stream's device (not a global first GPU) | IMPLEMENTED (`profile_for_stream`) | IMPLEMENTED | IMPLEMENTED |

Ported separately for `dg2-g10` and `bmg-g31`: the **dg2-g10 AOT link of `libstrata_sycl_kernels_base.so`
succeeded** (build dir `build-sycl-dg2`) with the new W1/W2 kernels in it — the ordinary A770 build still compiles.
No B70-only matrix template is compiled into the A770 image (no matrix kernels were added; XMX families are off).

## Per-package status

| WP | What | Status (B70) | Status (A770) |
| --- | --- | --- | --- |
| W0 | score/top-k split, validated runner + aggregator, baseline doc | IMPLEMENTED, MEASURED | n/a (harness) |
| W1 | exact hierarchical top-k + workspace + boundary/replay tests | IMPLEMENTED, TESTED (152 cases + parity), MEASURED; **QUALIFIED/DEFAULT** for few-query long context | IMPLEMENTED, TESTED (synthetic/test on B70); no measured record → **safe reference fallback** |
| W1 | old selector retained | IMPLEMENTED (reference + register) | same |
| W2 | query-tiled shared-key FP32 scorer, bitwise-exact | IMPLEMENTED, TESTED (bitwise across nq 1–256, active bounds, tails), MEASURED 1.2–1.5×; **QUALIFIED/DEFAULT** | IMPLEMENTED, TESTED; **no record → legacy default** |
| W2 | XMX scoring | REJECTED for now (no compiled qualified image; precision change) | same |
| W3 | sparse prompt-attention tiling / XMX | PENDING (not implemented) | PENDING |
| W4 | stable device expert grouping | PENDING (selector present, kernel not implemented) | PENDING |
| W5a | bounded expert intermediates | PENDING | PENDING |
| W5b/W5c | tile Dm / tile-local unpack | PENDING | PENDING |
| W6 | measured chunk selection | selector API present; tuning-file loader PENDING | PENDING |
| W7 | decode kernel/host/speculation sweep | PENDING (existing variants documented) | PENDING |

## W1 details — exact hierarchical top-k

- `qsa_block_topk_hier` in `src/kernels/sycl/qsa_select.cpp`: 4×8-bit radix passes with per-tile (1024-block)
  histograms, a per-query threshold reduction, ordered two-level compaction and per-tile emit. Cell weights
  (4 for complete blocks, `n_kv-4*n_bid` for the tail), lowest-ID ties, ascending output — proven equal to the
  reference selector for every case. No grid-wide sync inside a kernel; workspace only (capture-safe).
- Workspace: `qsa_topk_workspace_bytes(nq, max_blocks)`, identical formula in the kernel header and the runtime
  selector (asserted equal in `sycl_qsa_select`). Allocated in `Prefill::carve` (bounded by `sel_batch×max_blocks`,
  not T×context) and in `QsaBuffers` (decode, one query). Both are instance-owned and captured-execution-lifetime
  safe; no allocation inside capture.
- Tests: `sycl_qsa_select` hierarchical (152 cases) — block counts around 33,792, cell counts
  135167/135168/135169, 262144 context, partial blocks, all-equal, ties spanning tiles, signed zeros,
  NaN/±inf, identity widths, poisoned padding untouched, short/long/short graph replay — all exact vs the CPU sort
  oracle. Existing `sycl_qsa_topk_active_parity` still passes.
- Measured (B70, 262144 ctx, capacity 262144, 8 reps, reference vs hierarchical, ids identical 100%):

  | queries | reference ms | hierarchical ms | speedup |
  | ---: | ---: | ---: | ---: |
  | 1 | 0.580 | 0.078 | **7.39×** |
  | 4 | 0.522 | 0.099 | **5.28×** |
  | 8 | 0.486 | 0.132 | **3.69×** |
  | 16 | 0.474 | 0.188 | **2.52×** |
  | 64 | 0.487 | 0.534 | 0.91× |
  | 256 | 1.384 | 2.273 | 0.61× |

- **Promotion decision:** enabled in `auto` on B70 only for few queries (≤32) beyond the register capacity
  (`active`/capacity > 33,792), i.e. the decode path (`layer.cpp`, nq=1). It is **rejected for prefill batches**
  (nq=256) where it loses — those keep the established selector. No record exists for dg2, so A770 keeps its
  reference fallback. Built-in record: `builtin_tuning()` in `device_profile.cpp`.

## W2 details — tiled shared-key FP32 scoring

- `block_scores_tiled_kernel` (QT=8 queries per workgroup) reuses each key row across the query tile **within one
  kernel launch** (no extra host launches) and supports the active-block bound and arbitrary nq.
- Arithmetic is byte-for-byte the reference: same lane-local float4 dot, same 16-step xor reduction, per-head ReLU
  then head sum, same tail `+1e9`. Verified bitwise across nq ∈ {1,4,8,16,256}, strides {1025, 32770}, active
  bound and captured (bound omitted) modes, varying `n_bid` inside a query tile, incomplete tails, and no writes
  past `n_bid` (poisoned guard checked).
- Measured (B70, 262144 ctx, 8 reps): 0.76→0.50 ms (1.52×) at nq=8, 4.69→3.90 (1.20×) at nq=64, 19.04→15.56
  (1.22×) at nq=256, bitwise-identical. **QUALIFIED/DEFAULT for B70** (all shapes; never slower). dg2 has no
  record → legacy.
- XMX scoring is **rejected** for now: no compiled, application-validated matrix kernel exists in this build, so
  `select_scores(... xmx)` returns a clear rejection rather than enabling a precision-changing path.

## End-to-end evidence (B70, corrected default path)

`qsa score` / `qsa topk` appear separately in the phase table (W0). 8K decode: 17.33 tok/s (before W2) vs
17.34/17.36 (after W2) — no regression. See `RESULTS.md`.

## Blockers / pending

- A770 runtime qualification (hardware absent). Synthetic selectors pass; dg2 AOT build passes.
- W3/W4/W5/W6/W7 kernels not implemented; no `auto` enablement.
- 32K/261K repeated (≥3) baselines and persistent-server validation not run this session.

## Final local validation sweep (B70, `build-sycl`)

`ctest --test-dir build-sycl` → **62/64 passed** (55 s). The two failures are environmental, unchanged by this
work:

- `expert_multi_test` — this CPU lacks AVX512-VNNI/VBMI (the engine says so itself; the scalar fallback is
  test-only). Pre-existing.
- `sycl_platform_memory_test` — `mlock` fails because `ulimit -l` is too low; it prints "raise ulimit -l".
  Pre-existing.

New/affected tests all pass: `sycl_device_profile_test`, `sycl_qsa_select` (hierarchical + tiled bitwise),
`sycl_qsa_topk_active_parity`, `sycl_qsa_parity`, `sycl_qsa_prompt_attn_parity`, `sycl_kv_stream_parity`,
`sycl_kv_decode`, `sycl_qsa_select_bench`, `prefill_cache_loan_test`, `sycl_matrix_oracle_test`,
`sycl_file_expert_source_test`.

A770/dg2-g10 AOT link of `libstrata_sycl_kernels_base.so` and `libstrata_sycl_runtime.so` succeeds
(`build-sycl-dg2`). A770 runtime execution is pending hardware.
