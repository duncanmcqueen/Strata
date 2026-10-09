# Strata Intel performance — results

Date: 2026-10-07. All measurements on Intel Arc Pro B70 (`bmg`, id `0xE223`), Level Zero device 0, unless noted.
A770 (dg2) hardware unavailable — its results are build/synthetic only and marked pending.

## Source and binary identities

| item | before (baseline) | after (this work) |
| --- | --- | --- |
| Source | `HEAD 84e47ed` + uncommitted review fixes; stash commit `703feda1d5a4…` | same tree + W0/1A/W1/W2 edits |
| `build-sycl/strata` sha256 | (not rebuilt at baseline) | `69595f0ba6877dd382e187b0ba3c6070e5ec0fa67f62702013a1d702e350fdf9` |
| `libstrata_sycl_runtime.so` | — | `4faa1030b81eba3481e6fedf0eae98d652f07f0fd1541442db0f3073105d6d22` |
| `libstrata_sycl_kernels_base.so` | — | `ccedd5fb6371bbab191c197e79d33cf8907f31fa1293679e00394e19d1aef8c0` |
| Compiler | | DPC++ 2026.0.0 (2026.0.0.20260331) |

## Correctness (device)

| test | result |
| --- | --- |
| `prefill_cache_loan_test` (CPU) | PASS |
| `sycl_matrix_oracle_test` (CPU, nonfinite controls) | PASS |
| `sycl_file_expert_source_test` (CPU) | PASS |
| `sycl_device_profile_test` (synthetic, no GPU) | PASS (13 groups) |
| `sycl_qsa_select` (sorting oracle + FP64 scores + hier + tiled bitwise) | PASS |
| `sycl_qsa_topk_active_parity` | PASS |
| `sycl_qsa_parity`, `sycl_qsa_prompt_attn_parity`, `sycl_kv_stream_parity`, `sycl_kv_decode` | PASS |
| `sycl_qsa_select_bench` (registered) | PASS |

Exact IDs: hierarchical top-k outputs match the CPU sort oracle and the reference selector exactly (not within a
tolerance). Tiled scores are **bitwise** equal to the legacy scorer. Negative control: the FP64 score oracle in
`sycl_qsa_select` rejects nonfinite values; poisoned padding past the selection width / past `n_bid` is checked
untouched.

## Performance (kernel/selector level)

W1 hierarchical vs reference top-k (262144 ctx, 8 reps, ids identical 100%):

| queries | reference ms | hierarchical ms | speedup |
| ---: | ---: | ---: | ---: |
| 1 | 0.580 | 0.078 | 7.39× |
| 4 | 0.522 | 0.099 | 5.28× |
| 8 | 0.486 | 0.132 | 3.69× |
| 16 | 0.474 | 0.188 | 2.52× |
| 64 | 0.487 | 0.534 | 0.91× |
| 256 | 1.384 | 2.273 | 0.61× |

W2 tiled vs legacy scorer (262144 ctx, 8 reps, bitwise identical):

| queries | legacy ms | tiled ms | speedup |
| ---: | ---: | ---: | ---: |
| 8 | 0.759 | 0.499 | 1.52× |
| 64 | 4.689 | 3.903 | 1.20× |
| 256 | 19.041 | 15.560 | 1.22× |

## End-to-end (B70, corrected head-loan default path, single runs)

| run | prompt tok | prefill tok/s | TTFT ms | decode tok/s | qsa score ms | qsa topk ms |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| short (legacy) | 255 | 19.71 | 224970 | 10.19 | 0 | 1 |
| 8K (legacy) | 8225 | 57.27 | 355336 | 17.33 | 111 | 194 |
| 8K (tiled, run 1) | 8225 | 52.11 | 168770 | 17.34 | 166 | 89 |
| 8K (tiled, run 2) | 8225 | 58.57 | 160530 | 17.36 | 103 | 88 |

Prefill is dominated by host expert staging, which varies by seconds between runs (the machine is shared), so the
prefill column is not an A/B verdict. Decode is stable (17.33 → 17.34/17.36), i.e. no regression from enabling the
tiled scorer by default on B70. Raw logs: `logs/`, machine-readable records: `raw/runs.jsonl`.

## Memory / ownership

- W1 workspace: `qsa_topk_workspace_bytes(nq, max_blocks)` = `(nq·tiles·256 + nq·tiles·4 + nq·8)·4` bytes,
  `tiles = ceil(max_blocks/1024)`. Prefill batch: ~16 MiB at 256×65538. Decode: ~68 KiB at 1×65538. Allocated in
  `Prefill::carve` and `QsaBuffers`, both instance-owned and alive across graph replay; never allocated inside
  capture. The kernel-header formula and the runtime selector formula are asserted equal by a test.
- The existing >4 GiB staging, repeatable small-output GEMM, safe host event waits and graph-capture workarounds
  are untouched and remain enabled.

## Rollback / dispatch controls

- `STRATA_SYCL_TOPK=reference|legacy|hierarchical|auto` — `auto` = built-in per-device winners (B70 few-query
  long context → hierarchical; otherwise legacy/reference). A forced unsupported choice fails before any launch.
- `STRATA_SYCL_SCORES=legacy|tiled|xmx|auto` — `auto` = tiled on B70, legacy on A770/unknown.
- `STRATA_SYCL_DISPATCH_QUIET=1` — silence the one-per-configuration dispatch log.
- `STRATA_SYCL_GROUPING`, `STRATA_SYCL_EXPERTS`, `STRATA_SYCL_PROMPT_ATTN` — selector vocabulary exists; the
  kernels are not implemented so the overrides are rejected cleanly (no silent wrong path).

## Numerical / quality

W1 and W2 are **exact** (scheduling/layout class 1): exact IDs and bitwise scores, no tolerance loosened. No
precision-changing candidate was enabled, so the 0.5% perplexity budget is not exercised. XMX scoring is rejected
until a compiled, application-validated kernel exists.

## Remaining bottlenecks and gaps

1. **Selection scoring** remains the largest selection cost (19 ms at 256×65538; scoring is ~93% of selection).
   Tiling gave 1.22×; further gains need matrix arithmetic or better key reuse.
2. **Prefill** is dominated by host expert staging / `gemm down` (~50–54% of the GPU timeline at 8K).
3. **A770 runtime qualification pending** (no hardware). dg2-g10 AOT build passes.
4. W3/W4/W5/W6/W7 not implemented.
5. Repeated (≥3) 32K/261K baselines and the persistent-server repetition/cancellation tests are pending.
