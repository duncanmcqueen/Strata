# W0 — baseline and trustworthy benchmark infrastructure

Date: 2026-10-07. Machine: Intel Arc Pro B70 (Battlemage, PCI id `0xE223`, Level Zero device 0) plus an
**integrated** Intel Arc Graphics (Xe-LPG, id `0x7D55`, device 1). **No Arc A770/`dg2` card is present on this
machine**; A770 qualification is therefore synthetic-only plus a build check (below). The integrated GPU exercises
the "unknown architecture" fallback path on real hardware (`ONEAPI_DEVICE_SELECTOR=level_zero:1`).

## Baseline is the corrected working tree, not HEAD

`HEAD` is `84e47ed`, but the reviewed fixes are uncommitted. The checkpoint is recorded in
`checkpoint/`:

- `checkpoint/stash-commit.txt` = `703feda1d5a42753f29ad7fd8a0345cc6fc1e470` (all tracked changes, via `git stash create`)
- `checkpoint/untracked.sha256` — hashes of `include/strata/prefill/cache_loan.hpp`,
  `tests/core/prefill_cache_loan_test.cpp`, `tests/sycl/matrix_probe.cpp`
- `checkpoint/tracked.diff`, `checkpoint/git-status.txt`, `checkpoint/HEAD.txt`
- `checkpoint/bench-prefill.sha256` — the archived `bench/sycl-prefill/` artifacts

The shared head/tail cache-loan helper, persistent-server fix, corrected GEMM copy byte accounting and the
nonfinite matrix-oracle checks are all preserved. Nothing was reverted.

## Review-fix CPU tests (the required starting point)

```
ctest --test-dir build-sycl -R '^(prefill_cache_loan_test|sycl_matrix_oracle_test|sycl_file_expert_source_test)$'
# 3/3 tests passed (0.29 s)
```

`sycl_matrix_oracle_test` runs CPU-only NaN/Inf negative controls before any device initialisation.

## Toolchain and binary identities

- Compiler: `Intel(R) oneAPI DPC++/C++ Compiler 2026.0.0 (2026.0.0.20260331)`
- AOT device for `build-sycl`: `bmg-g31` (B70); targets `spir64_gen,spir64`
- `build-sycl/strata`                sha256 `69595f0b…` (the running binary; **not** the installed `engine/strata`)
- `build-sycl/libstrata_sycl_runtime.so` sha256 `4faa1030…`
- `build-sycl/libstrata_sycl_kernels_base.so` sha256 `ccedd5fb…`
- Full list: `raw/binary.sha256`

## W0 code changes

1. **Score/top-k split.** `src/prefill/prefill.cpp` `PfTimer` gained `kPfQsaScore` and `kPfTopk`; the selection
   loop is now two passes (all scores, then all top-k) so each is charged separately. The historical combined
   `kPfQsaSel` phase name is retained in the table but is 0 on the new path — never sum it with its components.
2. **Parser compatibility.** `bench/sycl-prefill/parse_run.py` now emits `qsa_score` and `qsa_topk` keys while
   still reading the historical `qsa_select` label (verified on `results/baseline-8k-auto.log`).
3. **Runner validation.** `bench/sycl-prefill/run_suite.sh` now rejects a run with nonzero exit, a missing final
   timing field, or a token-count mismatch (`valid=false`, `invalid_reason`), writes a uniquely-named log per run
   (never overwrites), keeps failed artifacts, and calls the new aggregator.
4. **Aggregation.** `bench/sycl-prefill/summarize.py` reports run count / median / min / max per label, lists
   rejected runs separately, and prints the configuration identity.

## Measured baseline (B70, corrected default path)

Single runs on an otherwise-busy machine; prefill is dominated by host expert staging (which varies by several
seconds run to run), so these are a starting point, not a promotion verdict. Commands and raw logs are in
`logs/` and `raw/runs.jsonl`.

| run | prompt | prefill tok/s | decode tok/s | qsa score ms | qsa topk ms | notes |
| --- | --- | ---: | ---: | ---: | ---: | --- |
| `smoke-short` | 255 | 19.71 | 10.19 | 0 | 1 | split phases visible |
| `baseline-8k` | 8225 | 57.27 | 17.33 | 111 | 194 | legacy scorer |
| `tiled-8k-1` | 8225 | 52.11 | 17.34 | 166 | 89 | W2 tiled scorer (default on B70) |
| `tiled-8k-2` | 8225 | 58.57 | 17.36 | 103 | 88 | W2 tiled scorer |

Decode is stable across the scorer change (17.33 vs 17.34/17.36); prefill variation is host-staging-bound.

## Warm-up and repetition policy

The engine performs its own untimed warm-up (expert placement / residency); suite runs are measured separately.
Three alternating reference/candidate repetitions are the promotion target; the runs above are single
observations and are labelled as such. Longer-context (32K/261K) baselines and the alternating repetitions are
listed as pending in `PROGRESS.md`.

## Server validation

Not performed in this session (no protocol harness was run). `src/program/generate.cpp` remains the source of
the `--serve` stdin protocol; repeated requests, cancellation and reduced-cache cases are pending.
