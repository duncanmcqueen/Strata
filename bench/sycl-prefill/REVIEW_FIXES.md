# Prefill review fixes — 2026-10-07

## Changes

- Shared cache-loan calculation sizes the selected head/tail range from actual offsets, rather than sizing the tail and borrowing the head. Insufficient capacity produces an explicitly unfit count for the existing chunk fallback rules.
- Persistent-server startup, retry, request relayout, short-request reservation and residency marking now use that same range policy. Existing restoration continues to refill the recorded (expert, slot) pairs. Resident pinning uses the same chosen range as execution.
- GEMM statistics count C's actual copied rows and charge an input copy only for nonzero beta. Counter updates and clocks are disabled when statistics are off (including an explicit value of zero). Per-expert host clocks are also disabled outside profiling.
- Floating-point matrix checks explicitly reject nonfinite outputs; CPU-only NaN/Inf negative controls run before GPU initialization.
- Historical reports distinguish repeated medians, single observations and earlier implementation variants. Invalid traffic and inferred copy-duration estimates are withdrawn. No new performance claim is made.

## Validation completed

Built successfully with the installed Intel toolchain and BMG AOT target:

```sh
source tools/sycl/env.sh
cmake --build build-sycl --target strata sycl_matrix_probe prefill_cache_loan_test sycl_file_expert_source_test -j 2
ctest --test-dir build-sycl -R '^(prefill_cache_loan_test|sycl_matrix_oracle_test|sycl_file_expert_source_test)$' --output-on-failure
```

Three tests passed. Coverage includes variable-sized head versus tail loans, insufficient capacity,
half-open bounds, simulated repeated loan/restoration, RAM-budget head retention, and nonfinite
matrix-oracle negative controls. `git diff --check` and benchmark-runner shell syntax checks passed.
The real `sycl_matrix_probe` was attempted and skipped: this environment exposes no GPU (`/dev/dri`
is absent). Compilation does not establish GPU runtime correctness.

The new binary is `build-sycl/strata`; the installed `engine/strata` has not been replaced.

## GPU validation still required

Run on an idle B70 with a controlled resident budget; record driver/toolchain identities, effective
configuration, model/prompt hashes, selected chunk, cache and loan sizes, and peak memory. Use a
separate untimed warm-up first. Existing manifests use machine-local model paths.

```sh
source tools/sycl/env.sh
unset STRATA_PREFILL_LEND_TAIL STRATA_PREFILL_TIMING STRATA_SYCL_GEMM_STAGE_STATS
unset STRATA_SYCL_GEMM_NO_STAGE STRATA_SYCL_DEVICE_EVENT_WAITS
export STRATA_BENCH_ENGINE="$PWD/build-sycl/strata"
bash bench/sycl-prefill/run_suite.sh bench/sycl-prefill/review.manifest bench/sycl-prefill/results/review.jsonl
```

The runner puts the selected binary's directory first in its library path. `review.manifest` contains
three alternating tail/head repetitions each for short, 8K, 32K, non-repeated 8K/32K and 261K prompts,
with a 64-token continuation. These runs have NOT been performed in the review-fix environment.
Report medians and ranges; increase repetitions where variance obscures the 5% regression gate.
Run a separate instrumented baseline/candidate pair to obtain corrected staging traffic; instrumentation
is off for the primary timing runs.

Also run the relevant GPU parity tests, including `sycl_gemm_4g`, `sycl_prefill_native_batch`,
`sycl_expert_cache_staging`, and `sycl_matrix_probe`. Test persistent `--serve` with alternating
large/small requests, prefix reuse, cancellation followed by another request, and continuation decode.
Compare both head/tail modes against the reference, verify range-specific residency and restoration,
and exercise reduced cache capacity and nonuniform slots. CPU loan simulation is not a substitute
for these engine/GPU checks. Repeat on A770 before claiming its validation.
