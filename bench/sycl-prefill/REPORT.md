# Intel SYCL prefill — benchmark artifacts and report

Machine: guilfoyle, Arc Pro B70 (Battlemage G31, 32 GiB), Core Ultra 5 125H, 30 GiB RAM.
Toolchain pinned in `tools/sycl/versions.txt` (DPC++ 2026.0, oneMKL 2026.0). Backend AOT `bmg-g31`.
Model: Coder native pack `coder-iq1_m` (48 layers x 256 experts; gate/up IQ2_S/IQ3_XXS/IQ3_S/IQ4_XS,
down IQ4_NL/Q2_0 — re-inventoried from `native_experts.txt`, no IQ1_M expert matrices), INT8 KV,
`--resident-experts`. Source snapshot `84e47ed`; changes on branch `intel-sycl`.

Primary numbers are final prefill wall time (the `strata generate` summary) with `STRATA_PREFILL_TIMING`
**off**, alternating base/candidate; phase/timing runs are separate. Measurements were taken with the
B70 otherwise idle; a vLLM container that occupies the card later in the session is noted where relevant.

## 1. Runner

- `run_prefill_bench.sh <tokens-file> <chunk> [extra args]` — one engine run; env passthrough.
- `run_suite.sh <manifest> <out.jsonl>` — serial runs, appends machine-readable JSON per run.
- `parse_run.py <log>` — one JSON object: inner/final prefill, TTFT, decode, phase table, metadata.
- `launch_detached.sh <log> <cmd>` — detached (`setsid`), `oom_score_adj=1000` (the kernel OOM killer
  targets the engine, not the agent; the session cgroup had logged 12 OOM kills).
- `make_prompt.py <out.ids> <n>` — non-repeated prompt from real repository text via the model tokenizer.

Artifacts: `artifacts/` holds the original `/tmp` prompt/log/reproduction files with `IDENTITIES.sha256`,
plus `prompt_nr8k.ids`/`prompt_nr32k.ids` (non-repeated, 16.1%/12.0% unique). Results JSONL:
`baseline`, `perf`, `perf2`, `perf3`, `perf4`, `nr`, `noborrow*`, `diag`, `correctness`.

## 2. P0 profile (8K, 8,225 tokens, chunk 8192, STRATA_PREFILL_TIMING=1)

GPU timeline 20,290 ms: gemm down 3,919 (19.3%), hc read 3,708 (18.3%), gdn 2,394, gemm gate/up 2,317,
qsa attn 1,520, dequant 1,440, qsa proj 1,364. Expert products (dequant+gu+down) 7,676 ms (37.8%).
Host instrumentation showed the per-expert host time was only 0.046 ms — so the phases are GPU-stream
time, not host dispatch. `STRATA_SYCL_GEMM_STAGE_STATS=1` then showed why the "GPU-stream" time was high.

## 3. Staging evidence and corrected accounting

`src/sycl_runtime/blas.cpp` stages operands ending beyond 4 GiB into their allocation.
Borrowing from the cache tail therefore adds operand copies. Historical negative-control
runs with staging disabled were faster but produced known wrong results; that control
must not be used in production.

The original instrumentation overcounted output copies: it charged both directions even
when beta was zero, and charged `ldc` rows rather than the actual `m` rows copied. The old
405,758 MiB / “393 GiB”, inferred ~29 GB/s, and inferred 13.5 seconds of copying are withdrawn.
The counter now counts the actual copy widths and directions and executes only when enabled.
A fresh instrumented GPU run is required to establish the corrected traffic and attribution.

## 4. Placement and review fixes

SYCL uses a head loan by default; `STRATA_PREFILL_LEND_TAIL=1` selects the reference tail policy.
Generation, resident pinning and persistent-server planning/relayout now use the same policy.
Slot counts are calculated from the selected range's actual byte offsets, including native
packs with nonuniform slots. Request-sized loans mark only slots inside that range and refill
the same slots afterward. CUDA/HIP retain tail loans.

Low placement reduces staging; it does not guarantee that all buffers end below 4 GiB,
especially for large chunks. The existing BLAS protection remains active whenever needed.
The head-resident complement follows the planned loan; RAM budget limits still apply.

## 5. Correctness evidence and limits

Saved `R_tail.bin` and `R_head.bin` compare byte-for-byte for the historical 8K run.
`STRATA_PREFILL_DUMP_R` samples the final residual at every 64th position; this is useful
partial-state evidence, not proof of all state or all workloads. The saved hash manifest
also lists older variants whose binary dumps are not all present.

The earlier implementation reported 59/61 CTest passes on B70, with failures attributed to
CPU AVX-512 and `mlock` limits. Those are historical results, not validation of the review fixes.
The new matrix oracle explicitly rejects NaN/Inf and has CPU-executable negative controls.
The probe still places its kernels in the test translation unit; `--linked` is not evidence
of a full in-library XMX implementation.

The review-fix environment exposes no `/dev/dri`. GPU correctness, persistent-server
repeated-request/cancellation checks and timing reruns remain pending. CPU loan tests cover
nonuniform head/tail sizing, insufficient capacity, half-open bounds, and simulated repeated
loan/restoration. See `REVIEW_FIXES.md` for current build/test results and the rerun checklist.

## 6. Historical performance: distinguish repetitions and variants

These measurements predate the review fixes. They do not establish performance of the
corrected sized-loan/server code.

| Workload | Tail reference ms | Head-pinned candidate ms | Candidate repetitions | Status |
| --- | ---: | ---: | ---: | --- |
| 8K repeated | 20,998.5 | 10,258.9 | 3 | Matched historical medians, 2.05x |
| 32K repeated | 75,929.0 | 34,841.9 | 3 | Matched historical medians, 2.18x |
| 261K repeated | 681,892.6 | 347,153.7 | 1 | Candidate is a single observation, not a median |
| 8K non-repeated | 20,706.7 | 10,295.5 | 1 | Candidate is a single observation |
| 32K non-repeated | 75,504.0 | 32,396.0 | 1 | Candidate is a single observation |

For the two three-run candidate sets, continuation decode medians were 32.69 and 33.88 tok/s,
versus 32.63 and 32.94 for the corresponding reference sets. Candidate 32K prefill ranged
from 32,774.1 to 40,285.4 ms; report this variability alongside the median.

The earlier short-prompt figure (2,274.2 ms) used a different, pre-head-pinning variant.
`perf4.jsonl` contains one final head-pinned short observation at 5,696.9 ms. Its comparability
and interference status need a fresh controlled check; it cannot establish a regression-free
median. Similarly, a single 261K continuation at 36.06 versus historical 37.55 tok/s does not
establish the 5% median regression gate.

**The full promotion gates are not yet established.** Re-run at least three alternating
reference/candidate repetitions of the final implementation for short, 8K, 32K, 261K and
non-repeated prompts, plus persistent-server requests. Increase repetitions when variance
prevents a decision. Retain separate final prefill, inner prefill, TTFT and decode metrics.

## 7. Best raw prefill (opt-in, decode trade-off)

`--no-prefill-borrow --prefill 8192` (own buffers, offset 0, no borrow): 8K 9,194 ms (894.6),
32K/8192 31,980 ms (1,028.9), 32K/32768 33,496 ms (982.3), 261K 349,012 ms (747.8) — single reps,
~2.3x. The ~5.8 GiB reserve shrinks the cache to 10,728 slots and regresses 261K continuation decode
~10% and short decode ~6%, so it is not the default; documented as the high-throughput option.

## 8. Matrix capability probe (`tests/sycl/matrix_probe.cpp`, linked into the kernel libraries)

On bmg-g31 (DPC++ 2026.0): FP16 and BF16 `joint_matrix` 16x16x16 (FP32 accumulate) and INT8
`joint_matrix` **8x16x32** (INT32 accumulate) match an independent reference exactly, including ragged
(zero-padded guard-region) shapes and multiple work-items. INT8 `16x16x16` is **rejected by the AOT
backend** ("unsupported number of rows/columns" plus undefined `OpJointMatrix...INTEL` builtins) — the
spec's B70 warning, confirmed. The probe also documents the two tiled-kernel traps: stores need the
*padded* stride and every tile load needs a sized guard region.

## 9. Remaining bottlenecks

- At 261K the dominant phase is now QSA attention (`qsa_select` 123.6 s = 35%, `qsa_attn` 56.8 s of a
  349 s no-borrow timeline), not the experts. P3 territory.
- The historical head-pinned borrow varied (8K 10.2-10.3 s; 32K 32.8-40.3 s).
  Investigate resident budget and page-cache effects in controlled reruns; attribution is not established.
- Expert products remain GPU-bound (`dequant` + `gemm_gu` + `gemm_d`); oneMKL itself is fast on B70
  (down ne=320 0.012 ms/call; dense 172 TFLOP/s), so the residual cost is the dequant kernels and
  per-expert tile/launch efficiency.

## 10. Environment notes and skipped work

- **Interference:** a vLLM XPU container (`qwen38`, `--gpu-memory-utilization 0.90`) occupied ~23 GiB of
  the card from 23:15 (and was restarted at 06:33), which collapsed the expert cache and thrashed RAM.
  Runs taken during that window (`f-*` in `results/`) are invalid and were discarded. Its exact
  `docker run` line is saved in `/home/dwmcqueen/relaunch_qwen38.sh`; a timer
  (`/home/dwmcqueen/schedule_qwen38.sh`) relaunches it at 06:50 as requested.
- Not tested: a production MTP acceptance run; A770 regression (the change is SYCL-gated and the shared
  borrow path is byte-for-byte unchanged for other backends); the matrix-probe kernel lives in the test
  TU, not a Strata kernel library, so a full in-library XMX integration was not attempted.
- `STRATA_VERIFY_PROFILE` was not used (as required).
