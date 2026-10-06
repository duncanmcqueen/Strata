# Intel Arc SYCL backend review

Reviewed on 2026-10-06: `99f3dbd..intel-sycl`, with branch head
`30accf1abdc1ddd68debc785800b8b97b5a0082a`.

This review is based on code inspection. No implementation files were changed,
builds run, or GPU work launched. This document was added afterward at the user's
request. No commit or push was made.

## Findings

### 1. Shared GEMM staging scratch can race across queues

- **Location:** `src/sycl_runtime/blas.cpp:262`
- **Severity:** race
- **Problem:** Every staged GEMM uses one process-wide scratch allocation.
  In-order execution protects reuse within one queue, but there is no dependency
  between queues. Concurrent first calls also race on the pointer's initialization.
- **Failure scenario:** Queue A copies its operands into scratch and starts GEMM;
  queue B overwrites the same scratch before A finishes reading it. Results become
  timing-dependent. Sequential host calls on different streams suffice because
  submission is asynchronous.
- **Suggested fix:** Own scratch per stream/device, or serialize its reuse with
  completion dependencies. Captured graphs also need exclusive scratch ownership
  during replay.

### 2. Doorbell checksum equality does not establish payload completion

- **Location:** `src/kernels/sycl/elementwise.cpp:277`
- **Severity:** race
- **Problem:** The wrapping sum is insensitive to permutations and cancelling
  changes. The code explicitly accommodates the tag overtaking payload stores,
  but accepts any visible payload with the expected sum.
- **Failure scenario:** Two expert IDs change from `[3, 7]` to `[7, 3]`. Their
  checksum contribution is unchanged. If both stores lag behind the new tag, the
  host accepts the old IDs while weights or activations may already be new.
  Activation changes with cancelling bit-pattern differences have the same problem.
- **Suggested fix:** Establish a verified visibility/completion protocol. If
  overlapping execution requires polling, associate generation information with
  individual payload chunks and validate a host snapshot before handing it to the
  pool. A stronger checksum reduces collision probability but does not provide
  synchronization.

### 3. Synchronization does not surface asynchronous execution errors

- **Location:** `src/sycl_runtime/runtime.cpp:650`
- **Severity:** bug
- **Problem:** Stream, device and event synchronization use `wait()`, as do
  synchronous copies. The installed SYCL headers distinguish this from
  `wait_and_throw()`, which delivers asynchronous errors to the handler. Even when
  the handler has populated `g_async_error`, synchronization returns success
  without checking it.
- **Failure scenario:** A kernel fails asynchronously;
  `cudaStreamSynchronize()` returns success, and the engine consumes invalid
  output. The error can remain undelivered until queue destruction.
- **Suggested fix:** Explicitly deliver asynchronous errors at synchronization
  boundaries and return the translated error. Apply this consistently to
  stream/device/event synchronization and synchronous memory operations.

### 4. Legacy default-stream ordering remains incomplete

- **Locations:** `src/sycl_runtime/memory.cpp:157`,
  `src/sycl_runtime/runtime.cpp:376`
- **Severity:** bug
- **Problem:** Async copies and kernel submissions on the null stream do not
  wait for blocking streams. Conversely, submissions to blocking streams do not
  depend on earlier null-stream work. Only synchronous copy/memset calls implement
  part of the ordering.
- **Failure scenario:** Enqueue an upload on a stream created with
  `cudaStreamCreate()`, then launch a consuming kernel on the null stream. CUDA
  orders these operations; this shim can run the consumer first.
- **Suggested fix:** Implement dependencies in both directions at submission
  boundaries, including copies, kernels, event records and graph launches. Add a
  regression test for both ordering directions. This limitation is acknowledged
  in the handoff, but remains a runtime contract violation.

### 5. Host-blocking stream waits can deadlock host/GPU handoffs

- **Location:** `src/sycl_runtime/runtime.cpp:682`
- **Severity:** bug
- **Problem:** The assertion that already-submitted event work cannot deadlock
  is incorrect: that work may itself wait for a future host action.
- **Failure scenario:** Stream A contains `wait_flag_ge(flag, 1)` followed by
  event E. The host calls `cudaStreamWaitEvent(B, E)` and intends to raise `flag`
  afterward. CUDA returns after enqueueing the dependency; this implementation
  blocks before the host can release A.
- **Suggested fix:** Preserve asynchronous dependency semantics. If device
  barriers are unsuitable on this driver, use a deferred host-side submission
  mechanism rather than blocking the caller. Document any deliberately restricted
  contract.

### 6. GEMM chunking uses the wrong B offset for transposed operands

- **Location:** `src/sycl_runtime/blas.cpp:278`
- **Severity:** bug
- **Problem:** `Bp` always advances by `c0 * ldb * element_size`. For transposed
  B, successive output columns correspond to successive rows of stored B, so the
  offset must be `c0 * element_size`. The guard only rejects staging when B itself
  is far; it allows transposed B when C triggers staging.
- **Failure scenario:** FP32 GEMM with `transb=T`, `m=1024`, `n=32768`, `k=16`,
  `ldb=32768`, ordinary A/B allocations and C past 4 GiB. C requires multiple
  scratch pieces. The second piece advances B hundreds of megabytes instead of
  into its next rows, causing invalid reads.
- **Suggested fix:** Use transpose-aware offsets and packing, or reject
  unsupported staged transpose combinations before submission.

### 7. An unavailable GEMM workaround silently falls back to the affected operation

- **Location:** `src/sycl_runtime/blas.cpp:266`
- **Severity:** bug
- **Problem:** When far A occupies at least 96 MiB, a column cannot fit, or far
  B is transposed, the shim invokes oneMKL directly and reports success. These are
  operands that allocation bookkeeping identified as requiring protection.
- **Failure scenario:** A BF16 GEMM with a partial-tile B past 4 GiB and a far A
  larger than the scratch allocation bypasses staging. On the documented affected
  oneMKL version, the original corruption remains possible.
- **Suggested fix:** Allocate sufficient scratch, implement another safe
  fallback, or return an explicit unsupported/allocation error. Do not silently
  claim success for an unprotected case.

### 8. The interleaved-C regression test bypasses staging

- **Location:** `tests/sycl/gemm_4g.cpp:75`
- **Severity:** bug
- **Problem:** Its half GEMMs have `48 × 3` outputs; the whole GEMM has
  `96 × 3`. Both satisfy the small-output threshold and return before reaching
  staging.
- **Failure scenario:** Restore the old whole-column C copy-back bug. This test
  still passes because neither comparison executes that copy-back. Consequently,
  `docs/INTEL_SYCL.md:195` overstates the current test's protection.
- **Suggested fix:** Add an interleaved-C case exceeding 65,536 output elements,
  with sentinel values in untouched rows and nonzero beta. Keep the existing case
  as a small-kernel test.

### 9. The handoff checkpoint contradicts the current status

- **Location:** `docs/INTEL_SYCL_HANDOFF.md:10`
- **Severity:** style
- **Problem:** The opening reports working inference, but the checkpoint says
  Batch 3 and engine integration are pending and calls this “not yet a working
  Intel inference engine.”
- **Failure scenario:** Someone follows the checkpoint and repeats completed
  integration work or misreports backend readiness.
- **Suggested fix:** Update the checkpoint or label it explicitly as a
  historical snapshot.

## Checks with no problem found

- Small-GEMM tail bounds, tiled alignment/evenness guards, odd-dimension
  fallback, local-memory initialization and fixed reduction order.
- Staged C's current `m`-row copy widths and beta handling for supported
  nontransposed-B cases.
- Q6_K array offsets, MMVQ/dequantizer dispatch, dense-upload rollback cleanup,
  normal destruction, and the draft-head's four-array gather arithmetic.
- AVX2 sign expansion, IQ3_S gather indices and IQ2_S scale/sign mapping against
  the previous arithmetic; pool splitting retains whole-row ownership.
- Linux allocation/free pairing and Windows guards; SYCL-only setup worker flags
  and `physical_cores()` parsing.
- Selected prompt-attention and native FlashAttention barrier paths.
- `git diff --check 99f3dbd..intel-sycl` passes.

## Verification limits

This was not an exhaustive audit of every new kernel. CUDA/HIP/Windows builds,
GPU timing behavior, and actual regression-test failures remain unverified. The
failure scenarios above are derived from the code; they were not executed.

## Resolution (2026-10-06)

| # | Finding | Outcome |
|---|---|---|
| 1 | Shared GEMM staging scratch | Fixed: one scratch per queue, created under a mutex, grown on demand after waiting for the queue. |
| 2 | Doorbell checksum | Fixed: each 32-bit word is mixed with its position (murmur3's finalizer) before summing; reorderings and cancelling changes no longer match (a stale read passes with a chance near 2^-32). Still detection, not ordering - documented. |
| 3 | Synchronization drops async errors | Fixed: stream, device and event synchronization and the synchronous copies use `wait_and_throw` and return the recorded asynchronous error (sticky until `cudaGetLastError`). |
| 4 | Legacy default-stream ordering | Documented as a runtime contract limit in INTEL_SYCL.md; the engine uses explicit streams. Not changed: barriers in both directions on every submission risk the hangs the host waits replaced. |
| 5 | Host-blocking `cudaStreamWaitEvent` | Documented: the narrower contract is stated in the code and INTEL_SYCL.md; the engine never waits outside a capture on work that needs a later host action. |
| 6 | Transposed-B offset | Fixed: pieces advance B by one row per output column for a transposed B; a far transposed B is staged whole. Test added (fails with the old offset). |
| 7 | Silent unprotected fallback | Fixed: the scratch grows to fit; an allocation failure returns `CUBLAS_STATUS_ALLOC_FAILED` instead of running the affected GEMM. |
| 8 | Interleaved-C test bypassed staging | Fixed: a staged case (1024 x 128 outputs, beta = 0 then 1, sentinel rows); with the old whole-column copy it reports 131,072 overwritten values. The small case is kept for the small-output kernel. |
| 9 | Stale handoff checkpoint | Fixed: marked historical and updated. |

Verified: `sycl_gemm_4g` passes (and fails as expected with the old copy-back, the old B offset, and with staging off); ctest 58/60 (the two known host-limit failures); the engine's reply with a fixed expert placement is unchanged over two runs (16.75 and 16.87 tok/s).
