# Comparison: upstream base v0.1.40.2 vs this branch, on an idle B70

Date: 2026-10-07. Machine: Intel Arc Pro B70 (`bmg`, id `0xE223`), 31.89 GiB VRAM, Level Zero device 0,
oneAPI DPC++ 2026.0. Model: **Coder IQ1_M** (the B70 benchmark model in upstream's release notes), native pack,
`data/expert-profile-coder.bin`, MTP draft layer (`Strata-data/mtp/rt`). Both engines: all 12,288 experts in
VRAM (upstream `--stream-experts`; this branch `--expert-cache auto` with no resident-RAM copy) = 23.42 GiB.

The GPU was made idle for this comparison by stopping the user's `vllm` container (`qwen38`) and the
`arcaine-dev-run` container. (Both were re-checked: `qwen38` has restart policy `no`; it did auto-start once
mid-session, which is why the VRAM numbers moved around earlier.)

## The two ports are different implementations

| | upstream `Niko1221/Strata` v0.1.40.2 | this branch (`intel-sycl`) |
| --- | --- | --- |
| Port | DPCT migration in a separate `sycl/` CMake project (`src/*.cu → sycl/src/*.dp.cpp`, vendored `dpct/`); upstream files never edited | hand-written per-kernel SYCL (`src/kernels/sycl/`) + CUDA-shaped runtime shim (`src/sycl_runtime/`, `sycl_compat/`) |
| Build | `cmake -S sycl -B build-sycl -DSTRATA_SYCL_AOT=bmg-g31` (JIT when empty) | `cmake -S . -B build-sycl -DSTRATA_ENABLE_SYCL=ON -DSTRATA_SYCL_AOT_DEVICES=…` |
| Device handling | one image, JIT/one AOT device; tested B70 + A750 | per-architecture AOT + the new section-1A device-profile dispatch |

## Measured on an idle B70 (single runs unless noted)

Both engines: `--spec 4 --spec-min-p 0.5 --mtp … --greedy`. Upstream also `STRATA_VERIFY_DEVICE_PLAN=1
STRATA_VERIFY_NO_HOST=1 --stream-experts --no-prefill-borrow`; this branch `--spec-split --kv int8`.

| workload | upstream v0.1.40.2 | this branch | upstream / this |
| --- | ---: | ---: | ---: |
| short prompt prefill (255 tok) | **318.9 tok/s** | 98.6 tok/s | 3.23× |
| short decode (256 tok, MTP) | **59.9 tok/s** | 28.9 tok/s | 2.07× |
| 8K prefill (8,225 tok) | **1,036.9 / 1,030.0** (median 1,033) | 408.6 / 781.4 / 258.4 / 777.3 (median ~593) | ~1.74× median, but this branch swings 258–781 |
| 8K decode (64 tok, MTP) | **84.6 / 81.9** (median 83) | 33.8 / 33.9 / 32.6 / 34.1 (median 33.8) | ~2.46× |
| 32K prefill (32,903 tok) | **997.6 tok/s** | 504.6 tok/s | 1.98× |
| 32K decode (64 tok, MTP) | **74.9 tok/s** | 24.7 tok/s | 3.03× |

Upstream is faster on every axis and much more consistent. It reproduced its advertised B70 design point on this
hardware: **8K decode 84.6 tok/s (81% drafts accepted, 3.05 tokens/round) vs the release note's 78.2**; 8K prefill
1,037 vs the note's ~1,000–1,117; 32K prompt 998 vs 888–1,117.

This branch is stable on decode (~33.8 tok/s) but its prefill is erratic (258–781 tok/s across four identical
8K runs). Before this session, under GPU contention, this branch's runner (resident-RAM mode) measured
17.3 tok/s decode at 8K — the idle-GPU all-in-VRAM config roughly doubles that, to 33.8.

## Secondary observations

- **Upstream abort at teardown:** the no-MTP runs (`--spec 4` without `--mtp`) exited 134 (SIGABRT) after
  printing their results, both short and 8K; every MTP run exited 0. It is a teardown-only crash, not a wrong
  answer, but worth reporting upstream.
- Upstream's log carries "this engine was compiled with CUDA 12.8 headers but loaded a CUDA 0.2 runtime
  (libcudart)" — on a pure SYCL build that message looks spurious.
- This branch's `--resident-experts` (its runner default) is not accepted by upstream's port (it wants
  `--stream-experts`), so the two runners' configurations are not interchangeable.

## Raw logs

`logs/upstream-full-short.log`, `upstream-full-8k.log`, `upstream-full-mtp-short.log`,
`upstream-full-mtp-8k{,2,3}.log`, `upstream-full-mtp-32k.log`;
`logs/local-full-short.log`, `local-full-8k{,-r2,-r3,-r4}.log`, `local-full-32k.log`.
