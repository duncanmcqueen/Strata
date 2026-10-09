# Best-model, greatest-context comparison: Strata (SYCL) vs vLLM, on one Arc Pro B70

Date: 2026-10-07/08. Machine: Intel Arc Pro B70 (`bmg`, `0xE223`), 31.89 GiB VRAM, Level Zero device 0,
oneAPI DPC++ 2026.0. GPU made idle for every run (the user's `qwen38` systemd service and the
`arcaine-dev-run` container were stopped; `qwen38` is restored at the end).

Question: **the most advanced coding model we can configure, at the greatest context, under each engine.**

## The two engines run *different base models* — this is an engine comparison, not same-weights

| | **Strata** (upstream v0.1.40.3, SYCL port) | **vLLM** (`qwen38`) |
|---|---|---|
| Model | Qwen3.8-Flash-Next **GSQ-RCO IQ3_XXS**, original (not the Coder) | Swift-1.5-**Qwen3.8-27B** GPTQ Int4 |
| Arch | `qwen4exp` **MoE**: 48 layers, 512 experts/layer top-10, hidden 2560, sparse indexer + PLE | `qwen3_5` **dense** hybrid: 64 layers (Gated-DeltaNet, 1 full-attn / 4), hidden 5120 |
| Weights | 75.84 GB, 2 shards, native pack (`dense.bin` 1.44 GiB, no `experts.bin`) | 17.38 GiB (W4A16 g128), 5 safetensors |
| Spec decode | MTP draft layer (base checkpoint's, `Strata-data/mtp/rt`), `--spec 4` | MTP-4 (`num_speculative_tokens: 4`, BF16 head) |
| KV | int8, `--kv-resident 32768` (KV in pinned host RAM, attended window in VRAM) | fp8 |
| Where experts live | VRAM cache (26 GiB, 11,638 slots) + **14.7 GiB mirrored in pinned host RAM** (9,015 experts); 450K–584K reads streamed from SSD | n/a (dense) |
| Native context | 262,144 | 262,144 |

Exact commands: Strata `build-sycl/strata --pack … --native S1 --ple-gguf S2 --expert-profile data/expert-profile.bin
--expert-cache auto --stream-experts --prefill 4096 --spec 4 --spec-min-p 0.5 --mtp <base> --tokens-file P
--max-context 262144 --kv int8 --kv-resident 32768 --greedy`.
vLLM `vllm serve /model --quantization gptq --dtype float16 --max-model-len 180000 --gpu-memory-utilization 0.93
--kv-cache-dtype fp8 --max-num-seqs 2 --max-num-batched-tokens 8192 --no-enable-prefix-caching
--speculative-config {method: mtp, num_speculative_tokens: 4}`.

## Measured (idle B70, greedy, single run each; 128 new tokens Strata / 64 vLLM)

| prompt tokens | Strata prefill tok/s | Strata decode tok/s | vLLM prefill tok/s | vLLM decode tok/s |
|---:|---:|---:|---:|---:|
| 131,072 | 417.5 | 24.5 | **781.7** | **53.8** |
| 160,000 | 380.3 | 22.4 | **688.2** | **47.5** |
| 179,000 | — | — | **638.3** | **48.6** |
| 199,999 | 388.5 | 10.2¹ | not servable (see below) | |
| 260,999 | **384.0** | 25.4 | not servable (see below) | |

¹ The 200K decode is an outlier against 131K/160K/261K (22–25); draft-window acceptance on that text, not a
trend. A repeat would be needed to pin it.

**vLLM is ~1.8× faster on prompt and ~2.1× faster on decode at the shared 160K point.** Both scale smoothly
with context.

## Context ceiling — the decisive difference

- **Strata served a 260,999-token prompt** in a 262,144 window (its full native context), at 384 tok/s prompt /
  25.4 tok/s decode. Metadata: `qwen4exp.context_length = 262144`.
- **vLLM cannot reach 200K here.** The B70 gives vLLM ~8.0 GiB of KV (fp8, 39 KiB/token). Its physical KV pool
  tops out at **~193,000 tokens**; empirically it served up to **179,000** tokens.
  - With `--max-model-len 196352` the KV pool is exactly **1.00× concurrency** for a 196K request, leaving **no
    room for the 4 MTP draft tokens**; the request sat in the scheduler as `Waiting: 1 reqs` until the client
    (1800 s) timed out, twice. This is a real limitation of vLLM's *best* (speculative) config here, not just a
    flag: turning MTP off would cost the 4-token/round speedup.
  - The model's own README says it is "validated serving up to 131,072".
- **Result: Strata gives ~1.45× the context of vLLM's practical ceiling (262K vs ~180K)** on this card, which is
  the property the request asked for ("greatest context").

## Quality is not established by these numbers

Different bases and different roles: Strata here runs the *original* Flash-Next IQ3_XXS (a general MoE, 512
experts); vLLM runs a *27B dense* Swift fine-tune. This session measured **throughput and context**, not coding
accuracy. A coding-quality comparison would need a shared benchmark (e.g. LCB/HumanEval) and, for Strata, the
**Coder** fine-tune — but the Coder has only one size (IQ1_M, half the experts) and is the model this branch's
earlier work used.

## Engine update to v0.1.40.3

The upstream base under test was updated from **v0.1.40.2 → v0.1.40.3** and rebuilt (`build-sycl/strata`).
v0.1.40.3 is Intel-relevant: it adds **Arc A-series/A770** support notes and config (`c454a4b`, `0130159`), a
**512-expert MTP router guard** (`ca016f2`, directly relevant to this 512-expert run), and removes the spurious
"compiled with CUDA headers … loaded a CUDA 0.2 runtime" warning — the verify lines now read
`captured the N-token window (upload no error, sync no error)` instead of `<FIXME: Placeholder>`.

Same Strata workload, 200K prompt, before/after the update:

| engine | prefill tok/s | decode tok/s |
|---|---:|---:|
| v0.1.40.2 | 314.7 | 14.1 |
| v0.1.40.3 | **388.5** | 10.2 |

## Raw logs

`logs/iq3xxs-{131k,160k,200k,261k}-ctx262144-int8-v01403.log`, `logs/iq3xxs-200k-ctx262144-int8.log` (v0.1.40.2),
`logs/iq3xxs-smoke.log`; `logs/vllm-{131k,160k,179k}.log`, `logs/vllm-131k-ignoreeos.log`.
Harness: `bench-vllm.py`, `run-iq3xxs.sh`, `pack-iq3xxs.sh`, `launch-vllm-bench.sh` (in `Strata-v0140/`).

## Confirmation: the published numbers track on the *Coder* (the model they were measured with)

Same engine (v0.1.40.3), same card, same long-context flags; **Coder IQ1_M** (256 experts/layer,
`data/expert-profile-coder.bin`, `--stream-experts`):

| prompt tokens | Strata prefill tok/s | Strata decode tok/s | published INTEL.md (Coder) |
|---:|---:|---:|---:|
| 131,072 | 576.1 | **94.95** | 888 / 66.6 |
| 260,999 | 515.2 | **93.70** | 757 / 55.9 |

Decode matches and *exceeds* the published figures; prefill is ~65–68%. The prefill gap is explainable:
at these contexts the auto expert cache fits only ~10,064 slots (25.9 GiB VRAM), not the 12,288 the
Coder occupies when fully resident, so 35K–71K experts are read from the pinned host mirror during the
prompt (35.6K / 70.9K reads, 57 s / 114 s host time). Decode is unaffected. This isolates the earlier
IQ3_XXS gap as **the model** (512 experts, ~14.7 GiB off-card), not the engine.

## Quick coding test (6 tasks, greedy, chat-formatted)

Both Strata models received a real chat prompt (`<|im_start|>…`); vLLM used `/v1/chat/completions`.
Assertions run on the model's final parseable function.

| task | Strata Coder IQ1_M | Strata IQ3_XXS | vLLM Swift-1.5-27B |
|---|---:|---:|---:|
| fib | 4/4 | 4/4 | 4/4 |
| is_prime | 6/6 | 6/6 | 6/6 |
| reverse_words | 3/3 | 3/3 | 3/3 |
| two_sum | 3/3 | 3/3 | 3/3 |
| flatten | 3/3 | 3/3 | 3/3 |
| roman | 3/3 | 3/3 | 3/3 |
| **TOTAL** | **22/22** | **22/22** | **22/22** |

Methodology gotcha: a naive first-block extractor produced *wrong* scores both ways — the Strata models,
run as raw completions, emitted a spurious new turn (`<|im_start|>user`) that a
`skip_special_tokens=True` decode glued onto the code; and vLLM, run in chat mode, put a reasoning
preamble with an abbreviated draft (`{'I':1,...}`) before the real function. Both are harness artifacts.
With chat formatting and a "pick the last parseable function" extractor, all three pass everything.

Judgement: on tasks this easy the three are indistinguishable — this only establishes a floor, not
coding quality. The one behavioural difference seen is verbosity: vLLM's Swift-27B emits a reasoning
preamble (needed up to 800 tokens to finish `flatten`/`roman`), while both Strata models finished within
320. A discriminating quality test needs harder, multi-step problems (e.g. LCB) — not built here.

Raw data: `logs/coding_{coder,iq3xxs,vllm}.json`, `logs/coding_scored_*.json`,
`logs/coder-prompt{131k,261k}-ctx262144-int8-v01403.log`.
