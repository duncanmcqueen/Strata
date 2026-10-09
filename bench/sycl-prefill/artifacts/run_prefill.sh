#!/bin/bash
set -u
cd /home/dwmcqueen/Strata
export LD_LIBRARY_PATH=/home/dwmcqueen/Strata/engine:/opt/intel/oneapi/compiler/latest/lib:/opt/intel/oneapi/mkl/latest/lib:${LD_LIBRARY_PATH:-}
export UR_L0_ENABLE_RELAXED_ALLOCATION_LIMITS=1 ZES_ENABLE_SYSMAN=1 NEO_FP64_EMULATION=1
export ONEAPI_DEVICE_SELECTOR=level_zero:0
G=/home/dwmcqueen/models/Qwen3.8-Flash-Next-GSQ-RCO-Coder-GGUF/IQ1_M
exec ./engine/strata \
  --pack /home/dwmcqueen/Strata-data/packs/coder-iq1_m \
  --native "$G/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00001-of-00002.gguf" \
  --ple-gguf "$G/Qwen3.8-Flash-Next-GSQ-RCO-IQ1_M-00002-of-00002.gguf" \
  --expert-profile /home/dwmcqueen/Strata/data/expert-profile-coder.bin \
  --expert-cache auto --prefill "$1" --spec 4 --spec-min-p 0.5 \
  --mtp /home/dwmcqueen/Strata-data/mtp/rt --max-context 65536 --spec-split --kv int8 --resident-experts \
  --max-new 8 --tokens-file /tmp/prompt32k.ids
