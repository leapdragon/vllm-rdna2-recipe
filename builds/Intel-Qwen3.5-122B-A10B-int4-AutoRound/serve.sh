#!/usr/bin/env bash
# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Start vLLM for Intel/Qwen3.5-122B-A10B-int4-AutoRound across three V620s as
# PP=3, with MTP speculative decoding: 39.7-41/33.4/21.9 t/s decode at
# 3.5k/13k/40k vs 27.0/25.7/22.1 without it — past llama.cpp on the same cards.
#
# Requirements (see BUILD.md for each):
#   - image with ALL NINE patches (0007/0008 MoE kernels; 0009 MTP-under-PP)
#     and the CURRENT fd_rdna2 plugin (GQA-generalized) installed
#   - the V2 model runner (V1's drafter path page-faults under PP)
#   - one-time conversions after download: convert.py (auto-round -> gptq)
#     AND quantize_mtp.py (int4-packs the 5 GB dense MTP head)
#   - TunableOp rows for the lm_head shapes (tn_151936_*) in your
#     tunableop_results*.csv — BUILD.md has the offline tuning one-liner;
#     without them the M<=4 logits GEMM runs ~4x slow
# PP_PARTITION unloads the last stage, which carries the drafter.
# Fallback (no MTP, V1 runner): MTP=0 EXTRA_ENV= PP_PARTITION= FD_RDNA2=0
MOE_CFG="${MOE_CFG:-$(cd "$(dirname "$0")" && pwd)/moe-config-gfx1030.json:E=256,N=1024,device_name=AMD_RADEON_PRO_V620_Azure,dtype=int4_w4a16.json}" \
MODEL="Intel/Qwen3.5-122B-A10B-int4-AutoRound" \
SERVED="qwen35-122b-autoround" \
QUANT="gptq" \
TP=1 PP=3 DEVICES="0,1,3" \
MTP="${MTP:-2}" PP_PARTITION="${PP_PARTITION-17,17,14}" \
EXTRA_ENV="${EXTRA_ENV-VLLM_USE_V2_MODEL_RUNNER=1}" \
FD_RDNA2="${FD_RDNA2:-1}" AR_RDNA2=0 CUSTOM_AR=0 \
GPUUTIL="${GPUUTIL:-0.95}" MAXSEQS="${MAXSEQS:-4}" \
KVDTYPE="${KVDTYPE:-int8_per_token_head}" \
MAXLEN="${MAXLEN:-131072}" \
exec "$(dirname "$0")/../../config/serve-rdna2-tp2.sh" "$@"
