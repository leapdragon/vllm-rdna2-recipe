#!/usr/bin/env bash
# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
#
# Qwen/Qwen3.5-122B-A10B-GPTQ-Int4 across THREE V620s as PP=3 (layer split).
# TP=3 is impossible for this model (2 KV heads); pipeline split is the route.
# See BUILD.md — decode is currently MoE-kernel-bound (~7 t/s); prefill
# 558-721 t/s with the tuned fused-MoE config this directory carries.
MOE_CFG="${MOE_CFG:-$(cd "$(dirname "$0")" && pwd)/moe-config-gfx1030.json:E=256,N=1024,device_name=AMD_RADEON_PRO_V620_Azure,dtype=int4_w4a16.json}" \
MODEL="Qwen/Qwen3.5-122B-A10B-GPTQ-Int4" \
SERVED="qwen35-122b-gptq" \
QUANT="gptq" \
TP=1 PP=3 DEVICES="0,1,3" \
MTP=0 \
FD_RDNA2=0 AR_RDNA2=0 CUSTOM_AR=0 \
GPUUTIL="${GPUUTIL:-0.95}" MAXSEQS="${MAXSEQS:-4}" \
KVDTYPE="${KVDTYPE:-int8_per_token_head}" \
MAXLEN="${MAXLEN:-131072}" \
exec "$(dirname "$0")/../../config/serve-rdna2-tp2.sh" "$@"
