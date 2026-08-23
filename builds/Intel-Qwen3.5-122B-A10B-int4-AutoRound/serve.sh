#!/usr/bin/env bash
# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Start vLLM for Intel/Qwen3.5-122B-A10B-int4-AutoRound across three V620s as
# PP=3. Successor to the official-GPTQ 122B build: same shapes and serving
# config, but this checkpoint quantizes the BACKBONE too (attention/GDN int4;
# only the shared expert stays fp16), removing ~30 ms/token of bf16 streaming.
#
# REQUIRES the one-time conversion in this directory (convert.py: auto-round
# config -> gptq + dynamic skips) after any download. MTP head is bf16 and
# unused (PP forbids speculation). MoE config JSON carries the gfx1030 sweep
# winners (same E=256/N=1024 shapes as the predecessor build).
MOE_CFG="${MOE_CFG:-$(cd "$(dirname "$0")" && pwd)/moe-config-gfx1030.json:E=256,N=1024,device_name=AMD_RADEON_PRO_V620_Azure,dtype=int4_w4a16.json}" \
# patch 0007 (moe_wna16) must be in your image \
MODEL="Intel/Qwen3.5-122B-A10B-int4-AutoRound" \
SERVED="qwen35-122b-autoround" \
QUANT="gptq" \
TP=1 PP=3 DEVICES="0,1,3" \
MTP=0 \
FD_RDNA2=0 AR_RDNA2=0 CUSTOM_AR=0 \
GPUUTIL="${GPUUTIL:-0.95}" MAXSEQS="${MAXSEQS:-4}" \
KVDTYPE="${KVDTYPE:-int8_per_token_head}" \
MAXLEN="${MAXLEN:-131072}" \
exec "$(dirname "$0")/../../config/serve-rdna2-tp2.sh" "$@"
