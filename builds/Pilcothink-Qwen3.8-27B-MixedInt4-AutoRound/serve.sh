#!/usr/bin/env bash
# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
#
# Start vLLM for Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound (AutoRound mixed
# int4/int8/fp16, GPTQ packing, symmetric g32).
#
# REQUIRES the two one-time checkpoint conversions in this directory — see
# BUILD.md: convert.py (auto-round config -> gptq + dynamic overrides) and
# mtp_to_dense.py (dequantize the int4 MTP head). Re-run both after any
# re-download. Symmetric GPTQ packing -> Exllama route, reference-build speed.
MODEL="Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound" \
SERVED="qwen38-27b-mixedint4" \
QUANT="gptq" \
MTP="${MTP:-2}" \
KVDTYPE="${KVDTYPE:-int8_per_token_head}" \
MAXLEN="${MAXLEN:-131072}" \
CUSTOM_AR=0 \
exec "$(dirname "$0")/../../config/serve-rdna2-tp2.sh" "$@"
