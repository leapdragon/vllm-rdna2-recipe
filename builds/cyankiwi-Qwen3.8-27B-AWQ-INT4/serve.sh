#!/usr/bin/env bash
# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
#
# Start vLLM for cyankiwi/Qwen3.8-27B-AWQ-INT4 (compressed-tensors, asym W4 g32).
#
# Asymmetric uint4 is outside Exllama's supported types, so this build departs
# from the shared kernel disable list: it enables RDNAHybridW4A16 (HIP skinny
# GEMV decode + dequant-to-dense rocBLAS prefill), which needs patch
# 0006-rdna-hybrid-w4a16-gfx1030.patch baked into the image. See BUILD.md.
# patch 0006 (RDNA-hybrid W4A16) must be in your image \
MODEL="cyankiwi/Qwen3.8-27B-AWQ-INT4" \
SERVED="qwen38-27b-awq" \
QUANT="compressed-tensors" \
DISABLED_KERNELS="RDNA3W4A16LinearKernel,ConchLinearKernel" \
MTP="${MTP:-2}" \
KVDTYPE="${KVDTYPE:-int8_per_token_head}" \
MAXLEN="${MAXLEN:-131072}" \
CUSTOM_AR=0 \
exec "$(dirname "$0")/../../config/serve-rdna2-tp2.sh" "$@"
