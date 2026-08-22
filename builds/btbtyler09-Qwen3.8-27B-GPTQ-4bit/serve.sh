#!/usr/bin/env bash
# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
#
# Start vLLM for btbtyler09/Qwen3.8-27B-GPTQ-4bit (the reference build).
# The shared launcher's defaults ARE this model; this wrapper exists so every
# build in builds/ starts the same way. See BUILD.md here for the numbers.
KVDTYPE="${KVDTYPE:-int8_per_token_head}" MAXLEN="${MAXLEN:-131072}" CUSTOM_AR=0 \
exec "$(dirname "$0")/../../config/serve-rdna2-tp2.sh" "$@"
