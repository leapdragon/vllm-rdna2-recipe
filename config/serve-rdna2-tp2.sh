#!/usr/bin/env bash
# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
# serve-rdna2-tp2.sh — max-performance two-card RDNA2 (gfx1030) vLLM server.
#
# Config lineage (see projects/vllm-rdna2/TESTS_RESULTS.md):
#   T12  Exllama dispatch + FULL_DECODE_ONLY graphs, TP=2  <- the winning TP=2 combo
#   T14/T15 gemv_plugin: NOT used here (validated only at TP=1; W4 GEMV was a
#           regression, lm_head GEMV neutral). Fewer moving parts for a live trial.
#   NEW  custom all-reduce ENABLED (peer P2P + peer atomics verified on devices 1+3,
#        2026-08-18: direct peer read/write/atomics clean, 14.04 GB/s, 28.4us @10KB).
#        Fall back with CUSTOM_AR=0 if it misbehaves.
set -euo pipefail

# ---------------------------------------------------------------------------
# Paths — every host path below is overridable via environment variables.
#
#   RECIPE_ROOT  this repo's root (auto-detected from this script's location);
#                holds the plugins and, by default, the writable state dirs
#   VLLM_SRC     your patched vLLM 0.27.1 checkout (the tree the image was
#                built from). OPTIONAL: when set, the key patched source files
#                are bind-mounted over the image so source edits deploy
#                without a rebuild; when unset, the image's baked copies run
#   HF_CACHE     HuggingFace cache holding the model weights
#   TUNEOP_DIR   TunableOp GEMM-tuning csv store
#   STATE_DIR    compile cache, torch extension cache, profiler traces
# ---------------------------------------------------------------------------
RECIPE_ROOT="${RECIPE_ROOT:-$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)}"
VLLM_SRC="${VLLM_SRC:-}"
HF_CACHE="${HF_CACHE:-$RECIPE_ROOT/hf-cache}"
TUNEOP_DIR="${TUNEOP_DIR:-$RECIPE_ROOT/tunableop}"
STATE_DIR="${STATE_DIR:-$RECIPE_ROOT/.state}"
mkdir -p "$HF_CACHE" "$TUNEOP_DIR" \
         "$STATE_DIR/ext-cache" "$STATE_DIR/traces"

# When VLLM_SRC is set, overlay the patched source files that change most often
# (attention, W4A16 kernels, platform gates) over the image's copies.
SRC_MOUNT=()
if [ -n "$VLLM_SRC" ]; then
  for _f in vllm/platforms/rocm.py \
            vllm/v1/attention/backends/triton_attn.py \
            vllm/v1/attention/ops/triton_unified_attention.py \
            vllm/model_executor/kernels/linear/mixed_precision/triton_w4a16.py \
            vllm/model_executor/kernels/linear/mixed_precision/rdna_hybrid_w4a16.py; do
    SRC_MOUNT+=(-v "$VLLM_SRC/$_f:/app/vllm-src/$_f:ro")
  done
fi

NAME="${NAME:-vllm-qwen38-tp2}"
TP="${TP:-2}"
PP="${PP:-1}"                             # pipeline-parallel stages (layer split); 1 = off
# Optional uneven layer split, e.g. PP_PARTITION=17,17,14 (must sum to the layer
# count). Use to unload the last stage when it carries extra weight — the MTP
# draft model lives entirely on the last PP rank.
PP_PART_ENV=()
[ -n "${PP_PARTITION:-}" ] && PP_PART_ENV=(-e VLLM_PP_LAYER_PARTITION="${PP_PARTITION}")
DEVICES="${DEVICES:-1,3}"                 # default V620 pair (2026-08-31 chain-walk: these root at
                                          # x8 Gen3; the 2,4 pair at x16 Gen3 — decode measured the
                                          # same on both, so the width does not matter for TP=2)
# The torch.compile/AOT cache is device-set-specific in practice: reusing a cache written on one
# pair from another pair crashed the worker with HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION as the
# MTP drafter's AOT artifacts loaded (2026-08-31, TROUBLESHOOTING 5b). Scope it by DEVICES.
COMPILE_CACHE_DIR="$STATE_DIR/compile-cache-${DEVICES//,/-}"
mkdir -p "$COMPILE_CACHE_DIR"
SERVE_EXTRA=""
# HW_QUEUES: hardware queues per device (GPU_MAX_HW_QUEUES). 4 is the measured optimum
# (+3.8%; 8 regresses 32% — 02-VERSIONS). Present since the first commit, silently dropped
# in the 2026-08-22 builds/ refactor, restored 2026-08-31 after the regression hunt.
# IMG: the image you built from the nine patches + plugins (02-VERSIONS build order), or the published
# one: IMG=ghcr.io/leapdragon/vllm-rdna2-recipe:0.27.1-rocm7.2.3-gfx1030 (see containers/README.md).
# The recorded numbers need a plugin-bearing image: one built before builds/shared/plugins existed
# boots fine but decodes at a fraction of the recorded rate (stock context slope, no fd_rdna2).
# DEV=1 instead mounts the repo's plugin trees and installs them at start, for
# iterating on plugin code without a rebuild.
IMG="${IMG:-vllm-gfx1030:0.27.1-patched}"
# Model under test. Defaults to the GPTQ quant every recorded number was measured on;
# override for experiments, e.g. MODEL=amd/Qwen3.8-27B-Quark-AWQ-MXFP4 QUANT=quark.
MODEL="${MODEL:-btbtyler09/Qwen3.8-27B-GPTQ-4bit}"
SERVED="${SERVED:-qwen38-27b-gptq}"
QUANT="${QUANT:-gptq}"
DTYPE="${DTYPE:-float16}"
# MTP speculative decoding. MTP=2 drafts two tokens per step; the head ships with the
# checkpoint (15 BF16 tensors, ~850 MB) and vLLM auto-detects qwen3_5 -> qwen3_5_mtp.
# Default 2 (validated 8/8, +24% @41k, T30); MTP=0 disables. Verification attention rides our
# batched multi-query kernel; without it (or with the pre-fix plugin) MTP is a 3.6x REGRESSION.
MTP="${MTP:-2}"
# SPEC_EAGER=1 runs only the DRAFTER eager (target keeps its cudagraphs) —
# isolates drafter-cudagraph interplay without changing the target's compile.
SPEC_ARG=""
SPEC_EAGER_FIELD=""
[ "${SPEC_EAGER:-0}" = "1" ] && SPEC_EAGER_FIELD=",\\\"enforce_eager\\\":true"
[ "${MTP}" != "0" ] && SPEC_ARG="--speculative-config {\\\"method\\\":\\\"qwen3_5_mtp\\\",\\\"num_speculative_tokens\\\":${MTP}${SPEC_EAGER_FIELD}}"
DEV_MOUNT=(); DEV_INSTALL=""
if [ "${DEV:-0}" = "1" ]; then
  DEV_MOUNT=(-v "$RECIPE_ROOT/builds/shared/plugins/fd_rdna2:/app/patches/fd_plugin"
             -v "$RECIPE_ROOT/builds/shared/plugins/ar_rdna2:/app/patches/ar_plugin")
  DEV_INSTALL="pip install --no-deps -q /app/patches/fd_plugin /app/patches/ar_plugin >/dev/null 2>&1; "
fi
CMODE="${CMODE:-3}"                       # 0 NONE, 1 stock compile, 2 dynamo-trace-once, 3 VLLM_COMPILE
CGMODE="${CGMODE:-FULL_DECODE_ONLY}"
[ "${PROFILE:-0}" = "1" ] && SERVE_EXTRA="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/traces --profiler-config.torch_profiler_with_stack=false"
# V1 async scheduling assumes width-1 sampled-token broadcasts between PP ranks,
# which speculative decoding violates — run MTP-under-PP with ASYNC_SCHED=0.
[ "${ASYNC_SCHED:-1}" = "0" ] && SERVE_EXTRA="$SERVE_EXTRA --no-async-scheduling"
CUSTOM_AR="${CUSTOM_AR:-1}"               # 1 = custom all-reduce (P2P), 0 = RCCL fallback
MAXLEN="${MAXLEN:-131072}"
KVDTYPE="${KVDTYPE:-auto}"          # auto|int8_per_token_head|int4_per_token_head|fp8_e5m2|fp8_e4m3
MAXSEQS="${MAXSEQS:-8}"
GPUUTIL="${GPUUTIL:-0.92}"
PORT="${PORT:-8000}"
VERBOSE="${VERBOSE:-0}"                   # 1 = deep engine diagnostics (needs this restart)

VERBOSE_FLAGS=""; VERBOSE_ENV=()
if [ "$VERBOSE" = "1" ]; then
  # --enable-log-requests logs PROMPT TEXT; do not use on real agent traffic you
  # care about keeping private.
  VERBOSE_FLAGS="--enable-logging-iteration-details --kv-cache-metrics --kv-cache-metrics-sample 1.0 --show-hidden-metrics-for-version 0.27"
  VERBOSE_ENV=(-e VLLM_LOG_BATCHSIZE_INTERVAL=10)
fi

AR_FLAG=""
CAR_ENV=()
if [ "$CUSTOM_AR" = "1" ]; then
  # vLLM hard-gates custom all-reduce to MI300 (gfx94/gfx95) in
  # RocmPlatform.use_custom_allreduce(); patch 0001's rocm.py adds an opt-in
  # RDNA branch (baked into the image, or overlaid via VLLM_SRC above).
  CAR_ENV=(-e VLLM_ROCM_FORCE_CUSTOM_ALLREDUCE=1)
else
  AR_FLAG="--disable-custom-all-reduce"
fi

# Optional tuned fused-MoE config: MOE_CFG="<host json path>:<configs filename>"
# mounts the file into vLLM's fused_moe configs dir (the boot log names the
# exact filename it looks for when it warns about a missing config).
MOE_MOUNT=()
if [ -n "${MOE_CFG:-}" ]; then
  MOE_MOUNT=(-v "${MOE_CFG%%:*}:/app/vllm-src/vllm/model_executor/layers/fused_moe/configs/${MOE_CFG##*:}:ro")
fi
# Generic extra bind mounts for iteration:
# EXTRA_MOUNT="<host>:<container>[,<host>:<container>...]"
if [ -n "${EXTRA_MOUNT:-}" ]; then
  IFS=',' read -ra _EXTRA <<< "${EXTRA_MOUNT}"
  for _em in "${_EXTRA[@]}"; do MOE_MOUNT+=(-v "${_em}:ro"); done
fi
# Generic extra environment for experiments/diagnostics, e.g.
# EXTRA_ENV="VLLM_USE_V2_MODEL_RUNNER=1"
EXTRA_ENV_ARGS=()
if [ -n "${EXTRA_ENV:-}" ]; then
  IFS=',' read -ra _EENV <<< "${EXTRA_ENV}"
  for _ee in "${_EENV[@]}"; do EXTRA_ENV_ARGS+=(-e "${_ee}"); done
fi

# Host groups that own /dev/kfd and /dev/dri/renderD*; the numeric ids differ between distros.
RENDER_GID="${RENDER_GID:-$(getent group render | cut -d: -f3)}"
RENDER_GID="${RENDER_GID:-$(stat -c %g /dev/dri/renderD128 2>/dev/null || echo 991)}"
VIDEO_GID="${VIDEO_GID:-$(getent group video | cut -d: -f3)}"
VIDEO_GID="${VIDEO_GID:-44}"

# Do not leave a stale container behind (the previous --rm setup vanished on exit,
# taking its exit status with it).
docker rm -f "$NAME" >/dev/null 2>&1 || true

exec docker run -d --name "$NAME" --network=host \
  --device /dev/kfd --device /dev/dri --group-add "$RENDER_GID" --group-add "$VIDEO_GID" \
  --ipc=host --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ulimit memlock=-1 \
  --restart=no \
  -e HEALTHCHECK_PORT="$PORT" \
  -v "$HF_CACHE:/root/.cache/huggingface" \
  -v "$TUNEOP_DIR:/tuning" \
  -e FD_RDNA2="${FD_RDNA2:-1}" -e FD_MAXQ="${FD_MAXQ:-4}" -e FD_CHUNK="${FD_CHUNK:-512}" -e FD_TILE="${FD_TILE:-32}" -e FD_WARPS="${FD_WARPS:-8}" \
  -v "$STATE_DIR/ext-cache:/ext-cache" \
  -e AR_RDNA2="${AR_RDNA2:-1}" -e AR_MAX_KB="${AR_MAX_KB:-512}" -e TORCH_EXTENSIONS_DIR=/ext-cache \
  -e PYTORCH_ROCM_ARCH=gfx1030 \
  "${SRC_MOUNT[@]}" "${CAR_ENV[@]}" "${VERBOSE_ENV[@]}" "${PP_PART_ENV[@]}" \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  -e ROCR_VISIBLE_DEVICES="$DEVICES" \
  -e VLLM_TARGET_DEVICE=rocm \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  -e VLLM_ROCM_USE_AITER=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e GPU_MAX_HW_QUEUES="${HW_QUEUES:-4}" \
  -v "$RECIPE_ROOT/.triton-cache:/triton-cache" \
  -e TRITON_CACHE_DIR=/triton-cache \
  -e PYTORCH_ALLOC_CONF="expandable_segments:${EXPANDABLE_SEGMENTS:-True}" \
  -e VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${EXEC_TIMEOUT:-300}" \
  -e VLLM_LOG_STATS_INTERVAL="${STATS_INTERVAL:-10}" \
  -v "$STATE_DIR/traces:/traces" \
  -v "$COMPILE_CACHE_DIR:/compile-cache" \
  "${DEV_MOUNT[@]}" "${MOE_MOUNT[@]}" "${EXTRA_ENV_ARGS[@]}" \
  -e VLLM_CACHE_ROOT=/compile-cache \
  "${PROF_ARGS[@]}" \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 \
  -e PYTORCH_TUNABLEOP_TUNING="${TUNEOP_TUNING:-0}" \
  -e PYTORCH_TUNABLEOP_FILENAME=/tuning/tunableop_results.csv \
  -e TOKENIZERS_PARALLELISM=false -e OMP_NUM_THREADS=8 \
  -e VLLM_DISABLED_KERNELS="${DISABLED_KERNELS:-RDNA3W4A16LinearKernel,RDNAHybridW4A16LinearKernel,TritonW4A16LinearKernel,ConchLinearKernel}" \
  --entrypoint sh "$IMG" -c \
  "${DEV_INSTALL}exec vllm serve $MODEL --served-model-name $SERVED --enable-prompt-tokens-details \
    --port "$PORT" \
    ${SERVE_EXTRA} \
    ${SPEC_ARG} \
    --dtype $DTYPE --quantization $QUANT --attention-backend TRITON_ATTN \
    --tensor-parallel-size "$TP" --pipeline-parallel-size "$PP" $AR_FLAG $VERBOSE_FLAGS \
    --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPUUTIL" \
    --kv-cache-dtype "$KVDTYPE" \
    --max-num-seqs "$MAXSEQS" --max-num-batched-tokens "${BATCHTOK:-8192}" \
    --enable-chunked-prefill --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
    --language-model-only --skip-mm-profiling \
    --compilation-config '{\"mode\":'\"$CMODE\"',\"cudagraph_mode\":\"'\"$CGMODE\"'\"}'"
