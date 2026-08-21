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

NAME="${NAME:-vllm-qwen38-tp2}"
TP="${TP:-2}"
DEVICES="${DEVICES:-1,3}"                 # the two x16-rooted V620s
SERVE_EXTRA=""
# The optimisation set (W4 blocking, both plugins) is baked into the -v2 image. DEV=1 instead
# mounts the working plugin trees and installs them at start, for iterating without a rebuild.
IMG="${IMG:-vllm-gfx1030:0.27.1-patched-v2}"
DEV_MOUNT=(); DEV_INSTALL=""
if [ "${DEV:-0}" = "1" ]; then
  DEV_MOUNT=(-v /home/perfekt/repos/vllm-rdna2/dev/ws1-attention/src/fd_plugin:/app/patches/fd_plugin
             -v /home/perfekt/repos/vllm-rdna2/dev/ws2-allreduce/src/ar_plugin:/app/patches/ar_plugin)
  DEV_INSTALL="pip install --no-deps -q /app/patches/fd_plugin /app/patches/ar_plugin >/dev/null 2>&1; "
fi
CMODE="${CMODE:-3}"                       # 0 NONE, 1 stock compile, 2 dynamo-trace-once, 3 VLLM_COMPILE
CGMODE="${CGMODE:-FULL_DECODE_ONLY}"
[ "${PROFILE:-0}" = "1" ] && SERVE_EXTRA="--profiler-config.profiler=torch --profiler-config.torch_profiler_dir=/traces --profiler-config.torch_profiler_with_stack=false"
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
CAR_MOUNT=(); CAR_ENV=()
if [ "$CUSTOM_AR" = "1" ]; then
  # vLLM hard-gates custom all-reduce to MI300 (gfx94/gfx95) in
  # RocmPlatform.use_custom_allreduce(). gfx1030-custom-allreduce.patch adds an
  # opt-in RDNA branch; mount the patched file over the image's editable source.
  CAR_MOUNT=(-v /home/perfekt/repos/vllm-rdna2/vllm-0.27.1/vllm/platforms/rocm.py:/app/vllm-src/vllm/platforms/rocm.py:ro)
  CAR_ENV=(-e VLLM_ROCM_FORCE_CUSTOM_ALLREDUCE=1)
else
  AR_FLAG="--disable-custom-all-reduce"
fi

# Do not leave a stale container behind (the previous --rm setup vanished on exit,
# taking its exit status with it).
docker rm -f "$NAME" >/dev/null 2>&1 || true

exec docker run -d --name "$NAME" --network=host \
  --device /dev/kfd --device /dev/dri --group-add 991 --group-add 44 \
  --ipc=host --cap-add SYS_PTRACE --security-opt seccomp=unconfined --ulimit memlock=-1 \
  --restart=no \
  -v /home/perfekt/repos/vllm-rdna2/hf-cache:/root/.cache/huggingface \
  -v /home/perfekt/repos/vllm-rdna2/tunableop:/tuning \
  -e FD_RDNA2="${FD_RDNA2:-1}" -e FD_CHUNK="${FD_CHUNK:-512}" -e FD_TILE="${FD_TILE:-32}" -e FD_WARPS="${FD_WARPS:-8}" \
  -v /home/perfekt/repos/vllm-rdna2/.ext-cache:/ext-cache \
  -e AR_RDNA2="${AR_RDNA2:-1}" -e AR_MAX_KB="${AR_MAX_KB:-512}" -e TORCH_EXTENSIONS_DIR=/ext-cache \
  -e PYTORCH_ROCM_ARCH=gfx1030 \
  -v /home/perfekt/repos/vllm-rdna2/vllm-0.27.1/vllm/v1/attention/backends/triton_attn.py:/app/vllm-src/vllm/v1/attention/backends/triton_attn.py:ro \
  "${CAR_MOUNT[@]}" "${CAR_ENV[@]}" "${VERBOSE_ENV[@]}" \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
  -e ROCR_VISIBLE_DEVICES="$DEVICES" \
  -e VLLM_TARGET_DEVICE=rocm \
  -e TORCH_BLAS_PREFER_HIPBLASLT=0 \
  -e FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
  -e VLLM_ROCM_USE_AITER=0 -e VLLM_ROCM_USE_AITER_MOE=0 \
  -e HIP_FORCE_DEV_KERNARG=1 \
  -e PYTORCH_ALLOC_CONF=expandable_segments:True \
  -e GPU_MAX_HW_QUEUES="${HW_QUEUES:-4}" \
  -v /home/perfekt/repos/vllm-rdna2/profile/ws3-traces:/traces \
  -v /home/perfekt/repos/vllm-rdna2/.compile-cache:/compile-cache \
  "${DEV_MOUNT[@]}" \
  -e VLLM_CACHE_ROOT=/compile-cache \
  "${PROF_ARGS[@]}" \
  -e VLLM_WORKER_MULTIPROC_METHOD=spawn \
  -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 \
  -e PYTORCH_TUNABLEOP_FILENAME=/tuning/tunableop_results.csv \
  -e TOKENIZERS_PARALLELISM=false -e OMP_NUM_THREADS=8 \
  -e VLLM_DISABLED_KERNELS=RDNA3W4A16LinearKernel,RDNAHybridW4A16LinearKernel,TritonW4A16LinearKernel,ConchLinearKernel \
  --entrypoint sh "$IMG" -c \
  "${DEV_INSTALL}exec vllm serve btbtyler09/Qwen3.8-27B-GPTQ-4bit --served-model-name qwen38-27b-gptq \
    --port "$PORT" \
    ${SERVE_EXTRA} \
    --dtype float16 --quantization gptq --attention-backend TRITON_ATTN \
    --tensor-parallel-size "$TP" $AR_FLAG $VERBOSE_FLAGS \
    --max-model-len "$MAXLEN" --gpu-memory-utilization "$GPUUTIL" \
    --kv-cache-dtype "$KVDTYPE" \
    --max-num-seqs "$MAXSEQS" --max-num-batched-tokens 8192 \
    --enable-chunked-prefill --enable-prefix-caching \
    --enable-auto-tool-choice --tool-call-parser qwen3_xml --reasoning-parser qwen3 \
    --language-model-only --skip-mm-profiling \
    --compilation-config '{\"mode\":'\"$CMODE\"',\"cudagraph_mode\":\"'\"$CGMODE\"'\"}'"
