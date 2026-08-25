#!/usr/bin/env bash
# Intel/Qwen3.5-122B-A10B-int4-AutoRound on FOUR V620s as flat TP=4 —
# The flagship 4-card config (2026-08-25):
#   58-59 / 58 / 49-55 t/s decode at 3.8k/13k/41k, TTFT 4.2-4.9s,
#   MTP acceptance ~2.3 tok/step. vs PP=3 (39.9/33.1/22.2): +43-124%,
#   near-FLAT decode curve (4-way attention sharding kills the ctx term).
#   Validated: full benchmark + 3-pass soak, ~90 min sustained, 0 gpu events.
#
# TP=4 previously dropped cards off the bus on EVERY attempt. What changed —
# the full mitigation stack (unwind experiments pending; treat ALL of it as
# load-bearing until proven otherwise):
#   KERNEL LINE (prerequisite, user-applied):
#     amdgpu.pcie_gen_cap=0x00070007  (Gen3 link cap — verify via
#         pp_dpm_pcie, the ONLY honest link source on this board)
#     amdgpu.aspm=0 amdgpu.runpm=0    (no link/device low-power races)
#     amdgpu.gpu_recovery=1           (wedge -> reset, not dead card)
#     amdgpu.noretry=1 amdgpu.ras_enable=0 amdgpu.gartsize=4096
#   POWER (prerequisite, per boot): 232 W caps —
#     sudo rocm-smi -d <your V620 indices> --setpoweroverdrive 232
#   RUNTIME (below): HSA_NO_SCRATCH_RECLAIM=1 (no mid-flight scratch regrow),
#     NCCL_P2P_LEVEL=PXB (P2P within root complex, SHM across).
#   WORKLOAD (below): BATCHTOK=2048 — shortens every hazard window at once
#     (unpreemptible dispatch time, scratch crossing, DMA burst, power ramp).
#     These are GRAPHICS cards; keep work items frame-sized.
#   Plus: warm persistent Triton cache; prefer a graduated ramp (small
#     requests first) after cold boots.
#
# No MOE_CFG: TP=4 shards MoE to N=256 — the tuned N=1024 JSON doesn't
# apply (re-sweep queued). lm_head TunableOp rows likewise PP-only shapes.
# Shares this build dir's convert.py + quantize_mtp.py conversions.
IMG="${IMG:-vllm-gfx1030:0.27.1-patched}" \
MODEL="Intel/Qwen3.5-122B-A10B-int4-AutoRound" \
SERVED="qwen35-122b-autoround" \
QUANT="gptq" \
TP=4 PP=1 DEVICES="${DEVICES:-1,2,3,4}" \
MTP="${MTP:-2}" BATCHTOK="${BATCHTOK:-2048}" \
EXTRA_ENV="${EXTRA_ENV-VLLM_USE_V2_MODEL_RUNNER=1,HSA_NO_SCRATCH_RECLAIM=1,NCCL_P2P_LEVEL=PXB}" \
FD_RDNA2="${FD_RDNA2:-1}" AR_RDNA2=0 CUSTOM_AR=0 \
GPUUTIL="${GPUUTIL:-0.9}" MAXSEQS="${MAXSEQS:-4}" \
KVDTYPE="${KVDTYPE:-int8_per_token_head}" \
MAXLEN="${MAXLEN:-131072}" \
exec "$(dirname "$0")/../../config/serve-rdna2-tp2.sh" "$@"
