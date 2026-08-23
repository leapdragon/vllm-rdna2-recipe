# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# BUILD — Intel/Qwen3.5-122B-A10B-int4-AutoRound

**Model**: `Intel/Qwen3.5-122B-A10B-int4-AutoRound` — AutoRound int4 (sym g128, GPTQ
packing) of the 122B MoE hybrid. 71.7 GB. Served as `qwen35-122b-autoround`, PP=3 across
all three V620s. **Replaced `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` on 2026-08-23** — that
build's bring-up record (PP enablement, MoE config sweep, moe_wna16 port) lives in
TESTS_RESULTS T35–T37 and the 20260822/23 nightly notes; this directory carries its
inherited assets (moe-config-gfx1030.json, moe_sweep.py — same E=256/N=1024 shapes).

## Why this checkpoint (and why the official one left)

The official Qwen GPTQ quantized **only the experts**, leaving ~18 GB of bf16 backbone
streaming every token — worse than the bytes suggest, since gfx1030 has no native bf16 and
pays a penalized GEMV path. Metadata inspection (no download needed) showed Intel's quant
covers the backbone too: every attention/GDN linear int4, only the **shared expert (+gate)
kept fp16** (144 modules), MTP head entirely bf16 and outside the quantize list. Result:

| | official GPTQ | this checkpoint (v4) | **+ skinny MoE kernel (v5, T38)** |
|---|---|---|---|
| **Decode** | 7.1 t/s | 15.7 / 15.0 | **26.9 / 25.4 / 21.8 t/s** (3.5k/14k/42k) |
| **Prefill** | 558 / 721 t/s | 566 / 741 | 569 / 763 t/s |

Factual spot-checks PASS; `baseline.json` is this model's own.

## Conversion required (script in this directory)

vLLM 0.27.1 has no "auto-round" method: `convert.py` rewrites the snapshot's config to
`gptq` + `dynamic` skip patterns (the shared-expert exclusions, gate/up mapped to
vLLM's fused `gate_up_proj`, plus `-:mtp\..*` — vLLM's built-in trigger to build the MTP
draft unquantized while the head is dense). Same class of surgery as the Pilcothink 27B
build. Re-run after any re-download. Weights untouched; metadata translation only.

**For MTP** (optional, see below): `quantize_mtp.py` RTN-packs the draft head's 768 expert
linears to the checkpoint's own int4 format (5.05 → 1.47 GB — dense, the draft starves the
last PP stage's KV at 131k) and removes the `-:mtp` skip so the draft builds quantized.
Experts only, deliberately: vLLM's AutoGPTQ *linear* gate decides quantized-vs-dense from
safetensors metadata fetched from the HF hub (local headers only as offline fallback), so
locally-packed non-expert linears would build dense regardless; RoutedExperts follow the
config directly. Idempotent; restores from `.dense-bak` when re-run.

## The corrected token budget, and what's next

~37 ms/token (was 64) ≈ ~5 ms skinny MoE + ~10 ms int4 backbone + ~22 ms
router/GDN/PP/other. The wvSplitK-class expert GEMV (T38, image v5) delivered the MoE leg:
432 GB/s effective, silu fused, no atomics, gated to M ≤ 8 (prefill keeps the swept Triton
config; `VLLM_ROCM_MOE_SKINNY=0` disables). Decode now sits at llama-server parity at
short context (26.9 vs their 30+); the remaining lever is profiling the ~22 ms of
router/GDN/PP overhead.

## MTP under PP: works, not yet profitable (T39, patch 0009)

The old "PP forbids speculation" wall is down (upstream PR #46994 backport + our V1
guards + the draft packing above), **but only on the V2 model runner**
(`VLLM_USE_V2_MODEL_RUNNER=1`) — the V1 runner's drafter path page-faults under PP, an
upstream-untested bug (their entire PP validation ran V2). Validated here: correct greedy
output, **81% acceptance at K=2** (mean 2.39 tokens/step — the RTN draft proposes well),
zero faults. Measured: MTP=2 decode **regresses** (20.9/12.9/4.9 t/s vs 27.0/25.7/22.1 at
MTP=0) — verification attention (q_len=3) rides the context-proportional prefill-class
kernel path, T29's disease at new shapes (2 KV heads, head_dim 256). Until a batched-MQ
kernel covers those shapes, **production stays `MTP=0` on the V1 runner** — V2 at MTP=0
measures identically to V1 (decode and prefill), so the future MTP campaign carries no
platform tax. To boot the MTP config: `MTP=2 PP_PARTITION=17,17,14
EXTRA_ENV=VLLM_USE_V2_MODEL_RUNNER=1` after running `quantize_mtp.py`, with the patch-0009
files in the image (or mounted).

## Working configuration

Inherited from the predecessor unchanged: `TP=1 PP=3 DEVICES=0,1,3`, image v5 (skinny MoE
GEMV, T38, on top of the moe_wna16
port), `GPUUTIL=0.95 MAXSEQS=4`, `MTP=0` (speculation works only on the V2 runner and is
currently a decode regression — see the MTP section above), `FD_RDNA2=0 AR_RDNA2=0`,
MOE_CFG → the swept fused-MoE config. Watch item also inherited: flaky load-time worker
death (~2-in-7 boots; plain retry works).
