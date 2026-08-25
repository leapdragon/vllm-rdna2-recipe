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

| | official GPTQ | this checkpoint (v4) | + skinny MoE kernel (v5, T38) | **+ MTP=2 (v6, T40)** |
|---|---|---|---|---|
| **Decode** | 7.1 t/s | 15.7 / 15.0 | 26.9 / 25.4 / 21.8 (3.5k/14k/42k) | **39.7–41 / 33.4 / 21.9 t/s** |
| **Prefill** | 558 / 721 t/s | 566 / 741 | 569 / 763 t/s | (unchanged at MTP=0) |

At 39.7+ t/s short-context, vLLM on these cards now **beats llama-server's 30+** on the
same model — the gap this campaign started from is closed and inverted.

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

## MTP under PP: the production config (patch 0009 + plugin update)

The old "PP forbids speculation" wall is down (upstream PR #46994 backport + our V1
guards + the draft packing above), **but only on the V2 model runner**
(`VLLM_USE_V2_MODEL_RUNNER=1`) — the V1 runner's drafter path page-faults under PP, an
upstream-untested bug (their entire PP validation ran V2). V2 itself is free: at MTP=0 it
measures identically to V1 on decode and prefill.

Getting MTP from correct-but-slower to the headline numbers took two more findings (T40):

1. **Verification attention** (q_len ≤ 4) was riding the context-proportional
   prefill-class kernel path — T29's disease at new shapes. Fixed by generalizing the
   `fd_rdna2` flash-decode plugin from the 27B's GQA 6 to any GQA ≤ 16 (the kernels were
   already parameterized; the wrapper derived `n_kv` from an assumed GQA). nq=2 rides the
   one-KV-pass batched path; nq=3–4 take per-position passes of the same fast kernel.
   The drafter's own propose attention rides it too. Offline test:
   `bench/attn/fd_gqa_test.py` (16/16 correctness vs fp32 reference).
2. **The logits GEMM** (fp16 lm_head, 3072×151936) ran an untuned Tensile pick at ~87 GB/s
   — 10.6 ms *per verify step* at M=3, plus the draft's M=1 calls.
   `vllm::rocm_unquantized_gemm`'s skinny fast path is CDNA-gated, and in-server TunableOp
   tuning never reached these calls. Fix: tune the shapes **offline** and merge the rows
   into the per-rank CSVs (409/379 GB/s → ~2.3–2.5 ms):

   ```bash
   docker run --rm -i --entrypoint python3 --device /dev/kfd --device /dev/dri \
     --group-add 991 --group-add 44 -e HSA_OVERRIDE_GFX_VERSION=10.3.0 \
     -e PYTORCH_TUNABLEOP_ENABLED=1 -e PYTORCH_TUNABLEOP_TUNING=1 \
     -e PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0 \
     -e PYTORCH_TUNABLEOP_FILENAME=/tuning/probe.csv \
     -v <repo>/tunableop:/tuning <image> - <<'PY'
   import torch
   W = torch.randn(151936, 3072, dtype=torch.float16, device="cuda")
   for M in (1, 3, 4):
       torch.nn.functional.linear(torch.randn(M, 3072, dtype=torch.float16,
                                              device="cuda"), W)
   PY
   # then append probe0.csv's tn_151936_* rows to tunableop_results{0,1,2}.csv
   ```

Config of record (this directory's `serve.sh`): `MTP=2
PP_PARTITION=17,17,14 EXTRA_ENV=VLLM_USE_V2_MODEL_RUNNER=1 FD_RDNA2=1`, after
`convert.py` + `quantize_mtp.py`. Acceptance ~2.2–2.4 tokens/step (81% at K=2), greedy
6/8 identical to the V1-recorded baseline (2 prose formatting divergences from V2
numerics; factual spot-checks pass). At 40k context MTP is parity, not a win — the
remaining context-proportional term is the verify passes themselves (~13 ms/step at 40k,
already near kernel bandwidth); long-context batch workloads can run `MTP=0`.

## Working configuration

`TP=1 PP=3 DEVICES=0,1,3`, an image with all nine patches + the current (GQA-general) fd_rdna2 plugin, `MTP=2 PP_PARTITION=17,17,14` on the V2 runner
(`EXTRA_ENV=VLLM_USE_V2_MODEL_RUNNER=1`), `FD_RDNA2=1 AR_RDNA2=0`, `GPUUTIL=0.95
MAXSEQS=4`, MOE_CFG → the swept fused-MoE config, TunableOp lm_head rows merged (see MTP
section). Fallback config in serve.sh's header comment (v5/V1/MTP=0). Watch item
inherited: flaky load-time worker death (~2-in-7 boots; plain retry works).

## Four cards: flat TP=4 — the current flagship (2026-08-25)

`./serve-4gpu.sh` — 122B across four V620s as plain tensor parallelism:

| ctx | decode t/s | fresh prefill t/s | fresh TTFT s |
|---|---|---|---|
| ~3.5k | 56–59 | 820–838 | ~4.2 |
| ~13k | 57–59 | — | — |
| ~41–45k | 49–55 | 536–537 | ~83 |

Prefill measured with UNIQUE prompts and `max_tokens=1`. Benchmark-harness
TTFT is prefix-cache-entangled and must not be used for prefill claims: a
repeated 44.5k prompt "prefills" in 7.9 s ("5,640 t/s") — pure cache
artifact; fresh is 83 s.

MTP=2, acceptance ~2.3 tokens/step; +43–124% over PP=3 and near-flat with
context (attention shards 4-way, collapsing the context-proportional term).
Cards draw ~180 W (bandwidth-bound) — cool and quiet at full speed.
Validated warm, cold-cache, and through a 3-pass soak: ~2.5 h sustained,
zero GPU events.

**Prerequisites are non-optional.** Every earlier TP attempt on this class
of platform dropped cards off the PCIe bus. The script header lists the
required kernel line, per-boot power caps, and runtime env; the rationale
lives in TROUBLESHOOTING.md §4 and 02-VERSIONS.md's platform-stability
table. Apply the whole stack — the load-bearing subset has not been
isolated.

**⚠ Tuning pending — decode numbers are a floor; prefill currently
trails PP=3 at long context** (537 vs the PP=3 band's upper 763 — suspects:
`--max-num-batched-tokens 2048` is a *stability* choice where the 27B
prefill sweep favored 8192, so any raise must be re-soaked; per-chunk
4-way all-reduces; untuned N=256 MoE prefill configs). TP=4 shards change every
tuned shape and none have been re-swept: the fused-MoE config here covers
`N=1024` (PP shapes; TP=4 needs an `N=256` sweep — use `moe_sweep.py`),
and the lm_head TunableOp rows cover the full vocab GEMM (TP=4 needs
per-rank `tn_37984_*` rows; same offline method as above). Both are open;
expect more once they land.
