# Changelog

**Copyright © 2026 Aron Hsiao. Licensed under the GNU General Public License v3.0 or later.**

Newest first. Companion to [README.md](README.md); per-model detail lives in each
`builds/*/BUILD.md`, and measured configs in [02-VERSIONS.md](02-VERSIONS.md).

**2026-08-25 — four cards, TP=4: 56–59 t/s on the 122B (+43–124% over PP=3).** The
config every earlier attempt said was impossible: flat tensor-parallel across four V620s,
MTP=2, near-FLAT decode from 3.8k to 41k (attention sharded 4-way collapses the context
term), cards bandwidth-bound at ~180 W — cool and quiet at full speed. Fresh prefill
830/537 t/s at 3.5k/45k — long-context prefill trails PP=3 pending the TP-shape tuning
pass (see the build's 4-GPU section; and never trust prefix-cached TTFT for prefill claims).
What changed: the platform-personality mitigation stack — kernel line
(`amdgpu.pcie_gen_cap=0x00070007` Gen3 link cap, `aspm=0`, `runpm=0`, `gpu_recovery=1`),
`HSA_NO_SCRATCH_RECLAIM=1`, `NCCL_P2P_LEVEL=PXB`, and `--max-num-batched-tokens 2048`
(batch size is a TIMING knob on graphics silicon: it sets unpreemptible dispatch length,
scratch-crossing odds, DMA burst duration, and power-ramp width all at once). Validated
warm, cold-cache, and soaked: ~2.5 h sustained, zero gpu events. Details and prerequisites
in the 122B build's 4-GPU section; updated TROUBLESHOOTING §4. **Still untuned for TP=4
shapes** (MoE N=256 sweep + lm_head TunableOp rows pending) — these numbers are a floor.
Credit where due: `HSA_NO_SCRATCH_RECLAIM` and the P2P level came from strip-mining
[edwinbrowwn/llama.cpp-rdna2](https://github.com/edwinbrowwn/llama.cpp-rdna2), a parallel
4×V620 effort on the llama.cpp side.

**2026-08-24 — the Polaris day: three ways an unsupported GPU poisons a supported fleet.**
A pre-Vega display card broke ROCm enumeration for every healthy card (all-or-nothing agent
discovery), contaminated device identity through amdsmi (which ignores visibility filters),
and — beyond any userspace fix — its kernel-side KFD node poisoned VMM/graph-capture memory
paths, producing arithmetically impossible "out of memory" errors on 64-byte allocations.
Card removed; every diagnostic and the one-byte vendor-binary doorbell patch documented in
the new **TROUBLESHOOTING.md** — failure modes that lie to you, symptom-first.

**2026-08-23 (later still) — MTP pays: 39.7–41 t/s on the 122B.** Two finishing moves on
top of patch 0009: `fd_rdna2` generalized from the 27B's GQA 6 to any GQA ≤ 16 (the Triton
kernels were already parameterized; the wrapper wasn't), and offline TunableOp rows for the
fp16 lm_head shapes (the ROCm skinny-gemm fast path is CDNA-gated; the default Tensile pick
cost 10.6 ms per verification step). 122B decode with MTP=2: 39.7–41 / 33.4 / 21.9 t/s at
3.5k/13k/40k — vLLM now beats llama.cpp on this model on these cards. Verification harness
for the plugin geometries in `verify/fd_gqa_test.py`.

**2026-08-23 (later) — patch 0009: MTP under pipeline parallelism (V2 runner).** Backport
of upstream PR #46994 plus our own V1-runner guards. The finding that matters: MTP under PP
requires `VLLM_USE_V2_MODEL_RUNNER=1` (the V1 drafter path page-faults under PP — upstream
never tested it), and the V2 runner measures identical to V1 on gfx1030 at MTP=0, so the
switch is free. On the 122B, speculation is *correct* (81% acceptance at K=2, draft head
RTN-packed to int4 by `quantize_mtp.py`) but currently a decode regression — verification
attention (q_len=3) rides the context-proportional prefill path; production stays MTP=0
until a batched-MQ kernel covers those shapes. Launcher gains `PP_PARTITION`, `ASYNC_SCHED`,
`SPEC_EAGER`, `EXTRA_ENV`.

**2026-08-23 (night) — patch 0008: the MoE decode skinny GEMV.** The kernel the 122B was
waiting for: wave-per-row expert GEMV at 432 GB/s effective (the tile-based MoE kernels
stream weights across the wrong axis and sit 10×+ off bandwidth at batch-1). Decode on the
Intel 122B: 15.6 → **26.9 t/s** — short-context parity with llama.cpp on the same three
cards, and the day's cumulative on this model is 7.1 → 26.9. The launcher also gains
`EXTRA_MOUNT` (comma-separated bind overlays) — born as a debugging tool for the
twin-apply-method pitfall documented in 01-PATCHES, kept because rebuild-free iteration on
any baked file is generally useful.


**2026-08-23 (evening) — 122B build replaced: backbone-quantized checkpoint doubles decode.**
`builds/Intel-Qwen3.5-122B-A10B-int4-AutoRound/` supersedes the official-GPTQ build. The
lesson that matters: the official Qwen GPTQ quantizes only the *experts*, leaving ~18 GB of
bf16 backbone streaming every token (extra-painful on gfx1030, which has no native bf16).
Intel's AutoRound covers the backbone (shared expert fp16, MTP untouched) — verified from
config + tensor-index metadata alone, before downloading a byte. Decode 7.1 → **15.7 t/s**,
same serving config. Read a MoE checkpoint's exclusion map before trusting its name.


**2026-08-23 (later) — patch 0007: the CUDA moe_wna16 MoE kernel ported to gfx1030.**
An afternoon-scale port (portable lop3/prmt/bf16 substitutions, CAS fp16 atomicAdd, four
un-gating layers). Correct in-server; ~1.1× the Triton MoE path at decode sizes. Its main
value was diagnostic: on the official 122B quant, decode splits roughly evenly between MoE
kernels, the checkpoint's unquantized bf16 backbone, and PP overhead — so this patch alone
does not move end-to-end numbers, and the 122B build's BUILD.md now carries the corrected
token budget. Lesson shipped with it: benchmark MoE kernels at the batch size the server
actually runs (M = tokens × top-k at decode is still tiny), not a convenient synthetic M.


**2026-08-23 — first model beyond two cards: Qwen3.5-122B-A10B at PP=3.**
`builds/Qwen-Qwen3.5-122B-A10B-GPTQ-Int4/` — the official GPTQ 122B MoE across three V620s
via pipeline parallelism (TP=3 is arithmetically impossible: 2 KV heads). New launcher
knobs: `PP` (pipeline stages) and `MOE_CFG` (mounts a tuned fused-MoE config JSON; the
build dir carries one swept offline on gfx1030, worth +75% prefill). Working and validated:
prefill 558–721 t/s; decode 7.1 t/s, bound by vLLM's Triton fused-MoE overhead at batch-1
(~45× off bandwidth — the CUDA moe_wna16 kernel that handles this regime is is_cuda()-gated
out of ROCm builds; porting it is the open campaign). Read the BUILD.md's quantization
section before extrapolating: only the experts are int4, the bf16 backbone caps decode
at ~28 t/s.


**2026-08-22 (later) — third build: AutoRound mixed-precision.**
`builds/Pilcothink-Qwen3.8-27B-MixedInt4-AutoRound/` — W4 g32 symmetric with int8 on 17
sensitivity-selected projections. Needs two scripted one-time checkpoint conversions (in the
build dir): vLLM has no "auto-round" quant method, so the config is rewritten to GPTQ with
the mixed-bits table as `dynamic` overrides, and the int4 MTP head is dequantized to dense
(vLLM builds MTP predictors unquantized). Decode 47.0/49.8/41.6 t/s, prefill 856/827/670 —
reference parity or better. A worked example of adapting a checkpoint whose *metadata*, not
tensors, is the incompatibility.


**2026-08-22 — per-model builds/ tree; second model (AWQ) at full speed; patch 0006.**
- The repo now mirrors our working layout: `builds/<model-id>/` holds a ready-made optimized
  configuration per model (`BUILD.md` + `serve.sh`); plugins moved to `builds/shared/plugins/`.
- **`cyankiwi/Qwen3.8-27B-AWQ-INT4`** (compressed-tensors, *asymmetric* W4 g32 — same bytes as
  the GPTQ reference) is brought up at parity or better: decode 49.8/49.0/43.9 t/s at
  3.8k/15k/41k, prefill 753/736/524 at 3.8k/7.5k/35k. Asymmetric uint4 is outside Exllama's
  types, which forces a different kernel route entirely — new **patch 0006** extends vLLM's HIP
  skinny int4 GEMV from its gfx11/gfx12 guard down to gfx1030 (wave32/LDS/v_dot2 envelope is
  identical) and adds a dequant-to-dense + rocBLAS prefill route. Two transferable lessons in
  01-PATCHES: check arch guards before assuming a "gfx11+" kernel is off-limits, and a fused-
  dequant Triton GEMM loses ~3× to dequantize-then-rocBLAS at prefill shapes.
- The shared launcher gained a `DISABLED_KERNELS` override so per-model builds can pick their
  weights kernel.

**2026-08-21 (later) — prefill +35% at long context.** Patch 0002 now also sets
`num_stages=1` for the gfx10x attention branch: pipelining was halving occupancy at
head_dim 256. Prefill 834/747/521 tok/s at 3.5k/15k/37k (was 816/608/386), outputs
byte-identical. The wider-query-tile route (upstream's Blackwell fix) was measured and is
*worse* on this chip — see 01-PATCHES.

**2026-08-21 — MTP enabled; prefill stall fixed.**
- `MTP=2` (the checkpoint's own multi-token-prediction head, `qwen3_5_mtp`) is now the default:
  **41.4 t/s @41k (+24%), 50.4 @14k (+36%)**, output-lossless. This required extending the
  `fd_rdna2` plugin to **batched multi-query verification** — without it, verification passes
  silently fall to the stock attention kernel and MTP measures as a **3.6× regression**. The
  plugin and MTP must be deployed together.
- **TunableOp now runs lookup-only** (`PYTORCH_TUNABLEOP_TUNING=0`). Tuning mode autotunes every
  never-seen GEMM shape mid-request, and prefill M is prompt-length-dependent — fresh prompts
  paid minutes-long stalls (bimodal 771 vs 37 tok/s at identical sizes). See pitfalls in 01.
- `verify/decode-rate.py` and `verify/prefill-rate.py` added; `longctx-decode.py` removed (its
  median-gap method mis-measures under speculative decoding).

**2026-08-21 (later still) — prefill campaign closed at the practical ceiling.** Final:
834/747/521 tok/s at 3.5k/15k/37k. Verified optimal and now documented as measured dead ends
(see 01-PATCHES): two custom prefill-attention kernel structures (0.70× and 0.58× — at prefill,
KV is cache-served and the stock kernel wins), chunk sizes other than 8192, and wider query
tiles. The reusable rule that came out of it: **pass `num_stages=1` in every Triton kernel on
this chip.** Full patch series re-verified: applied to pristine vLLM 0.27.1 it reproduces the
running tree exactly.

**2026-08-20 — initial publication.** Five patches, two plugins, 33.5 t/s @42k, no speculation.
