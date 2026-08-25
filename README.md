# vLLM + Radeon PRO V620 Linux Recipe

**Copyright © 2026 Aron Hsiao. Licensed under the GNU General Public License v3.0 or later.**

This is AI slop.

But hopefully the good kind. I bought 4x Radeon Pro v620 cards because they're dirt cheap and pretty
performant for the price being 32GB PCIe x4 cards, but I was perennially unsatisfied with llama.cpp's cache handling
for agentic work, especially for models that push up against the cards' practically productive limits.

Trying to deploy vLLM by hand didn't work out well and consulting with LLMs I got the answer that
RDNA2/NAVI21 was unsupported and unlikely to ever be supported; there is no vendor-supported path and
no vendor ships gfx1030 kernels in any vLLM wheel or image.

- There is another project, [https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2), but I'm on Linux, not Windows
- There is also llama.cpp, which supports these cards in a relatively performant way

So I pointed Qwen 3.8 Max and Opus 5.0 at both and said let's build out support and optimize it.
This is the result. First actual generatable run (after LLMs consumed the above repos) netted 11-18 t/s generation with a Qwen 27b Q4 quant.
After all optimizations, I got the the 2x v620 combo running Qwen 3.8 27b up to 42 t/s at 41k context.

Since then I've built out optimization examples for more models, so now we have GPTQ, AWQ, and Autoround, all Int4, and I'll
probably do a round of 8s before all is said and done just so we have a full set of examples. Note that most FP formats and boutique
APU formats do not perform well, and in some cases perform catastrophically.

Each optimization/configuration run for a new model takes 2-3 hours for the agent to complete, but is generally a hands-off
process. Hopefully others who try this will have the same experience.

At this point I have done a number of hours of real work powered by vLLM and all seems well. YMMV.

## What's in this repo really, and how to use it

This is **not** a fork, a distribution, or an installable package. It is a **recipe book**: two standalone plugins that
are shared for v620 model use, and then for each optimized model (see 'builds/'), needed patches or configuration items to
enable support and/or optimize performance when applied against pristine vLLM 0.27.1, along with:

- Achieved performance data on my system: two-card (TP=2) configs for the 27B builds, and a three-card (PP=3) config for the 122B MoE
- Summaries of what was done foreach model

This project is deliberately shaped to be handed to your LLM along with instructions to consume and then build and/or optimize your
compatible but not yet supported model of choice on v620 hardware. This repo should contain enough detail for a good model+harness to
reconstruct a working deployment: version pins, patches with stated *intent* (so they can be re-derived when they no longer apply cleanly),
the traps that look like other problems, and lightweight tools and testing/optimizing loop that a model can execute to get to a working
configuration for your model. Hopefully. :-D

Used in my case with:

- vLLM 0.27.1
- ROCm 7.2.3
- AMD clang 22.0.0git
- PyTorch 2.11.0+gitd0c8b1f
- Ubuntu LTS 26.04
- Qwen 3.8 Max / Claude Opus 5.0

## Read in this order

| | |
|---|---|
| **[00-HARDWARE.md](00-HARDWARE.md)** | The machine, the model, the measured silicon facts, and the performance targets. **Start here — the numbers are meaningless without it.** |
| **[01-PATCHES.md](01-PATCHES.md)** | What each patch does and why, how to apply and verify, then the pitfalls and the measured dead ends. |
| **[02-VERSIONS.md](02-VERSIONS.md)** | Exact pins, build order, every runtime setting and what it's worth, and what to do when the pins don't hold. |
| **[ADAPTING-PROCESS.md](ADAPTING-PROCESS.md)** | **Instructions for adapting this recipe** — the optimisation loop as working directions for an ingesting LLM (and its human), with PROFILE-NAVI21.md as the rosetta stone. Read when your model, card, or vLLM version differs from ours. |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Failure modes whose error messages lie: mixed-GPU enumeration aborts, wrong device identity, phantom OOMs during graph capture, and cards dropping off the bus — each symptom-first, with the diagnostic ladder that actually discriminates. Read when something impossible is happening. |
| **[PROFILE-NAVI21.md](PROFILE-NAVI21.md)** | Emergency break-glass option if success is elusive. The full silicon profile behind 00-HARDWARE's summary: 37 measured tests covering compute, memory, interconnect, and the design rules they imply. Don't pay attention to too many of the wild-eyed WAGs in it, as it hasn't really been cleaned up, but there's a lot of basic reference here resulting from empirical poking and prodding of the chip. |

```
patches/    nine patches against pristine vLLM 0.27.1
builds/     one directory per model we've actually brought up and optimized,
            named for the Hugging Face model id — each contains BUILD.md
            (the document of record: the model's structure, its quantization
            details, the measured numbers, and the *why* behind every
            configuration choice) and serve.sh (a ready-to-run launcher
            wrapper with that model's settings)
builds/shared/plugins/
            two net-new pip-installable packages (attention kernel, all-reduce)
config/     the shared serving launcher with every tuned default
verify/     correctness, performance and soak checks with expected results
tools/      operational conveniences: watch.py, a llama-server-style live
            monitor (per-second PREFILL/DECODE state, rates, cache hits)
```

**The `builds/` tree is the fastest path if your model is already in it.** Each directory is a
known-good, already-optimized configuration: read its `BUILD.md` for what to expect and run its
`serve.sh` to get it. The directory name is the model id with `/` replaced by `-`, so
`builds/cyankiwi-Qwen3.8-27B-AWQ-INT4/` serves `cyankiwi/Qwen3.8-27B-AWQ-INT4`. When you bring
up a model that isn't there, the process in ADAPTING-PROCESS.md is the path — and the convention
is to start by copying the closest existing build and adapting it.

## Scope — Radeon Pro v620 (narrowly)

**This is entirely about the Radeon PRO V620 (Navi 21, gfx1030).** Nothing else was tested or considered.
A bunch of work was done to profile the v620 hardware in detailed fashion and implement a working solution;
some or all of the optimizations are likely to be v620-specific.

Some of it is also architectural and may apply to other gfx1030 cards — the 64 KiB LDS fix, the
gfx10x platform recognition, the attention kernel. gfx1031 and below (RX 6700 XT and friends) are
untested and differ in CU count and cache.

If you try this on other RDNA2 hardware we'd genuinely like to hear how far you get, and see resharing back out to the world.
Also, feel free to take, fork, and rebake. Would love to see someone implement something more formal, more easily usable,
and more distributable than this for the RDNA2 family cards (hell I would use it) but in the meantime here we are.

## What is honestly not solved

- **The base image build is unverified end-to-end.** We identified the recipe (vLLM's own
  `rocm_base` plus one arch line) and built it as we went, retrospectively after each successful patch; nobody has yet run this recipe from scratch.
- **The two plugins monkey-patch vLLM internals** and will need rework on any version bump — and
  they fail *silently*. See the pitfalls in 01-PATCHES.md.
- **Hard host crashes under kernel-launch churn — largely tamed 2026-08-23.** The machine
  used to spontaneously freeze (no kernel trace) under launch storms: TunableOp tuning
  churn, and later, cold Triton/inductor compiles across 3 GPUs — dramatically amplified by
  `AMD_SERIALIZE_KERNEL/HIP_LAUNCH_BLOCKING` (never set those on multi-GPU cold-compile
  boots). Two kernel-line changes stopped the crashes here:
  `amdgpu.ppfeaturemask=0xfff77fff` (driver default + OverDrive for fan control, with
  GFXOFF and GFX_DCS off — the idle↔burst power-state churn suspects) and
  `amdgpu.gpu_recovery=0` (reset attempts with busy P2P peers wedged the whole box;
  recovery off converts hangs into a dead process instead of a dead machine). A crash
  mid-compile can leave a truncated artifact in the compile cache
  (`module ... has no attribute 'triton_poi_...'`) — wipe the cache directory.
- **MTP on the 122B is a win only up to ~mid context.** At ~40k it measures parity with
  MTP=0 (each verification re-reads the KV history once per draft position; the passes are
  already near kernel bandwidth), and time-to-first-token with MTP on is not yet honestly
  measured (the drafter re-processes prompt chunks to prime itself). Both are open items in
  the Intel build's BUILD.md.
- **Multi-stream decode on the AWQ/asymmetric path is unmeasured.** The HIP skinny GEMV that
  gives the hybrid kernel its decode speed covers batches of ≤5 rows; concurrent MTP streams
  exceed that and fall to a slower path. Single-stream numbers are solid; batch serving on that
  build needs measuring first.

## Changelog

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

## Licence and provenance

Our work here is **GPL-3.0-or-later** — so that improvements stay shareable.

The patch files necessarily contain small excerpts of vLLM source, which is Apache-2.0. vLLM's code
remains under its own licence; **this repository contains no vLLM source tree**, only diffs against
it and our own new files. Patch 0001 is adapted from upstream vLLM PR #52391 (gfx10x recognition),
with thanks to its author, who benchmarks on the same v620 hardware.
