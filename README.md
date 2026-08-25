# vLLM + Radeon PRO V620 Linux Recipe

**Copyright © 2026 Aron Hsiao. Licensed under the GNU General Public License v3.0 or later.**

This is AI slop.

But hopefully the good kind. I bought 4x Radeon Pro v620 cards because they're dirt cheap and pretty
performant for the price being 32GB PCIe x4 cards, but I was perennially unsatisfied with llama.cpp's cache handling
for agentic work, especially for models that push up against the cards' practically productive limits.

Trying to deploy vLLM by hand didn't work out well and consulting with LLMs I got the answer that
RDNA2/NAVI21 was unsupported and unlikely to ever be supported; there is no vendor-supported path and
no vendor ships gfx1030 kernels in any vLLM wheel or image.

- There is another project, [vllm-rocm-windows-rdna2](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2), but I'm on Linux, not Windows
- There is also [llama.cpp](https://github.com/ggml-org/llama.cpp), which supports these cards in a relatively performant way

So I pointed Qwen 3.8 Max and Opus 5.0 at both and said let's build out support and optimize it.
This is the result. First actual generatable run (after LLMs consumed the above repos) netted 11-18 t/s generation with a Qwen 27b Q4 quant.
After all optimizations, I got the the 2x v620 combo running Qwen 3.8 27b up to 42 t/s at 41k context.

Since then, I also discovered that:
there is another vLLM for RDNA2 project at:

- There is another vLLM for RDNA2 project at [vLLM RDNA2_extras](https://github.com/blivioniag/vllm/tree/rdna2_extras), and it is a bit more formal than this one is, though there are some areas where we don't overlap yet (i.e. MoE)

Since the start of his in mid-August, I've built out optimization examples for more models, so now we have GPTQ, AWQ, and Autoround, all Int4, and I'll
probably do a round of 8s before all is said and done just so we have a full set of examples. Note that most FP formats and boutique
APU formats do not perform well, and in some cases perform catastrophically.

Each optimization/configuration run for a new model takes 2-3 hours for the agent to complete, but is generally a hands-off
process. Hopefully others who try this will have the same experience.

At this point I have done a number of hours of real work powered by vLLM and all seems well. YMMV.

## What's in this repo really, and how to use it

This is **not** a fork, a distribution, or an installable package. It is a **recipe book**: two standalone plugins that
are shared for v620 model use, and then for each optimized model (see 'builds/'), needed patches or configuration items to
enable support and/or optimize performance when applied against pristine vLLM 0.27.1, along with:

- Achieved performance data on my system: two-card (TP=2) configs for the 27B builds, and three-card (PP=3) and **four-card (TP=4, the current flagship: 56–59 t/s)** configs for the 122B MoE
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

When I started this I thought nobody was doing RDNA2 inference seriously. That was wrong — there's
an active community. See [RDNA2-RESOURCES.md](RDNA2-RESOURCES.md) for links.

## Read in this order

| | |
|---|---|
| **[00-HARDWARE.md](00-HARDWARE.md)** | The machine, the model, the measured silicon facts, and the performance targets. **Start here — the numbers are meaningless without it.** |
| **[01-PATCHES.md](01-PATCHES.md)** | What each patch does and why, how to apply and verify, then the pitfalls and the measured dead ends. |
| **[02-VERSIONS.md](02-VERSIONS.md)** | Exact pins, build order, every runtime setting and what it's worth, and what to do when the pins don't hold. |
| **[ADAPTING-PROCESS.md](ADAPTING-PROCESS.md)** | **Instructions for adapting this recipe** — the optimisation loop as working directions for an ingesting LLM (and its human), with PROFILE-NAVI21.md as the rosetta stone. Read when your model, card, or vLLM version differs from ours. |
| **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** | Failure modes whose error messages lie: mixed-GPU enumeration aborts, wrong device identity, phantom OOMs during graph capture, and cards dropping off the bus — each symptom-first, with the diagnostic ladder that actually discriminates. Read when something impossible is happening. |
| **[PROFILE-NAVI21.md](PROFILE-NAVI21.md)** | Emergency break-glass option if success is elusive. The full silicon profile behind 00-HARDWARE's summary: 37 measured tests covering compute, memory, interconnect, and the design rules they imply. Don't pay attention to too many of the wild-eyed WAGs in it, as it hasn't really been cleaned up, but there's a lot of basic reference here resulting from empirical poking and prodding of the chip. |
| **[CHANGELOG.md](CHANGELOG.md)** | What changed when, newest first — one entry per campaign, with the reasoning and the numbers that moved. Read to see how the recipe got here, or to find when a given patch or finding landed. |
| **[RDNA2-RESOURCES.md](RDNA2-RESOURCES.md)** | The rest of the gfx1030/RDNA2 world: the community wiki, other vLLM and llama.cpp forks, prebuilt images, V620 power/P2P tooling, upstream links. Read if this recipe isn't the right shape for your problem — someone else's project may be. |

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

Moved to [CHANGELOG.md](CHANGELOG.md) — newest first, one entry per campaign.

## Licence and provenance

Our work here is **GPL-3.0-or-later** — so that improvements stay shareable.

The patch files necessarily contain small excerpts of vLLM source, which is Apache-2.0. vLLM's code
remains under its own licence; **this repository contains no vLLM source tree**, only diffs against
it and our own new files. Patch 0001 is adapted from upstream vLLM PR #52391 (gfx10x recognition),
with thanks to its author, who benchmarks on the same v620 hardware.
