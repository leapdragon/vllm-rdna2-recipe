# vLLM + Radeon PRO V620 Linux Recipe

**Copyright © 2026 Aron Hsiao. Licensed under the GNU General Public License v3.0 or later.**

This is AI slop.

But hopefully the good kind. I bought 4x Radeon Pro v620 cards because they're dirt cheap and pretty
performant for 32GB PCIe x4 cards, but I was perennially unsatisfied with llama.cpp's cache handling
for agentic work, especially for models that push up against the cards' practically productive limits.

Trying to deploy vLLM by hand didn't work out well and consulting with LLMs I got the ansewr that
RDNA2/NAVI21 was unsupported and unlikely to ever be supported; there is no vendor-supported path and
no vendor ships gfx1030 kernels in any vLLM wheel or image.

- There is another project, [https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2), but I'm on Linux, not Windows
- There is also llama.cpp, which supports these cards in a relatively performant way

So I pointed Qwen 3.8 Max and Opus 5.0 at both and said let's build out support and optimize it.
This is the result. First actual generatable run (after LLMs consumed the above repos) netted 11-18 t/s generation with Qwen 27b. After all optimizations, the 2x v620
combo running Qwen 3.8 27b is steady at &gt;32-34 t/s *before MTP* with the expected speed cost as context accumulates.

We have not yet enabled out MTP, so I presume we'll be in the 40 t/s range once we do that. YMMV. 

## What this is, and how to use it

This is **not** a fork, a distribution, or an installable package. It is a **recipe**: five small
patches against pristine vLLM 0.27.1, two standalone plugin packages, a serving configuration, and
— most importantly — **a written record of what was measured, what was implemented, and what's
not worth pursuing.**

It is deliberately shaped to be handed to your LLM along with instructions to consume and then build similar.
This repo should contain enough detail for a good LLM harness to reconstruct a working deployment: version pins, patches with stated
*intent* (so they can be re-derived when they no longer apply cleanly), the traps that look like
other problems, and falsifiable targets so you can tell whether you actually arrived.

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

```
patches/    five patches, 182 lines, against pristine vLLM 0.27.1
plugins/    two net-new pip-installable packages (attention kernel, all-reduce)
config/     the serving launcher with every tuned default
verify/     correctness, performance and soak checks with expected results
```

## Scope — Radeon Pro v620 (narrowly)

**This is entirely about the Radeon PRO V620 (Navi 21, gfx1030).** Nothing else was tested or considered.
A bunch of work was done to profile the v620 hardware in detailed fashion and implement a working solution;
some or all of the optimizations are likely to be v620-specific.

Some of it is also architectural and should may to other gfx1030 cards — the 64 KiB LDS fix, the
gfx10x platform recognition, the attention kernel. gfx1031 and below (RX 6700 XT and friends) are
untested and differ in CU count and cache.

If you try this on other RDNA2 hardware we'd genuinely like to hear how far you get, and see resahring back out to the world.
Also, feel free to take, fork, and rebake. Would love to see someone implement something more formal, more easily usable,
and more distributable than this for the RDNA2 family cards, but in the meantime here we are.

## What is honestly not solved

- **The base image build is unverified end-to-end.** We identified the recipe (vLLM's own
  `rocm_base` plus one arch line) and built it as we went, retrospectively after each successful patch; nobody has yet run this recipe from scratch.
- **The two plugins monkey-patch vLLM internals** and will need rework on any version bump — and
  they fail *silently*. See the pitfalls in 01-PATCHES.md.
- **Speculative decoding (MTP) is untouched**, and is the largest remaining lever by some distance.

## Licence and provenance

Our work here is **GPL-3.0-or-later** — so that improvements stay shareable.

The patch files necessarily contain small excerpts of vLLM source, which is Apache-2.0. vLLM's code
remains under its own licence; **this repository contains no vLLM source tree**, only diffs against
it and our own new files. Patch 0001 is adapted from upstream vLLM PR #52391 (gfx10x recognition),
with thanks to its author, who benchmarks on the same v620 hardware.
