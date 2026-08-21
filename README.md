# vLLM on the Radeon PRO V620 — a working recipe

**Copyright © 2026 Aron Hsiao. Licensed under the GNU General Public License v3.0 or later.**

Running vLLM productively on Radeon PRO V620 cards is possible. Every LLM we asked said it wasn't,
and there is no vendor-supported path: AMD ships no gfx1030 kernels in any vLLM wheel or image.
This repository contributes the profiling and testing we did to get there, so that other V620
owners don't have to repeat it.

**On this hardware, with the model below, it gets you from 18.4 to 33.5 tokens/sec at 42k context**
— and from "vLLM does not run at all" to "vLLM runs."

## What this is, and how to use it

This is **not** a fork, a distribution, or a package you install. It is a **recipe**: five small
patches against pristine vLLM 0.27.1, two standalone plugin packages, a serving configuration, and
— most importantly — **a written record of what we measured, what we got wrong, and which promising
ideas turned out to be dead ends.**

It is deliberately shaped to be handed to a capable LLM coding partner. Point one at this repo and
it should have enough to reconstruct a working deployment: exact version pins, patches with stated
*intent* (so they can be re-derived when they no longer apply cleanly), the traps that look like
other problems, and falsifiable targets so you can tell whether you actually arrived.

We think that is a realistic path. We do not think it is a push-button one — you will need a
partner that can compile things, read errors, and measure. But the hard part was never the code;
it was learning which of a hundred plausible ideas were worth trying. That part is written down.

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

## Scope — please read before assuming this applies to you

**This is entirely about the Radeon PRO V620 (Navi 21, gfx1030).** We tested on nothing else and
expect success on nothing else. We make no claim about any other RDNA2 part.

Some of it is architectural and should carry to other gfx1030 cards — the 64 KiB LDS fix, the
gfx10x platform recognition, the attention kernel. The *numbers* will not: they depend on 32 GB of
VRAM, 506 GB/s of measured bandwidth, and two ×16-rooted slots on an AMD X399 platform. gfx1031 and
below (RX 6700 XT and friends) are untested and differ in CU count and cache.

If you try this on other RDNA2 hardware we'd genuinely like to hear how far you get.

## What is honestly not solved

- **The base image build is unverified end-to-end.** We identified the recipe (vLLM's own
  `rocm_base` plus one arch line) after the fact; nobody has yet run it from scratch. See 02.
- **The two plugins monkey-patch vLLM internals** and will need rework on any version bump — and
  they fail *silently*. See the pitfalls in 01.
- **Speculative decoding (MTP) is untouched**, and is the largest remaining lever by some distance.
- The machine has crashed twice under long unattended runs. Current suspicion is a display-GPU
  runtime-PM interaction, not this code, but it is unresolved.

## Licence and provenance

Our work here is **GPL-3.0-or-later** — so that improvements stay shareable.

The patch files necessarily contain small excerpts of vLLM source, which is Apache-2.0. vLLM's code
remains under its own licence; **this repository contains no vLLM source tree**, only diffs against
it and our own new files. Patch 0001 is adapted from upstream vLLM PR #52391 (gfx10x recognition),
with thanks to its author, who benchmarks on the same V620 hardware.
