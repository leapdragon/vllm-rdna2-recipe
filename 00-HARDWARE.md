# 00 — The hardware and model these numbers describe

Copyright © 2026 Aron Hsiao · GPL-3.0-or-later (see LICENSE)

**Every performance figure in this repo is meaningless without this page.** A token rate is a
property of a specific card, in a specific slot, running a specific quantisation of a specific
model. Change any of them and the targets no longer apply.

## The machine

| | |
|---|---|
| Platform | **AMD X399** (Threadripper 2950X, 16C/32T), 128 GB DDR4 |
| GPUs | 3 × **Radeon PRO V620** (Navi 21, gfx1030, 32 GB) + 1 × RX 6700 XT (gfx1031, display only) |
| Slots used for inference | **two ×16-rooted V620s** — this matters (see below) |
| Host OS | Ubuntu 26.04, kernel 7.0 |
| Card power cap | 232 W (lowered from 250 W; costs nothing — decode is bandwidth-bound) |

**Slot width is load-bearing.** The two V620s used for tensor parallelism sit in full-bandwidth
×16 slots. Measured peer-to-peer bandwidth is **14.04 GB/s** between them, versus **7.0 GB/s** for
an ×8-rooted pair. If your cards are in ×8 slots, expect the communication numbers to halve.

## The measured silicon facts that drive every design decision

Navi 21 / gfx1030, measured on this hardware — not vendor specifications:

| Property | Measured | Consequence |
|---|---|---|
| **Achievable DRAM bandwidth** | **506 GB/s** | The denominator for everything. Decode is bandwidth-bound. |
| Matrix units | **none** | No WMMA/MFMA. But the full dot-product family is present (`v_dot4_i32_i8`, `v_dot2_f32_f16`, `v_dot8_i32_i4`), and Triton's `tl.dot` reaches them. |
| LDS per workgroup | **64 KiB hard cap** | Upstream attention kernels overflow it at `head_dim ≥ 256`. See patch 0002. |
| Wavefront | wave32 (36 WGPs / 72 CUs) | |
| Infinity Cache | 128 MB | **Cannot be recruited for KV** — a 43k-token context exceeds it. Benchmarks that fit in it lie. |
| Peer-to-peer | **real**, kernel-level | Push (peer store) **14.30 GB/s** vs pull (peer load) **5.70 GB/s** — any custom collective must be push-based. |
| Cross-device coherence | **absent** | Coarse-grained memory (plain `hipMalloc`) is invisible to a peer mid-kernel. See 01-PATCHES pitfalls. |

## The model

**`btbtyler09/Qwen3.8-27B-GPTQ-4bit`** — the targets below are for this quant specifically.
(A second quant of the same model, `cyankiwi/Qwen3.8-27B-AWQ-INT4` — asymmetric W4 at identical
bytes, which forces a different kernel route via patch 0006 — is brought up at parity or better;
see `builds/cyankiwi-Qwen3.8-27B-AWQ-INT4/BUILD.md` for its numbers.)

| | |
|---|---|
| Quantisation | **GPTQ 4-bit**, `group_size=32`, symmetric, `desc_act=False` |
| Architecture | Qwen3.5/3.8 hybrid — **48 GatedDeltaNet (linear attention) + 16 full-attention layers** (`full_attention_interval=4`), 64 total |
| Hidden / intermediate | 5120 / 17408 |
| Heads | 24 query / 4 KV, **`head_dim=256`** ← this is why patch 0002 exists |
| Vocab | 248,320, **untied** `lm_head` (fp16, 2.37 GB, streamed every token) |
| Bytes streamed per token | **17.175 GB** whole-model — **8.588 GB per card at TP=2** |
| KV cache | `int8_per_token_head`, 520-byte entry (K 0..255, kscale 256..259, V 260..515, vscale 516..519) |

**The floor this implies:** 8.588 GB per card ÷ 506 GB/s = **18.22 ms/token = 54.9 t/s**. Nothing
short of a different quantisation moves it. We reach 33.5 t/s, i.e. ~61% of the absolute ceiling.

## The targets

TP=2 on the two ×16 V620s, int8 KV, `max_model_len=131072`, single stream, greedy:

| Context | Stock vLLM + gfx1030 patches | Full set, `MTP=0` | **Full set with `MTP=2`** |
|---:|---:|---:|---:|
| ~3.5k | 27.8 t/s | 37.4 t/s | **43–51 t/s** |
| ~14k | 24.5 t/s | 36.9 t/s | **50.4 t/s** |
| ~41k | 18.4 t/s | 33.5 t/s | **41.4 t/s** |

MTP figures move with draft acceptance (~58% per draft on this quant), so short-context numbers
spread with prompt content. Prefill (cold prompts): **834 tok/s @3.5k, 747 @15k, 521 @37k**.

Context slope 0.4879 → **0.0805 µs/ctx-token** (6.06× flatter).

For scale: llama.cpp on the same model and machine reaches 31–40 t/s at 45k with 2-draft MTP.
The `MTP=2` column uses the same technique (the checkpoint's own MTP head) and passes that band.

These targets are the two-card 27B reference. The four-card 122B TP=4 flagship and three-card PP=3 build (MTP)
has its own table in `builds/Intel-Qwen3.5-122B-A10B-int4-AutoRound/BUILD.md` —
headline 39.7–41 t/s at 3.5k, also past llama.cpp on the same model.

## Scope — read this before assuming it transfers

This work targets the **Radeon PRO V620 (Navi 21, gfx1030)** and nothing else. We did not test on
any other RDNA2 part and make no claim it works there.

Some of it should generalise to other gfx1030 cards (W6800, W6900X, RX 6800/6900 XT): the LDS tile
fix, the gfx10x platform recognition, and the attention kernel are architectural. The *numbers*
will not transfer — they depend on 32 GB of VRAM, 506 GB/s, and two ×16 slots. gfx1031/gfx1032
(RX 6700 XT and below) are untested and have different CU counts and cache sizes.

If you try it on other RDNA2 hardware, we would be glad to hear how far you get.
