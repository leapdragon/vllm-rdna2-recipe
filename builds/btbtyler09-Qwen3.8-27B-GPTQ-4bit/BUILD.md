# BUILD — btbtyler09/Qwen3.8-27B-GPTQ-4bit

**Model**: `btbtyler09/Qwen3.8-27B-GPTQ-4bit` — Qwen3.8-27B, GPTQ W4, group_size 32,
**symmetric** (uint4b8). 19.54 GB. Served as `qwen38-27b-gptq`. **The reference build** — the
recipe's patches and tuned defaults were developed against this checkpoint, and the shared
launcher's defaults are its working configuration.

## The model, and why its structure shaped the optimisation

| Fact | Consequence on this platform |
|---|---|
| Hybrid architecture: **48 GatedDeltaNet (linear-attention) + 16 full-attention layers** (`full_attention_interval=4`, 64 total) | Only 16 layers carry a KV cache; the GDN layers are pure GEMM work. Weights bandwidth, not attention, dominates the decode token. |
| **`head_dim=256`**, 24 query / 4 KV heads | The single most consequential number. 256-wide heads overflow gfx1030's 64 KiB LDS under stock Triton tiling (engine death at boot) — the reason patch 0002 and the `num_stages=1` rule exist. |
| Vocab 248,320 with **untied fp16 `lm_head` (2.37 GB)** | Streamed every token; ~28% of the decode byte budget, and the largest untouched lever. |
| Bytes per decode token: **17.175 GB whole-model → 8.588 GB/card at TP=2** | ÷ 506 GB/s measured bandwidth = **18.22 ms = 54.9 t/s hard floor** without speculation. Every decode decision is judged against this. |
| Ships its own **MTP head** (`qwen3_5_mtp`) | Free speculative decoding — but verification carries multiple query tokens, which is why `fd_rdna2` implements batched multi-query attention. Deploy MTP and that plugin together or MTP measures as a large loss (see 01-PATCHES). |

## The quantization, and why it selects the kernel route

GPTQ, W4, group_size 32, **symmetric, `desc_act=False`**. Symmetric W4 loads as `uint4b8`,
which is inside **ExllamaLinearKernel's** supported types — and Exllama's HIP `q_gemm` is the
fastest W4 path measured on gfx1030. This build **forces Exllama** by disabling every
alternative (the launcher's default `VLLM_DISABLED_KERNELS` list); left to itself, vLLM's
dispatch lands on a Triton fallback in the ~4 t/s class. The forcing list doubles as a
tripwire: a checkpoint Exllama *cannot* implement fails loudly at boot instead of silently
serving slow — exactly how the sibling AWQ build's different route was discovered.

Quant-shape consequences: `group_size=32` doubles scale traffic vs g128, but dequant runs at
1.88× the memory ceiling (PROFILE-NAVI21.md §2) — dequant is free, bandwidth is everything.
Exllama's `BLOCK_KN_SIZE` is raised 128 → 256 by patch 0004 (+3.3% @42k): fewer, longer
K-strips suit 72 CUs at these shapes.

## Achieved performance (in-server, MTP=2)

| | 3.5k ctx | 14–15k ctx | 37–41k ctx |
|---|---|---|---|
| **Decode** | 43–51 t/s | 50.4 t/s | 41.4 t/s |
| **Prefill** | 834 t/s | 747 / 608 t/s | 521 t/s |

Decode without MTP: 33.5 t/s @41k (~61% of the bandwidth ceiling). Beats llama.cpp's
31–40 t/s @45k on the same cards. Point estimates; ±5% run-to-run/boot-to-boot spread.

> **2026-08-31 (resolved same day):** these decode numbers stopped reproducing (27–31 t/s, same
> image, same outputs) and a day of elimination — image lineage, power caps, PCIe link width,
> HW queues, P2P latency, kernel cmdline, checkpoint — found the cause in a torch profile: the
> **tuned TunableOp lm_head rows had been lost** (see the section below). With the rows restored
> the table reproduces exactly: 49.0 / 45.3 / 41.8 t/s at 3.5k / 13k / 42k, outputs 8/8 identical.
> The 2026-08-25 platform-stability cmdline was exonerated along the way — it has no measurable
> decode cost.

## TunableOp lm_head rows — load-bearing, shipped in `tunableop/` here (2026-08-31)

The fp16 lm_head (vocab 248,320, TP-sharded to 124,160 per rank) is unquantized, and rocBLAS's
heuristic gives its skinny decode GEMMs (`tn_124160_{1,3}_5120`) a macro-tile meant for large M:
**11.1 ms at 115 GB/s instead of 2.9–3.5 ms at 360–430 GB/s**. MTP=2 pays that three times per
step (verify at M=3, two drafts at M=1), so losing the tuned rows costs **~40% of decode
(49 → 27 t/s)** — context-flat, prefill untouched, outputs byte-identical, so nothing looks
broken. The rows exist only in the gitignored live `tunableop/` directory unless shipped; their
loss burned a full diagnostic day on 2026-08-31 (the 122B build documents the same disease for
its own head — its BUILD.md's offline-probe recipe applies here with `W = (124160, 5120)`).

Restore by copying the shipped per-rank CSVs into the live dir:

```bash
cp builds/btbtyler09-Qwen3.8-27B-GPTQ-4bit/tunableop/tunableop_results{0,1}.csv tunableop/
```

or retune: one boot with `TUNEOP_TUNING=1` (warmup tunes the M=8192 prefill shapes, ~17 min),
a few short greedy requests after `/health` (the M=1/3 head rows only appear under decode
traffic), then a **graceful** stop. The rows carry Validator stamps (torch 2.11.0, ROCm 7.2,
rocBLAS 5.2.0, gfx1030); TunableOp silently ignores a CSV whose validators mismatch — which
re-creates this exact regression — so retune after any stack bump.

## Working configuration, item by item, with the why

- **Image**: patches 0001–0005 applied per 01-PATCHES (0006 is harmless here — it only
  affects kernels this build keeps disabled). No vendor wheel ships gfx1030 kernels.
- **Launch `./serve.sh`** — this model *is* the shared launcher's defaults.
- **`MTP=2`** — the checkpoint's own head; +24% @41k, output-lossless, ~58% acceptance/draft.
  Requires `fd_rdna2`'s multi-query verification path (`FD_MAXQ=4`).
- **`FD_RDNA2=1`** — flash-decode plugin over the int8 KV layout; context slope 6.06× flatter
  than stock. `FD_MAXQ=4` because the batched kernel tile fits 4 query positions.
- **`AR_RDNA2=1`** — push-based TP=2 all-reduce, +1.9%. Adopted because it is free, not
  because comms was large (2.35 ms/token measured in-server).
- **KV `int8_per_token_head`** — halves KV bytes, doubles reachable context; the 520-byte
  entry layout is what `fd_rdna2`'s loads are shaped to.
- **TP=2 on the two x16-rooted cards** — the other slots are x8 on this Threadripper
  platform; all-reduce latency is PCIe-bound.
- **Chunked prefill 8192** (`--max-num-batched-tokens`) — swept; 4096 and 16384 both worse.
- **`GPU_MAX_HW_QUEUES=4`** — 8 was a 32% regression.
- **CompilationMode 3 + FULL_DECODE_ONLY graphs** — decode is graph-captured (launch overhead
  matters at 20 ms/token); prefill is not (shapes vary).
- **TunableOp lookup-only, with the shipped lm_head rows** — tuning mode autotunes
  prompt-length-dependent shapes mid-request (see the pitfall in 01-PATCHES), and the tuned
  rows themselves are LOAD-BEARING: losing them silently costs ~40% of decode (next section).
- **dtype float16** — gfx1030 has no bf16 opcodes.
