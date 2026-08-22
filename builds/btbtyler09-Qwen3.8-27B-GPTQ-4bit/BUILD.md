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
- **TunableOp lookup-only** — tuning mode autotunes prompt-length-dependent shapes
  mid-request; see the pitfall in 01-PATCHES.
- **dtype float16** — gfx1030 has no bf16 opcodes.
