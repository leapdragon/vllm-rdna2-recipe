# BUILD — btbtyler09/Qwen3.8-27B-GPTQ-4bit

**Model**: `btbtyler09/Qwen3.8-27B-GPTQ-4bit` — Qwen3.8-27B, GPTQ W4, group_size 32,
**symmetric** (uint4b8). 19.54 GB. Served as `qwen38-27b-gptq`. The reference build the rest
of this recipe was developed against.

## Achieved performance (measured in the serving process, MTP=2)

| | 3.5k ctx | 14–15k ctx | 37–41k ctx |
|---|---|---|---|
| **Decode** | 43–51 t/s | 50.4 t/s | 41.4 t/s |
| **Prefill** | 834 t/s | 747 / 608 t/s | 521 t/s |

Decode without speculation: 33.5 t/s @41k. Point estimates; run-to-run and boot-to-boot
spread on this platform is ~±5%. Beats llama.cpp on the same cards (31–40 t/s @45k).

## Working configuration

- **Image**: patches 0001–0005 applied per `01-PATCHES.md` (0006 is also safe to include;
  it only affects kernels this build keeps disabled)
- **Launch**: `./serve.sh` — this model *is* the shared launcher's defaults
- **Weights kernel**: ExllamaLinearKernel, forced via `VLLM_DISABLED_KERNELS` (every W4
  alternative disabled); `BLOCK_KN_SIZE=256` from patch 0004
- **Attention**: stock Triton unified attention with the gfx10x LDS/stages fix (patch 0002);
  fd_rdna2 flash-decode plugin (`FD_RDNA2=1 FD_MAXQ=4`)
- **All-reduce**: ar_rdna2 push plugin (`AR_RDNA2=1`)
- **Speculation**: `qwen3_5_mtp`, 2 drafts (`MTP=2`)
- **KV cache**: `int8_per_token_head`, max_model_len 131072
- **Serving**: TP=2 across the two x16-rooted cards, dtype float16, CompilationMode 3,
  FULL_DECODE_ONLY graphs, `GPU_MAX_HW_QUEUES=4`, chunked prefill 8192,
  TunableOp **lookup-only** (see the tuning pitfall in `01-PATCHES.md`)
