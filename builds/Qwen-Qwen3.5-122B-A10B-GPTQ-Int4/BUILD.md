# BUILD — Qwen/Qwen3.5-122B-A10B-GPTQ-Int4

**Model**: `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` — the official Qwen GPTQ quant of the 122B
MoE hybrid (256 experts, 8 active/token, 48 layers = 12 attention + 36 GDN, 2 KV heads,
head_dim 256). 78.84 GB on disk. Served as `qwen35-122b-gptq`. **The first model beyond
two cards**: PP=3 (layer split, 16/16/16) across all three V620s. Brought up 2026-08-22/23.

## Why this build exists

The first model class this machine could not previously reach: too big for the TP=2 pair
(no W4 fits 2×32 GB), and TP=3 is arithmetically impossible (2 KV heads, 32 Q heads, 256
experts — nothing divides by 3). Pipeline parallelism is the only vLLM route on three
cards, and this build proves it works. It is also the `moe_wna16`-on-gfx1030 pathfinder —
the first MoE expert kernels ever run on this chip (RESEARCH-4CARD.md risk #1, confirmed).

## The quantization — read the `dynamic` map before estimating anything

**Only the experts are int4** (GPTQ sym g128, per-expert tensors: 36,864 quantized linears).
The `dynamic` config excludes everything else: **all attention/GDN projections, the shared
expert, and lm_head are bf16**. Consequences:

- ~16 GB of *always-active* bf16 backbone streams every token → **decode ceiling ≈ 28 t/s**
  regardless of MoE kernel quality. (A community quant with a quantized backbone would
  roughly double the ceiling — worth surveying as a follow-on.)
- The MTP head is a full bf16 MoE layer (~4.8 GB, 785 tensors) — skipped at load since
  MTP is off under PP.
- No Exllama anywhere: nothing the MPLinear registry handles is quantized here. The
  experts run vLLM's `moe_wna16` → Triton fused-MoE int4 path.

## Achieved performance (in-server, PP=3, MTP=0)

| | 3.8k ctx | 13–14k ctx |
|---|---|---|
| **Decode** | 7.1 t/s | 7.0 t/s |
| **Prefill** | 558 t/s | 721 t/s (@7k) |

Validate: factual PASS, 7/8 identical across config changes. Flat decode vs context (KV is
tiny here: 2 KV heads × 4 attention layers/stage). For scale: llama.cpp on this machine
serves this model class at usable rates; vLLM decode is not yet competitive — see below.

## The MoE kernel story (how these numbers happened, and what's left)

1. First boot ran the **default fused-MoE config** ("Using default MoE config. Performance
   might be sub-optimal!") → 6.4 t/s decode, 311–421 t/s prefill.
2. Hand-guessed configs failed twice (narrow-N lost 20%; wide-N overflowed the 64 KiB LDS —
   N=1024×K=64 tiles want 128 KB).
3. **Offline sweep** (`moe_sweep` harness — direct `fused_experts` calls at the exact
   shapes E=256/N=1024/K=3072/topk=8/g128, cache-defeating layer rotation, ~350 configs):
   winners baked into `moe-config-gfx1030.json`, mounted via the launcher's `MOE_CFG` knob
   to the exact path vLLM's warning names. Result: **decode +11%, prefill +75%**.
   All winners use `num_stages=1` (the standing rule holds for this kernel too).
4. **The CUDA `moe_wna16` kernel is now ported to gfx1030** (patch 0007) and this
   build runs it: correct in-server, ~1.1× the Triton path at real decode sizes. That
   port also *corrected the attribution*: T36's "MoE is 45× off bandwidth" came from
   sweeping M=8; at the real M=1 the MoE costs ~46 ms of the 140 ms token. The rest is
   ~45 ms of bf16 backbone streaming (this quant's unquantized attention/GDN — fixable
   only by a backbone-quantized checkpoint, which would lift the ceiling toward
   ~55 t/s) and ~50 ms unattributed with decode graphs confirmed captured (PP stage
   handoffs suspected — profiling is the open lever).

## Working configuration — deltas from the shared launcher

- **`TP=1 PP=3 DEVICES=0,1,3`** — new launcher knob `PP` (pipeline-parallel stages).
- **`GPUUTIL=0.95 MAXSEQS=4`** — required: at 0.92 one stage (embedding/lm_head extras +
  MoE activation-profiling peak) leaves only 0.16 GiB for KV. At 0.95/4: 2.51 GiB, far
  more than 131k context needs.
- **`MTP=0`** — vLLM restricts speculative decoding under PP.
- **`FD_RDNA2=0 AR_RDNA2=0 CUSTOM_AR=0`** — no all-reduce exists at TP=1; the attention
  plugin is untested at 2 KV heads (extending it is a sized follow-on, after the MoE
  kernel which dominates).
- **`MOE_CFG`** — mounts `moe-config-gfx1030.json` (this dir) to
  `configs/E=256,N=1024,device_name=AMD_RADEON_PRO_V620_Azure,dtype=int4_w4a16.json`.

## Caveats and watch items

- **Flaky boot ~2-in-7**: a PP worker occasionally dies during weight loading (once
  "could not determine the shape of object type 'torch.storage.UntypedStorage'", once a
  silent death mid-shard-load; all 39 shards verify clean, and a plain retry boots).
  Likely a 3-rank × 74 GB mmap race. Retry before diagnosing.
- Decode at 7 t/s is a **pathfinder number, not a production number** — usable for
  correctness/quality work; the MoE kernel campaign is the gate to production viability.
- MAXLEN validated at 131072 (KV is nowhere near the constraint); 262144 native untested.
- `baseline.json` here is this model's own greedy baseline.
