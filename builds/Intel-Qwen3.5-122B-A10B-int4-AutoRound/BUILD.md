# BUILD — Intel/Qwen3.5-122B-A10B-int4-AutoRound

**Model**: `Intel/Qwen3.5-122B-A10B-int4-AutoRound` — AutoRound int4 (sym g128, GPTQ
packing) of the 122B MoE hybrid. 71.7 GB. Served as `qwen35-122b-autoround`, PP=3 across
all three V620s. **Replaced `Qwen/Qwen3.5-122B-A10B-GPTQ-Int4` on 2026-08-23** — that
build's bring-up record (PP enablement, MoE config sweep, moe_wna16 port) lives in
the recipe changelog; this directory carries its
inherited assets (moe-config-gfx1030.json, moe_sweep.py — same E=256/N=1024 shapes).

## Why this checkpoint (and why the official one left)

The official Qwen GPTQ quantized **only the experts**, leaving ~18 GB of bf16 backbone
streaming every token — worse than the bytes suggest, since gfx1030 has no native bf16 and
pays a penalized GEMV path. Metadata inspection (no download needed) showed Intel's quant
covers the backbone too: every attention/GDN linear int4, only the **shared expert (+gate)
kept fp16** (144 modules), MTP head entirely bf16 and outside the quantize list. Result:

| | official GPTQ | **this build** |
|---|---|---|
| **Decode** | 7.1 t/s | **15.7 / 15.0 t/s** (3.2k / 15.7k) |
| **Prefill** | 558 / 721 t/s | 566 / 741 t/s (unchanged, as expected) |

Factual spot-checks PASS; `baseline.json` is this model's own.

## Conversion required (script in this directory)

vLLM 0.27.1 has no "auto-round" method: `convert.py` rewrites the snapshot's config to
`gptq` + 144 `dynamic` skip patterns (the shared-expert exclusions, gate/up mapped to
vLLM's fused `gate_up_proj`). Same class of surgery as the Pilcothink 27B build, but
simpler — no int8 tiers and no MTP dequant (the head was never quantized). Re-run after
any re-download. Weights untouched; metadata translation only.

## The corrected token budget, and what's next

~64 ms/token ≈ **~46 ms MoE experts** + ~10 ms int4 backbone + ~8 ms PP/other. Two earlier
attributions died on this build's evidence: an earlier "MoE is everything" (wrong at M=1) and
T37's "~50 ms unattributed, PP suspected" (mostly the *underestimated* bf16 backbone, now
gone). The MoE expert path — 46 ms to stream 1.8 GB that bandwidth prices at ~4 ms — is now
**72% of the token and the single remaining lever**: a wvSplitK-class expert GEMV targets
~25–30 ms/token ≈ **35–40 t/s**, which would pass llama-server's 30+ on this model.

## Working configuration

Inherited from the predecessor unchanged: `TP=1 PP=3 DEVICES=0,1,3`, image v4 (moe_wna16
port), `GPUUTIL=0.95 MAXSEQS=4`, `MTP=0` (PP forbids speculation; head is bf16 and skipped
at load), `FD_RDNA2=0 AR_RDNA2=0`, MOE_CFG → the swept fused-MoE config. Watch item also
inherited: flaky load-time worker death (~2-in-7 boots; plain retry works).
