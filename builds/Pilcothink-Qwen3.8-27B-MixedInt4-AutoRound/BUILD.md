# BUILD — Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound

**Model**: `Pilcothink/Qwen3.8-27B-MixedInt4-AutoRound` — Qwen3.8-27B, AutoRound (learned
rounding), mixed precision: W4 group_size 32 **symmetric** for the bulk, **int8 for 17
sensitivity-selected projections**, fp16 for the small GDN gates. 20.5 GB. Served as
`qwen38-27b-mixedint4`. Brought up 2026-08-22.

## Why this build exists

The strongest-priors quality candidate of the AutoRound field surveyed 2026-08-22: it stacks
the finest grid (g32), learned rounding, *and* — most importantly for this architecture —
int8 on the layers where 4-bit hurts a hybrid model most. Same bytes class as the other two
builds; the open question it exists to answer is whether those choices show up in output
quality. (MTP-head quantization affects only draft acceptance, never accuracy — verification
guarantees target-model outputs.)

## The quantization design

- **W4 g32 symmetric** (uint4b8) for all large projections → the Exllama route, same as the
  reference build.
- **int8** (uint8b128) for 17 projections chosen by sensitivity: `o_proj` on 7 attention
  layers, GDN `out_proj` on the 6 earliest layers, 2 `down_proj`, and layer 0's fused
  `in_proj_qkvz`. Early layers and output projections — where error compounds through the
  GDN recurrence — get double precision.
- **fp16 kept**: the tiny `in_proj_a`/`in_proj_b` GDN gates on all 48 layers (the same
  pieces cyankiwi's exporter ignored — independent agreement on what's fragile) and `mtp.fc`.
- lm_head fp16 (no known lm_head opportunity). MTP head shipped **quantized int4** — see below.

## The two conversions this checkpoint needs (scripts in this directory)

1. **`convert.py` — config rewrite.** The checkpoint declares `quant_method: "auto-round"`,
   which vLLM 0.27.1 does not know. Its tensors are already GPTQ-packed
   (`packing_format: auto_round:auto_gptq`), so the fix is metadata-only: rewrite the
   snapshot's `config.json` to `quant_method: gptq` and translate the per-layer
   `extra_config` table into GPTQ `dynamic` regex overrides (`+:` bits-8 entries, `-:`
   skip entries). Patterns target vLLM's *fused* module names (`in_proj_qkvz`,
   `in_proj_ba`) — the exporter's choices are fusion-consistent, so this is lossless.
2. **`mtp_to_dense.py` — MTP dequantization.** vLLM's `Qwen3_5MultiTokenPredictor` builds
   its layers unquantized (every prior checkpoint shipped fp16 MTP heads), so the int4 MTP
   tensors fail weight loading. The script dequantizes the 7 MTP linears to dense bf16
   (GPTQ g32 sym; stored zero-nibble 7 = legacy minus-one convention, true zero 8) into a
   new shard, then strips the packed originals from `model_extra_tensors.safetensors` —
   necessary because vLLM's loader streams every tensor in every shard, ignoring the index.
   Draft quality is therefore "int4-quality weights in fp16 containers"; measured acceptance
   is healthy (decode at reference rates).

**Both scripts run inside the container** (the snapshot blobs are root-owned) and must be
re-run after any re-download of the snapshot — the conversion edits the HF cache in place.

## Achieved performance (in-server, MTP=2)

| | 3.5–3.8k ctx | 13–17k ctx | 42k ctx |
|---|---|---|---|
| **Decode** | 47.0 t/s | 49.8 t/s | 41.6 t/s |
| **Prefill** | 856 t/s | 827 / 670 t/s | — |

Decode at reference parity. Prefill measured *above* the GPTQ reference (827 vs 747 @7.5k,
670 vs ~608 @16k, ~+10%) — plausibly the fp16-kept layers skipping dequant-reconstruct, but
not isolated; treat as in-band-or-better. Point estimates, ±5% spread. Validate: factual
spot-checks PASS (391, Canberra); baseline.json is this model's own.

## Working configuration — deltas from the reference build only

- **`QUANT=gptq`** after conversion — dispatches `Using ExllamaLinearKernel for
  AutoGPTQLinearMethod`; the default disable list applies unchanged (Exllama handles both
  uint4b8 and uint8b128, so the mixed int4/int8 layers all take the fast HIP kernel).
- Image: standard v2/v3 (no kernel work needed — symmetric checkpoint).
- Everything else inherited from the shared launcher.

## Caveats

- **The conversion lives in the HF cache**: `huggingface-cli download`/refresh of the
  snapshot reverts config.json and resurrects the packed MTP tensors — re-run both scripts.
- Draft head is int4-quality; if acceptance ever looks degraded vs the reference build,
  this is the first suspect (measured fine at bring-up).
- The actual purpose — quality comparison vs the GPTQ reference and cyankiwi AWQ on real
  workloads — has not been run yet.
