# BUILD — cyankiwi/Qwen3.8-27B-AWQ-INT4

**Model**: `cyankiwi/Qwen3.8-27B-AWQ-INT4` — Qwen3.8-27B, compressed-tensors export of an AWQ
recipe, W4 group_size 32, **asymmetric** (int8 zero-points, MSE observer). 19.57 GB. Served
as `qwen38-27b-awq`.

## Why this build exists

Same architecture and essentially the same bytes as the GPTQ reference (19.57 vs 19.54 GB) —
a **quality experiment at identical cost**, not a speed lever. The hypothesis: asymmetric
quantization with an MSE observer can place the 16 levels per group better than symmetric
GPTQ. Its value to *you* is different: it is the worked example of adapting this recipe to a
checkpoint whose quantization format forces a different kernel route.

## The model: identical structure, different quantization coverage

Architecture facts (head_dim 256, 48 GDN + 16 attention layers, fp16 lm_head, MTP head,
54.9 t/s bandwidth floor) are the reference build's — see its BUILD.md. What differs is what
the exporter chose to quantize:

- **All 15 MTP-head tensors left unquantized** → `MTP=2` carries over intact. Check this
  *before* downloading any quant: a quantized-away MTP head costs ~24% decode here.
- **lm_head left unquantized** — same streaming cost as the reference.
- **GDN large projections quantized**; the small, sensitive GDN pieces and the vision tower
  are in the ignore list (we serve `language_model_only`, so the tower never loads).

## The quantization, and why it forces a different kernel route

**Asymmetric W4 arrives as ScalarType `uint4` with real zero-point tensors — outside
Exllama's supported types** (`uint4b8`/`uint8b128` only; its "zero-point" path *synthesizes*
constant zeros for symmetric checkpoints, it never consumes stored ones). The reference
build's forcing disable-list therefore fails **loudly at boot** on this model — by design,
and better than the alternative: with the list relaxed, vLLM's dispatch serves it through
pure-Triton W4A16 at **7.3 t/s decode** (18× off), flat across context.

The working route is **RDNAHybridW4A16LinearKernel**, which upstream gates to gfx11/gfx12.
Everything its device code needs — wave32, 64 KiB LDS, `v_dot2_f32_f16`, DPP `row_shr` — is
gfx10.3 hardware, so **patch 0006** extends the guard (plus a scalar fallback for the one
missing instruction, bf16 dot, unused under fp16 serving). The kernel then dispatches three
ways by batch size M:

| M | Route | Why |
|---|---|---|
| ≤ 5 | HIP skinny GEMV (`wvSplitK_int4_g`) | Decode is bandwidth-bound single-row work; the hand-written HIP GEMV streams weights near the ceiling. 6× over the Triton fallback. |
| ≥ 256 | Triton dequant → dense fp16 → rocBLAS | Prefill is compute-bound GEMM, where a fused-dequant Triton kernel loses ~3× to the vendor library; dequant traffic amortizes to ~1% at M=8192. Same structure Exllama uses internally for large M. |
| in between | Fused Triton GEMM, gfx10 tiles, `num_stages=1` | Tail chunks and small batches. |

## Achieved performance (in-server, MTP=2)

| | 3.8k ctx | 15–17k ctx | 35–41k ctx |
|---|---|---|---|
| **Decode** | 49.8 t/s | 49.0 t/s | 43.9 t/s |
| **Prefill** | 753–778 t/s | 736 / 593 t/s | 524 t/s |

At parity or better vs the reference on both axes — decode *ahead* at long context (43.9 vs
41.4; the skinny GEMV edges `q_gemm` there), prefill 524 vs 521 at ~36k. Point estimates,
±5% run-to-run spread.

## Working configuration — deltas from the reference build only

- **Image**: patches 0001–**0006** — 0006 is required here and touches a `.cu`, so rebuild
  the C++ extension after applying (`python3 setup.py build_ext --inplace`, ~4 min
  incremental). Harmless to the reference build, whose disable list keeps the hybrid off.
- **`QUANT=compressed-tensors`** — the checkpoint's native format; `gptq` would misparse it.
- **`DISABLED_KERNELS=RDNA3W4A16LinearKernel,ConchLinearKernel`** — drops RDNAHybrid and
  TritonW4A16 from the default disable list: the hybrid outranks Triton in vLLM's ROCm
  priority order and Exllama self-rejects, so the hybrid is chosen, with Triton as the loud
  fallback. Verify the boot log says `Using RDNAHybridW4A16LinearKernel`.
- Everything else (MTP=2, int8 KV, TP=2, float16, graphs, plugins, chunked prefill 8192,
  TunableOp posture) is inherited from the shared launcher unchanged.

## Caveats

- **Single-stream decode numbers.** The skinny path covers M ≤ 5, and each MTP=2 stream
  contributes M=3 — two or more concurrent decode streams fall to the fused Triton path.
  Measure before batch-serving this build.
- Generate this model's own greedy baseline at bring-up; never compare baselines across
  quants — the numerics legitimately differ.
