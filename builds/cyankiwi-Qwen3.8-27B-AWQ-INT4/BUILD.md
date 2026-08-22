# BUILD — cyankiwi/Qwen3.8-27B-AWQ-INT4

**Model**: `cyankiwi/Qwen3.8-27B-AWQ-INT4` — Qwen3.8-27B, compressed-tensors AWQ recipe,
W4 group_size 32, **asymmetric** (int8 zero-points, MSE observer). 19.57 GB — same bytes as
the GPTQ reference; a quality alternative at identical size, not a speed lever. MTP head
present and unquantized. Served as `qwen38-27b-awq`.

## Achieved performance (measured in the serving process, MTP=2)

| | 3.8k ctx | 15–17k ctx | 35–41k ctx |
|---|---|---|---|
| **Decode** | 49.8 t/s | 49.0 t/s | 43.9 t/s |
| **Prefill** | 753–778 t/s | 736 / 593 t/s | 524 t/s |

At parity or better vs the GPTQ reference on both axes (decode ahead at long context;
prefill 524 vs 521 at ~36k). Point estimates, ~±5% run-to-run spread.

## Working configuration

Differs from the reference build only where the checkpoint forces it — asymmetric uint4 is
outside Exllama's supported types, so the whole Exllama route is unavailable:

- **Image**: patches 0001–0006 — **0006 is required here**: it extends vLLM's HIP skinny
  int4 GEMV (`wvSplitK_int4_g`) from its gfx11/gfx12 guard down to gfx1030 and adds the
  prefill routes below. It touches a `.cu` file, so rebuild the C++ extension after applying
  (`python3 setup.py build_ext --inplace`, ~4 min incremental)
- **Launch**: `./serve.sh` — sets `QUANT=compressed-tensors` and a per-model
  `DISABLED_KERNELS` list that re-enables the hybrid kernel
- **Weights kernel**: RDNAHybridW4A16LinearKernel, three-way dispatch by batch size M:
  - M ≤ 5 → HIP skinny GEMV (decode; 6× over the pure-Triton fallback)
  - M ≥ 256 → single-pass Triton dequant to dense fp16 + rocBLAS (prefill; 3× over the
    fused Triton GEMM — the same structure Exllama uses internally for large M)
  - in between → fused Triton GEMM with gfx10 tiles, `num_stages=1`
- **Everything else inherited from the shared launcher** (MTP=2, int8 KV, TP=2, float16,
  CompilationMode 3, FULL_DECODE_ONLY graphs, both plugins, chunked prefill 8192,
  TunableOp lookup-only)

**Caveat**: the skinny decode path covers M ≤ 5; with MTP=2 each stream contributes M=3, so
two or more concurrent decode streams fall onto the fused Triton path. The decode numbers
above hold for a single stream — fine for single-user serving, measure before batching.
