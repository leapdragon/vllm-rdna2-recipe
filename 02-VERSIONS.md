# 02 — Versions, pinning, and what to do when the pins don't hold

Copyright © 2026 Aron Hsiao · GPL-3.0-or-later (see LICENSE)

## The exact stack these numbers came from

| Component | Pin | Notes |
|---|---|---|
| vLLM | **0.27.1** | Patches target this tag. |
| ROCm | **7.2.3**, in-container | HIP 7.2.53211-c2d9476115, AMD clang 22.0.0git (`roc-7.2.3`). **Never mount host ROCm.** Host here is a mixed apt-7.1 + TheRock-7.14 install; mounting it caused page faults. |
| PyTorch | **2.11.0+gitd0c8b1f**, built from source for gfx1030 | Built by vLLM's own `docker/Dockerfile.rocm_base` with patch 0005 applied. |
| Triton | ships with that torch | The attention plugin is Triton; no separate pin. |
| Model | `btbtyler09/Qwen3.8-27B-GPTQ-4bit` | GPTQ 4-bit, group_size 32, symmetric. |
| Host OS | Ubuntu 26.04, kernel 7.0 | |
| Container runtime | Docker, `--device /dev/kfd --device /dev/dri` | Needs `--group-add` for `render` and `video`. |

## Build order

1. **Base image** — vLLM's `docker/Dockerfile.rocm_base` **with patch 0005**. Builds PyTorch from
   source for gfx1030. **Hours, not minutes.** This is the step that makes the card work at all.
2. **vLLM image** — patched 0.27.1 source, `pip install -e .` with `VLLM_TARGET_DEVICE=rocm`,
   `PYTORCH_ROCM_ARCH=gfx1030`. ~10 minutes for the extensions.
3. **Plugins** — `pip install ./plugins/fd_rdna2 ./plugins/ar_rdna2`.
4. **Serve** — `config/serve-rdna2-tp2.sh`.

⚠️ **We did not rebuild the base image from scratch to verify step 1.** Our working base was built
by hand before this recipe existed; we later identified that vLLM's own `rocm_base` recipe plus
the one-line arch change is what produces it. The reasoning is sound and the patch is trivial, but
**you will be the first to run it end to end.** Budget accordingly, and if it diverges, check the
torch build arch list — `torch.cuda.get_arch_list()` must include `gfx1030` — **with GPU devices
attached to the container**, or it returns an empty list with no error and tells you nothing.

## Runtime configuration and what each setting is worth

All in `config/serve-rdna2-tp2.sh`. Each is individually reversible.

| Setting | Value | Worth | Notes |
|---|---|---|---|
| `--compilation-config mode` | **3** (`VLLM_COMPILE`) | **+20%** | Cuts tiny elementwise kernels per decode step from 2,553 to 338. |
| `cudagraph_mode` | `FULL_DECODE_ONLY` | (with mode 3, ±0.1% vs `FULL_AND_PIECEWISE`) | An older result calling piecewise a 2× regression **does not reproduce**; it was measured while thermally throttled. |
| `GPU_MAX_HW_QUEUES` | **4** | +3.8% | 8 is a **32% regression**. |
| `FD_RDNA2` | 1 | slope 6.06× flatter | Attention plugin. |
| `AR_RDNA2` | 1 | +1.9% | All-reduce plugin. |
| `--kv-cache-dtype` | `int8_per_token_head` | large at long context | |
| `VLLM_DISABLED_KERNELS` | forces Exllama | ~4 t/s → ~10 t/s class | See pitfalls in 01. |
| `--tensor-parallel-size` | 2 | | Must be the two ×16-rooted cards. |
| Card power cap | 232 W | **free** | Decode is bandwidth-bound; the ~30% clock throttling it causes does not slow decode. |

## When the pins don't hold

**A newer vLLM.** Patches 0001–0003 target files that move between releases. Patch 0004 is one
constant in the GPTQ kernel and is likely to survive. The two plugins monkey-patch
`triton_attn.unified_attention` and `GroupCoordinator._all_reduce_out_place` — **assume both break
on any minor version bump**, and note that they fail *silently* (the plugin still loads). Re-point
them and confirm the fast path is taken by checking the delegation counters in the log.

**A different Qwen3.x quant.** Group size, symmetry, and whether `lm_head` is quantised all change
the byte budget and therefore the targets. Recompute bytes-per-token from the safetensors headers
before comparing against our numbers.

**A different RDNA2 card.** See the scope note in `00-HARDWARE.md`. The architectural fixes should
carry; the numbers will not.

**Upstream lands PR #52391.** Then patch 0001 is redundant — check before applying.

## Verifying you arrived

```bash
verify/validate.py --compare verify/baseline-rccl.json   # expect: identical=8 diverged=0
verify/longctx-decode.py --ctx 200,4000,16000,43000      # expect: ~37 / ~37 / ~37 / ~33.5 t/s
verify/soak.py --minutes 30 --conc 3                     # expect: SOAK CLEAN
```

`baseline-rccl.json` holds greedy outputs from a known-good run. **8/8 identical is a hard gate**
for every change in this repo except one: none of these optimisations alter numerics, so any
divergence is a bug — with the single exception of the TunableOp first-run artefact described in
01-PATCHES.

If you get correct output but ~18 t/s at 42k, the plugins are not active. If you get ~4 t/s, the
Exllama kernel is not being forced.
