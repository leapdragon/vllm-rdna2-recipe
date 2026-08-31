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

The whole order is automated in [`containers/build.sh`](containers/build.sh) (`--base` runs step 1), and
the result is published as `ghcr.io/leapdragon/vllm-rdna2-recipe` — see [containers/README.md](containers/README.md).
Step by step, for doing it by hand:

1. **Base image** — vLLM's `docker/Dockerfile.rocm_base` **with patch 0005**. Builds PyTorch from
   source for gfx1030. **Hours, not minutes.** This is the step that makes the card work at all.
2. **vLLM image** — patched 0.27.1 source (all nine patches; 0006 touches a `.cu`, so it is
   part of this compile), `pip install -e .` with `VLLM_TARGET_DEVICE=rocm`,
   `PYTORCH_ROCM_ARCH=gfx1030`. ~10 minutes for the extensions. Adding a patch later only
   needs an incremental `python3 setup.py build_ext --inplace` (~4 min), not a rebuild.
3. **Plugins** — `pip install ./builds/shared/plugins/fd_rdna2 ./builds/shared/plugins/ar_rdna2`.
4. **Serve** — `builds/<your-model>/serve.sh` if your model has a build directory (the
   already-optimized path), else `config/serve-rdna2-tp2.sh` directly with `MODEL=`/`SERVED=`/
   `QUANT=` set.

✅ **Step 1 has been run end to end (2026-08-31).** Until then this paragraph warned that our base was
built by hand before the recipe existed and that vLLM's own `rocm_base` recipe plus the arch change had
never been executed from scratch. It has now: [`containers/Dockerfile.rocm_base`](containers/Dockerfile.rocm_base)
— upstream's file with patch 0005 — built with `PYTORCH_ROCM_ARCH=gfx1030` in ~65 minutes on 32 cores
and produced, package for package, the stack in the table above (torch `2.11.0+gitd0c8b1f`, torchvision
`0.24.1+d801a34`, torchaudio `2.9.0+eaa9e4e`, Triton 3.6.0, amdsmi 26.2.2 — and, like the hand-built base,
no flash-attention/AITER/MORI: patch 0005 skips those CDNA-only stages, without which flash-attention's
`setup.py` rejects `gfx1030` 57 minutes in). The runtime image built on it passes the same checks and a
GPU smoke test; details in [containers/README.md](containers/README.md) "Build status". The arch-list trap
stands: `torch.cuda.get_arch_list()` must include `gfx1030` **with GPU devices attached to the container**,
or it returns an empty list with no error and tells you nothing. Without a device,
`torch._C._cuda_getArchFlags()` still returns the list torch was compiled for — that, plus the code-object
target IDs inside vLLM's extensions, is what `containers/Dockerfile` checks at build time.

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
| `MTP` | **2** | 27B: +24% @41k, +36% @14k · 122B: +47% @3.5k, +30% @13k, parity @40k | Speculative decoding via the checkpoint's own MTP head; output-lossless. **Requires the shipped `fd_rdna2`** (batched verification) or it becomes a large regression, and **under PP it additionally requires the V2 runner + TunableOp lm_head rows** (Intel BUILD.md). `MTP=0` disables. |
| `TUNEOP_TUNING` | **0** | prevents minutes-long prefill stalls | TunableOp lookup-only; `1` re-enables autotuning for deliberate offline sessions only (see pitfalls in 01). The results CSV itself is **load-bearing**: the tuned lm_head rows are worth ~1.7× decode under MTP, and a missing/validator-mismatched CSV silently reverts to the ~115 GB/s heuristic pick (27B BUILD.md, TROUBLESHOOTING 5c). Per-build reference CSVs are shipped in `builds/*/tunableop/`. |
| `FD_MAXQ` | 4 | — | Widest verification batch the attention plugin takes. The one-KV-pass batched kernel covers `nq × PAD ≤ 32` columns (PAD 8 at GQA ≤ 8, 16 at GQA ≤ 16); wider cases run per-position passes of the same kernel. |
| `--max-num-batched-tokens` (`BATCHTOK`) | **8192** | swept optimum | 4096 and 16384 both measure worse at mid/long context. |

| `--tensor-parallel-size` | 2 | | Two-card 27B builds. The 122B runs `TP=4` across four cards (flagship; REQUIRES the platform-stability stack below) or `TP=1 PP=3` on three. |
| Card power cap | 232 W | **free** | Decode is bandwidth-bound; the ~30% clock throttling it causes does not slow decode. |

**Pipeline-parallel / MTP-under-PP knobs** (added for the 122B build; harmless elsewhere):

| Setting | Value (122B) | Notes |
|---|---|---|
| `PP` | 3 | `--pipeline-parallel-size` (layer split). TP=3 is arithmetically impossible on this model (2 KV heads). |
| `PP_PARTITION` | `17,17,14` | Uneven layer split (`VLLM_PP_LAYER_PARTITION`); unloads the last stage, which hosts the MTP draft model. |
| `EXTRA_ENV` | `VLLM_USE_V2_MODEL_RUNNER=1` | **MTP under PP requires the V2 model runner** — the V1 drafter path page-faults under PP (patch 0009's section in 01-PATCHES). Comma-separated `NAME=VALUE` passthrough. |
| `ASYNC_SCHED` | leave default | `0` forces sync scheduling — needed only for MTP-under-PP experiments on the V1 runner (its async PP broadcast assumes width-1 samples). Unnecessary on V2. |
| `SPEC_EAGER` | leave default | `1` runs only the drafter eager (diagnostic; isolates drafter-cudagraph interplay). |
| `MOE_CFG` | build-provided | Mounts a tuned fused-MoE config JSON into vLLM's configs dir. |
| `EXTRA_MOUNT` | — | Comma-separated `host:container` overlays for rebuild-free iteration. |

**Note on the launcher's paths and bind mounts:** every host path in
`config/serve-rdna2-tp2.sh` is an environment variable with a sensible default —
`RECIPE_ROOT` (auto-detected from the script's own location; plugins and state dirs live
under it), `HF_CACHE`, `TUNEOP_DIR`, and `STATE_DIR` (compile/extension caches, created on
demand). The one with no default is `VLLM_SRC`: point it at your patched vLLM 0.27.1
checkout and the launcher bind-mounts the most-edited source files (`rocm.py`, attention,
the W4A16 kernels) over the image's copies, so post-bake patch changes deploy without a
rebuild. Leave it unset and the image's baked copies run — correct whenever your image was
built from the fully-patched tree.

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

The expectations below are for the reference 27B GPTQ build at TP=2; every other build's
numbers are in its own `builds/<model>/BUILD.md`.

```bash
verify/validate.py --compare verify/baseline-rccl.json    # expect: identical=8 diverged=0 (MTP on or off)
verify/decode-rate.py --ctx 4000,16000,43000              # MTP=2: ~43-51 / ~50 / ~41 t/s; MTP=0: ~37 / ~37 / ~33.5
verify/prefill-rate.py --ctx 4000,16000,37000             # ~834 / ~747 / ~521 tok/s, and UNIFORM across trials
verify/soak.py --minutes 30 --conc 3                      # expect: SOAK CLEAN
```

Do not measure speculative decoding with median inter-token-gap tools (stream deltas can carry
several tokens) or with total-wall ÷ tokens (charges the partial-block re-prefill to decode) —
both mis-measured MTP as a regression here before `decode-rate.py`. If prefill rates are wildly
non-uniform across trials, TunableOp tuning is on: see the pitfalls in 01.

`baseline-rccl.json` holds greedy outputs from a known-good run. **8/8 identical is a hard gate**
for every change in this repo except one: none of these optimisations alter numerics, so any
divergence is a bug — with the single exception of the TunableOp first-run artefact described in
01-PATCHES.

If you get correct output but ~18 t/s at 42k, the plugins are not active. If you get ~4 t/s, the
Exllama kernel is not being forced.


## Platform-stability stack (required for multi-card tensor parallelism)

V620s are SR-IOV graphics cards; sustained lockstep collective kernels sit
outside their qualification envelope and can wedge the command processor
(`qcm fence timeout` → `device lost from bus`) — with or without P2P. The
following stack made flat TP=4 stable (validated ~2.5 h sustained, zero
events; treat as all-load-bearing until noted otherwise):

| Layer | Setting | Why |
|---|---|---|
| kernel cmdline | `amdgpu.pcie_gen_cap=0x00070007` | Cap links at Gen3. Links *train* at Gen4 but deliver less (marginal signaling); verify via `pp_dpm_pcie` — `current_link_width`/`current_link_speed` sysfs lie on some boards. |
| kernel cmdline | `amdgpu.aspm=0 amdgpu.runpm=0` | No link/device low-power exits racing an idle→full-burst transition (where the drops clustered). |
| kernel cmdline | `amdgpu.gpu_recovery=1` | A wedge becomes a GPU reset instead of a card lost until reboot. |
| env | `HSA_NO_SCRATCH_RECLAIM=1` | No mid-flight scratch reclaim/regrow (queue surgery at dispatch time of the largest kernels — also the likely truth behind llama.cpp's classic "batch 4096+ is crashy" lore). |
| env | `NCCL_P2P_LEVEL=PXB` | RCCL P2P within a root complex, SHM across. |
| vLLM | `--max-num-batched-tokens 2048` | Batch size is a TIMING knob: unpreemptible dispatch length, scratch-crossing odds, DMA burst duration, power-ramp width all scale with it. Keep work items frame-sized. |
| per boot | power caps at 232 W | Gentler transients (TP=4 decode only draws ~180 W anyway — bandwidth-bound). |

**Measured cost of this stack (2026-08-31): none.** A same-day regression hunt briefly blamed this
cmdline for a 27-vs-50 t/s decode gap; the true cause was the lost TunableOp lm_head rows (the 27B
BUILD.md's "load-bearing rows" section), and with them restored the recorded numbers reproduce on the
fully hardened platform. The stack stays: it is what stopped the card-drop crashes
([TROUBLESHOOTING §4](TROUBLESHOOTING.md)) and it costs nothing.
