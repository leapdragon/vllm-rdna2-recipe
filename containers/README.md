# Container image

Copyright © 2026 Aron Hsiao · GPL-3.0-or-later (see [LICENSE](../LICENSE))

The recipe, prebuilt: pristine vLLM 0.27.1 with [patches 0001–0009](../01-PATCHES.md) applied
and compiled for gfx1030, the `fd_rdna2` / `ar_rdna2` plugins installed, on ROCm 7.2.3 with a
PyTorch 2.11 built from source for gfx1030. It is the same stack [02-VERSIONS.md](../02-VERSIONS.md)
pins, produced by two Dockerfiles in this directory instead of by hand.

| Image | Tag | What |
|---|---|---|
| `ghcr.io/leapdragon/vllm-rdna2-recipe` | `0.27.1-rocm7.2.3-gfx1030`, `latest` | Runtime image — [`Dockerfile`](Dockerfile). 26.9 GB. |
| `ghcr.io/leapdragon/vllm-rdna2-recipe-base` | `rocm7.2.3-torch2.11.0-gfx1030` | PyTorch/ROCm base — [`Dockerfile.rocm_base`](Dockerfile.rocm_base). 24.2 GB. Only needed to rebuild the runtime image. |
| `ghcr.io/leapdragon/vllm-rdna2-recipe` | `0.27.1-rocm7.2.3-gfx1030-asbuilt` | Optional: a snapshot of the hand-built image every number in this repo was measured on. Not reproducible from the Dockerfiles; published only as a reference point. |

Tags are immutable in intent: a rebuild of the same stack gets a new suffix, not a moved tag.
`latest` follows the newest runtime tag.

## Run it

The recommended path is the recipe's own wrapper, which carries the tuned flags, mounts, and
plugin knobs that every measured number used — point it at the published image:

```bash
git clone https://github.com/leapdragon/vllm-rdna2-recipe && cd vllm-rdna2-recipe
IMG=ghcr.io/leapdragon/vllm-rdna2-recipe:0.27.1-rocm7.2.3-gfx1030 \
  ./builds/btbtyler09-Qwen3.8-27B-GPTQ-4bit/serve.sh      # or any builds/*/serve.sh
```

The minimal bare invocation, for when you want to see every moving part:

```bash
docker run -d --name vllm-rdna2 --network=host \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" \
  --ipc=host --ulimit memlock=-1 --security-opt seccomp=unconfined --cap-add SYS_PTRACE \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  -v vllm-rdna2-ext:/ext-cache         -e TORCH_EXTENSIONS_DIR=/ext-cache \
  -v vllm-rdna2-compile:/compile-cache -e VLLM_CACHE_ROOT=/compile-cache \
  -v vllm-rdna2-triton:/triton-cache   -e TRITON_CACHE_DIR=/triton-cache \
  -e FD_RDNA2=1 -e AR_RDNA2=1 \
  ghcr.io/leapdragon/vllm-rdna2-recipe:0.27.1-rocm7.2.3-gfx1030 \
    btbtyler09/Qwen3.8-27B-GPTQ-4bit --served-model-name qwen38-27b-gptq \
    --dtype float16 --quantization gptq --attention-backend TRITON_ATTN \
    --tensor-parallel-size 2 --max-model-len 32768 \
    --enable-chunked-prefill --enable-prefix-caching --language-model-only --skip-mm-profiling
```

- The entrypoint is `vllm serve`; everything after the image name is its arguments
  (`docker run --rm --device /dev/kfd --device /dev/dri IMAGE --help` — upstream vLLM infers the
  device while parsing arguments, so even `--help` wants `/dev/kfd`). `--entrypoint bash` gets you a shell.
- The three named volumes persist the JIT-built all-reduce extension, the torch.compile cache
  (cold boot compiles for 10–15 min; warm is 3–4 min) and the Triton kernel cache.
- `render`/`video` are the host groups that own `/dev/kfd` and `/dev/dri/renderD*`; the numeric
  ids differ between distributions, hence `getent`.
- Health: `docker inspect --format '{{.State.Health.Status}}' vllm-rdna2` — the built-in check
  polls `/health` on `HEALTHCHECK_PORT` (default 8000; pass `-e HEALTHCHECK_PORT=` if you move `--port`)
  with a 30-minute start period for the cold-compile case.
- Runs as root, like the upstream ROCm images: `/dev/kfd` access is granted by the group adds.

### What is baked in

| Path / setting | Purpose |
|---|---|
| `/app/vllm-src` | The patched vLLM tree, installed editable. `git -C /app/vllm-src diff --stat` is exactly the nine patches against tag `v0.27.1`; `config/serve-rdna2-tp2.sh`'s `VLLM_SRC=` overlay mounts land here. |
| `/app/plugins/{fd_rdna2,ar_rdna2}` | The plugin sources as installed (`vllm.general_plugins` entry points). Switched on by `FD_RDNA2=1` / `AR_RDNA2=1`; knobs in [02-VERSIONS.md](../02-VERSIONS.md). |
| `/app/recipe/{patches,verify,…}` | The patches, the verification clients (`verify/validate.py`, `decode-rate.py`, …) and the key docs, for reference from inside the container. |
| `/app/versions.txt` | Provenance: base image pins (every upstream repo + commit the base was built from), `vllm=`, `vllm_commit=`, `recipe_patches=`, `torch=`. |
| `HSA_OVERRIDE_GFX_VERSION=10.3.0` | Lets the gfx1030 code objects run on the rest of Navi 2x (gfx1031/1032/…). No-op on a real gfx1030. |
| `FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE`, `VLLM_ROCM_USE_AITER=0`, `VLLM_ROCM_USE_AITER_MOE=0`, `TORCH_BLAS_PREFER_HIPBLASLT=0` | Triton attention is the only attention path on this chip; AITER and hipBLASLt have no gfx1030 kernels. |
| `HSA_NO_SCRATCH_RECLAIM=1`, `HIP_FORCE_DEV_KERNARG=1`, `VLLM_WORKER_MULTIPROC_METHOD=spawn`, `TOKENIZERS_PARALLELISM=false`, `SAFETENSORS_FAST_GPU=1` | Upstream ROCm image defaults plus the recipe's stability settings ([02-VERSIONS.md](../02-VERSIONS.md) "stability"). |

Everything in the table is an environment default and can be overridden with `docker run -e`.
TunableOp, the all-reduce/attention plugin knobs, MTP, and the model flags are deliberately
*not* baked: they are per-model policy and live in the wrapper.

### Verify

```bash
IMG=ghcr.io/leapdragon/vllm-rdna2-recipe:0.27.1-rocm7.2.3-gfx1030
docker run --rm --entrypoint cat "$IMG" /app/versions.txt
# No devices needed: the arch list torch was compiled for (this is what the build-time check uses)
docker run --rm --entrypoint python3 "$IMG" -c "import torch; print(torch._C._cuda_getArchFlags())"   # -> gfx1030
# With devices attached — without them torch.cuda.get_arch_list() is an EMPTY list and no error:
docker run --rm --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" \
  --entrypoint python3 "$IMG" -c "import torch; print(torch.cuda.get_arch_list(), torch.cuda.get_device_name(0))"
# -> [..., 'gfx1030', ...] AMD Radeon PRO V620
```

Then the recipe's own checks against a running server: [`verify/validate.py`](../verify/validate.py)
(correctness), `verify/decode-rate.py` / `verify/prefill-rate.py` (numbers to compare with the
`builds/*/BUILD.md` tables).

## Build it yourself

```bash
./containers/build.sh --base     # 1. base: PyTorch/Triton (+FA/AITER/MORI only for CDNA arch lists) from source — ~65 min on 32 cores, every core
                                 # 2. runtime: clone v0.27.1, apply patches, compile vLLM for gfx1030 — 15–30 min
./containers/build.sh            # runtime only, FROM the published base (or BASE_IMAGE=… for a local one)
```

- `Dockerfile.rocm_base` is upstream vLLM's `docker/Dockerfile.rocm_base` at `v0.27.1`, vendored
  verbatim with patch 0005 applied: gfx1030 in the arch default, and the CDNA-only stages
  (flash-attention, AITER, MORI) skipped when the arch list has no `gfx9xx` — flash-attention refuses
  gfx10xx outright, and the measured base never contained the three. `build.sh` passes
  `PYTORCH_ROCM_ARCH=gfx1030` so nothing is compiled for other targets. It does not read
  `MAX_JOBS`: expect all cores for the duration — don't benchmark on the same box meanwhile. On a
  32-core host the PyTorch stage alone is ~45 min (Triton ~16 min, hipify ~5 min).
- `Dockerfile` pins the vLLM tag **and** its commit (`6e448d0ea9…`): a moved tag fails the build
  instead of silently changing what the patches apply to. Every extension is checked at build
  time for gfx1030 code objects (torch via its compiled-in arch flags), for the patch-0007/0008 ops,
  the patch-0009 relay, and a stock (not binary-patched) HSA runtime — no GPU needed for any of it.
- `pip` resolves vLLM's requirements under a constraints file generated from the base, so it can
  never replace the source-built torch/torchvision/torchaudio/triton with PyPI wheels.
- Knobs: `REGISTRY_REPO`, `VERSION`, `BASE_TAG`, `BASE_IMAGE`, `PYTORCH_ROCM_ARCH`, `MAX_JOBS`
  (`./containers/build.sh --help`). Adding a patch later: put it in `patches/`, rebuild — the
  clone/patch layers rerun, the base is cached.
- There is no CI build. Building PyTorch from source for a GPU target needs more disk and hours
  than hosted runners allow; the images are built on the maintainer's machine and pushed.

### Build status

| Date | What | Result |
|---|---|---|
| 2026-08-31 | Runtime `Dockerfile` against the pre-existing hand-built base (the one 02-VERSIONS warned about) | **PASS.** Clone+patches+deps+compile 4.7 min at `MAX_JOBS=16` (32-core host), whole build ~5.5 min; image 27.5 GB (vs 34 GB hand-built). All build-time checks green. GPU smoke test: `torch.cuda.get_arch_list()` → `['gfx1030']`, fp16 matmul, `_rocm_C` bound, Triton 3.6.0 — on an RX 6700 XT (gfx1031) through the baked `HSA_OVERRIDE_GFX_VERSION`. |
| 2026-08-31 | `Dockerfile.rocm_base` from scratch (`--base`), then the runtime image FROM it | **PASS** (attempt 2). Attempt 1 died at flash-attention after 57 min — `setup.py` rejects `['gfx1030']`, upstream's RDNA strip assumes a CDNA-led list — so patch 0005 now skips FA/AITER/MORI for RDNA-only arch lists. Base: PyTorch stage 45 min, Triton 16 min, ~65 min total on 32 cores, **24.2 GB**, package-for-package identical to the hand-built base (torch `2.11.0+gitd0c8b1f`, torchvision `0.24.1+d801a34`, torchaudio `2.9.0+eaa9e4e`, Triton 3.6.0, amdsmi 26.2.2; no FA/AITER/MORI). Runtime FROM it: compile 5.8 min, **26.9 GB**, all build-time checks green, GPU smoke test passed (arch list, fp16 matmul, `_rocm_C`, `moe_wna16_gemm`, plugins import). |

| 2026-08-31 | Serving test: `builds/btbtyler09-…/serve.sh` (TP=2, MTP=2, int8 KV) from the published image | **PASS.** Cold boot 703 s (warm 171 s); `verify/validate.py` 8/8 byte-identical to the recorded RCCL baseline, spot-checks pass; plugins engaged. Decode 27–29 t/s flat across 3.5k–42k ctx, prefill 382–701 t/s — below BUILD.md's table because the *host* has changed since those numbers: cards now power-capped to 170 W (pinned at the cap during decode; recorded era 232 W, ~180 W draw) and the TP pair's root links verified x8 Gen3 (all-reduce is PCIe-bound). Same-day A/B on the same host: the newest hand-built image (v7) scores the same (24–27 t/s); the pre-plugin Aug-18 image collapses to 3–17 t/s with the stock context slope. The published image reproduces the as-built stack; the delta vs the recorded table is environmental. |

(Updated by the maintainer after each build.)

## Publishing (maintainer)

```bash
# A classic PAT with write:packages (and read:packages); repo scope alone cannot push.
echo "$GHCR_TOKEN" | docker login ghcr.io -u leapdragon --password-stdin
./containers/build.sh --push                         # runtime + latest
./containers/build.sh --base --push                  # base too
./containers/build.sh --push --asbuilt vllm-gfx1030:0.27.1-patched-v7   # optional as-built snapshot
```

The first push creates the package **private**; make it public and confirm it is linked to this
repository under *Package settings* (the `org.opencontainers.image.source` label does the linking).
The ROCm layer alone is 21.5 GB in a single blob — pushes need a connection that can sustain that.

## Licences

The recipe's own material in the image (patches, plugins, scripts, docs) is GPL-3.0-or-later.
vLLM is Apache-2.0; PyTorch BSD-3-Clause; ROCm components MIT/others — see
`org.opencontainers.image.licenses` and the upstream projects. Model weights are **not** in the
image; they download into the mounted Hugging Face cache under their own licences.
