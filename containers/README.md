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

## Getting started — pull, serve, query

What the host needs — and, deliberately, what it does not:

- A **Navi 2x GPU** with enough VRAM for the model (the TP=2 27B presets want two 32 GB V620s;
  the baked `HSA_OVERRIDE_GFX_VERSION=10.3.0` lets the rest of the family — RX 6700/6800/6900
  class — run the same code objects). The `amdgpu` kernel driver (in any recent mainline kernel)
  and Docker. **No ROCm install on the host** — the image carries the entire userspace, and
  mounting a host ROCm into it is the #1 way to break it ([TROUBLESHOOTING](../TROUBLESHOOTING.md)).
- Disk: ~27 GB for the image, ~15 GB for the reference model's weights (downloaded into the
  mounted Hugging Face cache on first run).

```bash
docker pull ghcr.io/leapdragon/vllm-rdna2-recipe:0.27.1-rocm7.2.3-gfx1030
docker run --rm ghcr.io/leapdragon/vllm-rdna2-recipe:0.27.1-rocm7.2.3-gfx1030 list-presets

docker run -d --name vllm-rdna2 --network=host \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" \
  --ipc=host --ulimit memlock=-1 --security-opt seccomp=unconfined \
  -e ROCR_VISIBLE_DEVICES=0,1 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  ghcr.io/leapdragon/vllm-rdna2-recipe:0.27.1-rocm7.2.3-gfx1030 preset:qwen38-27b-gptq
```

First boot is the slow one: the weights download (~15 GB), then vLLM cold-compiles for **10–15
minutes** before the server answers — watch `docker logs -f vllm-rdna2`, wait for
`docker inspect --format '{{.State.Health.Status}}' vllm-rdna2` to say `healthy`, or poll
`curl -s localhost:8000/health`. (Persist `/compile-cache` as shown under Presets and the next
boot takes ~3 minutes.) Then it is a standard OpenAI-compatible server:

```bash
curl -s http://localhost:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{
  "model": "qwen38-27b-gptq",
  "messages": [{"role": "user", "content": "Say hello in five words."}],
  "max_tokens": 50}'
```

Set `ROCR_VISIBLE_DEVICES` to the two cards you want (indices as `rocm-smi` lists them). Expect
~40–49 tokens/s decode at temperature-0 quality identical to the recorded baseline; if you see a
flat ~27 t/s instead, read [TROUBLESHOOTING 5c](../TROUBLESHOOTING.md).

## Run it

The recommended path is the recipe's own wrapper, which carries the tuned flags, mounts, and
plugin knobs that every measured number used — point it at the published image:

```bash
git clone https://github.com/leapdragon/vllm-rdna2-recipe && cd vllm-rdna2-recipe
IMG=ghcr.io/leapdragon/vllm-rdna2-recipe:0.27.1-rocm7.2.3-gfx1030 \
  ./builds/btbtyler09-Qwen3.8-27B-GPTQ-4bit/serve.sh      # or any builds/*/serve.sh
```

### Presets — one line, no repo clone

The image carries each build's tuned configuration and its load-bearing TunableOp rows
(`/app/recipe/builds/*/preset.env`), driven by the entrypoint:

```bash
docker run --rm IMAGE list-presets                # what <name> can be
docker run -d --name vllm-rdna2 --network=host \
  --device /dev/kfd --device /dev/dri \
  --group-add "$(getent group render | cut -d: -f3)" --group-add "$(getent group video | cut -d: -f3)" \
  --ipc=host --ulimit memlock=-1 --security-opt seccomp=unconfined \
  -e ROCR_VISIBLE_DEVICES=1,3 \
  -v "$HOME/.cache/huggingface:/root/.cache/huggingface" \
  IMAGE preset:qwen38-27b-gptq
```

- **Model**: `preset:<name>` picks a supported build (`qwen38-27b-gptq` is the reference; the AWQ
  and MixedInt4 27B siblings are presets too; the 122B refuses with instructions — it needs
  one-time weight conversions and the repo wrapper).
- **Devices**: `-e ROCR_VISIBLE_DEVICES=<i,j>` — standard ROCm device selection; TP=2 presets want
  exactly two.
- **Knobs**: `-e MTP=0..3` (speculative depth; default 2), `-e PORT=<n>` (add
  `-e HEALTHCHECK_PORT=<n>` so the container healthcheck follows), `-e DRYRUN=1` (print the
  resolved env + full `vllm serve` command and exit), extra vLLM flags appended after the preset
  name override the preset's (`… preset:qwen38-27b-gptq --max-model-len 32768`).
- **TunableOp seeding**: with no `/tuning` mount the shipped per-rank CSVs are seeded
  automatically (the rows worth ~1.7× decode — [TROUBLESHOOTING 5c](../TROUBLESHOOTING.md));
  mount `-v vllm-rdna2-tuning:/tuning` to persist, and the same automatic seeding fills an empty
  mount. Mounted `/compile-cache`, `/triton-cache`, `/ext-cache` are picked up when present —
  without them the first boot cold-compiles (~12 min) every time.
- A plain model tag instead of `preset:` behaves exactly as before: the arguments go verbatim to
  `vllm serve`.

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

- The entrypoint is a thin shim over `vllm serve` (`preset:<name>` expands a build's tuned
  configuration; anything else passes through verbatim); everything after the image name is its arguments
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
| `/app/recipe/{patches,verify,builds,…}` | The patches, the verification clients (`verify/validate.py`, `decode-rate.py`, …), the key docs, and the per-build presets (`builds/*/preset.env` + their TunableOp CSVs) used by `preset:<name>`. |
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

| 2026-08-31 | Serving test: `builds/btbtyler09-…/serve.sh` (TP=2, MTP=2, int8 KV) from the published image | **PASS.** Cold boot 703 s (warm 171 s); `verify/validate.py` 8/8 byte-identical to the recorded RCCL baseline, spot-checks pass; plugins engaged. Decode 27–29 t/s flat across 3.5k–42k ctx, prefill up to 759 t/s — below BUILD.md's decode table because the *host* changed after those numbers were recorded (cause found the same day — see the last row: the lost TunableOp lm_head rows). Same-day A/B: v7 hand-built = this image (24–29 t/s); v3 — the image the numbers were measured on — also 27–31 t/s; the pre-plugin Aug-18 image collapses to 3–17 t/s with the stock context slope. The published image reproduces the as-built stack faithfully. |

| 2026-08-31 | Pair test: same serve on the other V620 pair (`DEVICES=2,4`, x16 Gen3 root links vs the default pair's x8 Gen3) | **PASS after one real finding.** First boot crashed with an aperture violation — the shared compile cache from the 1,3 runs is device-set-specific (now [TROUBLESHOOTING 5b](../TROUBLESHOOTING.md), and the wrapper scopes the cache per `DEVICES`). With fresh caches: 8/8 outputs identical, decode 22–28 t/s — same as the x8 pair within noise, so link width is not the decode bottleneck. |
| 2026-08-31 | Cap test (170 W → 220 W) and provenance test (v3, the exact image behind BUILD.md's numbers) | Decode unchanged at 220 W (cards draw 208–214 W for the same 25–28 t/s — bandwidth-bound, exactly as 02-VERSIONS says: the cap is NOT the delta). v3 on today's host: 27–31 t/s, outputs identical. Both hypotheses died the same day; see the next row. Prefill unaffected (759 t/s @7.5k). |
| 2026-08-31 | **Regression found and fixed: the tuned TunableOp lm_head rows had been lost.** A torch profile showed three ~10.5 ms full-shard lm_head GEMMs per MTP step (`[1..3,5120]×[5120,124160]` fp16 at 115 GB/s — rocBLAS's heuristic tile); a micro-benchmark reproduced it exactly and TunableOp tuning recovered 2.9–3.5 ms (360–430 GB/s). After an offline tuning session (17 min warmup + decode traffic), lookup-only from the published image measures **49.0 / 45.3 / 41.8 t/s** at 3.5k/13k/42k — the recorded table, reproduced, outputs 8/8 identical. CSVs now shipped in `builds/btbtyler09-…/tunableop/`; symptom documented as TROUBLESHOOTING 5c. The Aug-25 stability cmdline, caps, links and queues were all exonerated. |

| 2026-08-31 | Preset acceptance: bare `docker run … preset:qwen38-27b-gptq` with `ROCR_VISIBLE_DEVICES=1,3` and only the HF-cache mount (no repo clone, no `/tuning`, no compile caches) | **PASS.** The shim resolved the tuned configuration (verified against `/proc/1/environ`), auto-seeded the shipped TunableOp rows into `/tuning`, cold boot 813 s, outputs 8/8 byte-identical, decode 36.5–48.5 t/s across 3.4k–41k — single-boot spread around the recorded band, clearly the tuned signature (untuned reads a flat ~27). |

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
