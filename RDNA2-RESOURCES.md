# The wider gfx1030 / RDNA2 universe

**Copyright © 2026 Aron Hsiao. Licensed under the GNU General Public License v3.0 or later.**

Companion to [README.md](README.md) — the recipe itself lives there. This file is the map of
everyone *else* working on this hardware.

When I started this I thought nobody was doing RDNA2 inference seriously. That was wrong — there's an
active community, and several of these projects solve problems this repo doesn't (or solve them
differently and better). Linking generously so you don't have to rediscover them the way I did.
If you're landing here from a search engine at 2am with four cheap AMD cards, start with the wiki.

### Start here

| Project | What it is / why you'd want it |
|---|---|
| **[Wiki GFX1030](https://blivioniag.github.io/gfx1030-wiki/)** | The community hub for gfx1030: hardware list, ROCm install, env-var cheat sheets, power tuning, PCIe P2P, vLLM and llama.cpp guides, troubleshooting. The single best orientation point for this hardware. |
| **[The wiki's own resources page](https://blivioniag.github.io/gfx1030-wiki/meta/resources.html)** | Its curated link list, including the gfx1030 Discord where much of the material originates (`#vllm-rdna`, `#llamacpp` channels) and a verification-status page auditing which claims are reproduced vs community-reported. |

### Other people running LLMs on these cards

| Project | What it is |
|---|---|
| **[blivioniag/vllm @ `rdna2_extras`](https://github.com/blivioniag/vllm/tree/rdna2_extras)** | A vLLM fork with hand-written RDNA2 **HIP** kernels — FlashAttention (`fa_rdna2`, head_size 256), W4A16 GEMM (decode + prefill), MoE expert GEMMs, FP8 paths, and a complete hand-ported **GDN / gated-delta-net chain**. Where this recipe uses Triton plugins and targeted patches, they write HIP. If you want native kernels without patching anything yourself, start here. |
| **[blivioniag/vllm-rdna (Docker Hub)](https://hub.docker.com/r/blivioniag/vllm-rdna)** | Prebuilt vLLM images for seven RDNA archs, including `-extras` tags built from the fork above. `blivioniag/rocm-rdna` is the matching ROCm + PyTorch base — useful on its own if you just want working `torch` on a Radeon. |
| **[blivioniag/vllm-rdna-docker](https://github.com/blivioniag/vllm-rdna-docker)** | The build system behind those images (`docker-bake.hcl`, RDNA patches, CI). Read it if you want to build your own variants. |
| **[edwinbrowwn/llama.cpp-rdna2](https://github.com/edwinbrowwn/llama.cpp-rdna2)** | A llama.cpp fork with serious RDNA2/V620 work: MMQ/MMVQ kernels, native RDNA2 FlashAttention, tensor-parallel RCCL tuning and P2P all-reduce schedules, graph fusions, DFlash2/MTP speculative decoding. Developed on 4× V620. If you're on GGUF rather than vLLM, this is the fork to use. Their docs (`docs/gfx1030-*`, `docs/rdna2-*`) are worth reading even if you never build it. |
| **[sebastianmechno-sys/vllm-rocm-windows-rdna2](https://github.com/sebastianmechno-sys/vllm-rocm-windows-rdna2)** | vLLM on RDNA2 under Windows. |
| **[skyne98/wiki-gfx906](https://github.com/skyne98/wiki-gfx906)** | Sibling wiki for gfx906 (Vega 20 / MI50) — different silicon, same spirit of making unsupported cards work. |

### Hardware tuning (V620-specific)

| Project | What it is |
|---|---|
| **[blivioniag/v620_toolbox](https://github.com/blivioniag/v620_toolbox)** | Two things this repo doesn't do. (1) **Power tuning**: the V620's VBIOS declares a 250 W *minimum* power limit, but the SMU accepts down to 120 W — a small, PCI-ID-gated kernel patch unlocks it (Fedora and Ubuntu 26.04 paths, boot-cap service, verification scripts). (2) **PCIe P2P**: kernel config deltas and readiness/verification scripts for GPU↔GPU peer-to-peer, validated 12/12 on 4× V620. |

### Upstream and vendor

| Link | Why |
|---|---|
| [vLLM](https://github.com/vllm-project/vllm) · [llama.cpp](https://github.com/ggml-org/llama.cpp) | The upstreams everything here is built on. |
| [vLLM PR #52391](https://github.com/vllm-project/vllm/pull/52391) | RDNA gfx1030 platform detection — upstream CI enablement for consumer Radeon; the ancestry of this repo's patch 0001. |
| [ROCm/TheRock](https://github.com/ROCm/TheRock) | ROCm build system, including gfx103X enablement work. |
| [ROCm Device Support Wishlist](https://github.com/ROCm/ROCm/discussions/4276) | Community-tracked support matrix — where to register that you care about a GPU family. |
| [ROCm docs](https://rocm.docs.amd.com/) · [system requirements](https://rocm.docs.amd.com/projects/install-on-linux/en/latest/reference/system-requirements.html) · [GPU architecture specs](https://rocm.docs.amd.com/en/latest/reference/gpu-arch-specs.html) | Official documentation. Navi 21 = gfx1030. |

### Where this repo fits

This one is a **recipe book against pristine vLLM 0.27.1** — patches with stated intent, per-model
build documents, measured numbers, and the traps that look like other problems. It is aimed at
someone handing the whole thing to an LLM and saying "now do this for my model." If instead you
want prebuilt images and native HIP kernels, use the `-extras` images above; if you want GGUF, use
the llama.cpp fork. These projects overlap and disagree in places — where they disagree with what's
written here, believe whoever actually measured it on hardware like yours, and A/B it yourself.
