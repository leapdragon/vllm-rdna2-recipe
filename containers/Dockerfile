# syntax=docker/dockerfile:1
# ---------------------------------------------------------------------------------------------
# vllm-rdna2-recipe — runtime image
#
#   pristine vLLM 0.27.1  +  patches/0001–0009  +  plugins fd_rdna2 / ar_rdna2
#   on the recipe's base (containers/Dockerfile.rocm_base: ROCm 7.2.3, PyTorch 2.11 built from
#   source for gfx1030).  Everything compiled here is compiled for gfx1030 and verified at build.
#
# Build context is the repository root:        ./containers/build.sh        (see containers/README.md)
# Provenance inside the image:                 /app/versions.txt
# The patched tree is an editable install at   /app/vllm-src   — `git diff --stat` there is exactly
#   the nine patches against the tag, and config/serve-rdna2-tp2.sh can overlay files onto it.
# ---------------------------------------------------------------------------------------------
ARG BASE_IMAGE=ghcr.io/leapdragon/vllm-rdna2-recipe-base:rocm7.2.3-torch2.11.0-gfx1030
FROM ${BASE_IMAGE}
ARG BASE_IMAGE

ARG VLLM_REPO=https://github.com/vllm-project/vllm.git
ARG VLLM_REF=v0.27.1
ARG VLLM_COMMIT=6e448d0ea9bf3d88d898b65449ca6dc2aec170ac
ARG VLLM_VERSION=0.27.1
ARG PYTORCH_ROCM_ARCH=gfx1030
ARG MAX_JOBS=16

ENV DEBIAN_FRONTEND=noninteractive \
    PIP_ROOT_USER_ACTION=ignore \
    PIP_NO_CACHE_DIR=1
WORKDIR /app

# 1. Pristine vLLM at the pinned tag. The tag must resolve to the recorded commit — a moved tag
#    would silently change what the patches apply to.
RUN git clone --depth 1 --branch "${VLLM_REF}" "${VLLM_REPO}" /app/vllm-src \
 && cd /app/vllm-src \
 && head="$(git rev-parse HEAD)" \
 && { [ "$head" = "${VLLM_COMMIT}" ] \
      || { echo "ERROR: ${VLLM_REF} resolves to $head, expected ${VLLM_COMMIT}" >&2; exit 1; }; }

# 2. The recipe's patches, in order (01-PATCHES.md). 0005 only edits docker/Dockerfile.rocm_base —
#    the base image's own recipe — and is applied anyway so the tree equals the documented one.
COPY patches/ /app/recipe/patches/
RUN cd /app/vllm-src \
 && for p in /app/recipe/patches/000?-*.patch; do \
        echo "==> $(basename "$p")"; git apply --whitespace=nowarn "$p" || exit 1; \
    done \
 && git diff --stat | tail -1

# 3. Constraints: while resolving vLLM's requirements, pip must never replace the source-built
#    torch / torchvision / torchaudio / triton with PyPI wheels (which carry no gfx1030 code).
RUN python3 - <<'EOF' > /tmp/constraints.txt
import importlib.metadata as m
for p in ("torch", "torchvision", "torchaudio", "triton"):
    try:
        print(f"{p}=={m.version(p)}")
    except m.PackageNotFoundError:
        pass
EOF
RUN cat /tmp/constraints.txt && grep -q '^torch==' /tmp/constraints.txt

# 4. Dependencies, then every vLLM extension compiled for gfx1030 (~10–20 min at MAX_JOBS=16).
#    vLLM takes its version from setuptools-scm; with the patches applied the tree is "dirty" and
#    would report 0.27.2.dev0+g…d<date>.rocm723 — a version that changes daily and is part of the
#    torch.compile cache key. VLLM_VERSION_OVERRIDE (setup.py's own hook) pins it to exactly 0.27.1.
ENV VLLM_TARGET_DEVICE=rocm \
    PYTORCH_ROCM_ARCH=${PYTORCH_ROCM_ARCH} \
    ROCM_HOME=/opt/rocm \
    TORCH_BLAS_PREFER_HIPBLASLT=0 \
    TORCHINDUCTOR_COMPILE_THREADS=1
RUN cd /app/vllm-src \
 && (pip uninstall -y vllm >/dev/null 2>&1 || true) \
 && pip install -c /tmp/constraints.txt -r requirements/rocm.txt \
 && MAX_JOBS="${MAX_JOBS}" VLLM_VERSION_OVERRIDE="${VLLM_VERSION}" \
    pip install -c /tmp/constraints.txt --no-build-isolation -e . \
 && { echo "vllm=$(python3 -c 'import vllm; print(vllm.__version__)')"; \
      echo "vllm_commit=${VLLM_COMMIT}"; \
      echo "recipe_patches=$(cd /app/recipe/patches && ls *.patch | tr '\n' ' ')"; \
      echo "torch=$(python3 -c 'import torch; print(torch.__version__, "hip", torch.version.hip)')"; } \
    >> /app/versions.txt

# 5. Build-time verification — the checks the recipe's build phases asserted, consolidated. None of
#    them needs a GPU. The arch check reads the code-object target IDs out of the binaries, because
#    torch.cuda.get_arch_list() returns [] with no device attached and tells you nothing.
RUN cd /app/vllm-src && python3 - <<'EOF'
import ctypes, glob, os, pathlib, subprocess
import torch, vllm, vllm._custom_ops  # noqa: F401  (binds _C, _moe_C, _rocm_C)

src = pathlib.Path("/app/vllm-src")
arch = os.environ["PYTORCH_ROCM_ARCH"]
target_id = f"amdgcn-amd-amdhsa--{arch}".encode()

def so(rel, floor):
    p = next(src.glob(rel), None)
    assert p is not None and p.stat().st_size > floor, (rel, p)
    return p

for rel, floor in (("vllm/_C.abi3.so", 1_000_000), ("vllm/_moe_C.abi3.so", 100_000),
                   ("vllm/_C_stable_libtorch.abi3.so", 10_000_000), ("vllm/_rocm_C.abi3.so", 1_000_000)):
    p = so(rel, floor)
    assert target_id in p.read_bytes(), f"{p.name} carries no {arch} code objects"
rocm_c = so("vllm/_rocm_C.abi3.so", 1_000_000).read_bytes()
assert b"moe_skinny_int4_decode" in rocm_c, "patch 0008 kernel missing from _rocm_C"
assert hasattr(torch.ops._moe_C, "moe_wna16_gemm"), "patch 0007 op not bound"
assert "on_gfx10x" in (src / "vllm/platforms/rocm.py").read_text(), "patch 0001 missing"
for f in ("vllm/v1/worker/gpu/pp_utils.py", "vllm/v1/worker/gpu/model_runner.py"):
    assert "broadcast_draft" in (src / f).read_text(), f"patch 0009 missing in {f}"

assert vllm.__version__ == os.environ["VLLM_VERSION"], vllm.__version__
assert torch.__version__.startswith("2.11"), torch.__version__
assert (torch.version.hip or "").startswith("7.2"), torch.version.hip
libtorch_hip = pathlib.Path(torch.__file__).parent / "lib" / "libtorch_hip.so"
assert subprocess.run(["grep", "-q", target_id.decode(), str(libtorch_hip)]).returncode == 0, \
    f"{libtorch_hip} was not built for {arch} (base image built with the wrong PYTORCH_ROCM_ARCH?)"

# Stock HSA runtime: earlier hand-built images carried a binary-patched libhsa; this one must not.
hsa = sorted(glob.glob("/opt/rocm-*/lib/libhsa-runtime64.so.1.*"))
assert hsa, "libhsa-runtime64 not found"
lib = ctypes.CDLL(hsa[-1])
assert hasattr(lib, "hsa_iterate_agents") and hasattr(lib, "hsa_amd_ais_file_write"), "HSA ABI incomplete"
if hsa[-1].endswith("1.18.70203"):
    with open(hsa[-1], "rb") as f:
        f.seek(0x1FFB0)
        b = f.read(1)
    assert b == b"\x00", f"libhsa is binary-patched (byte 0x1ffb0 = {b.hex()}), expected stock"

print(f"verified: vllm {vllm.__version__} on torch {torch.__version__} (HIP {torch.version.hip}); "
      f"extensions + libtorch_hip carry {arch} code; patches 0001/0007/0008/0009 present; stock libhsa")
EOF

# 6. The two plugins (vllm.general_plugins entry points). ar_rdna2 JIT-compiles its HIP source into
#    TORCH_EXTENSIONS_DIR on first use — mount that directory to keep the build across containers.
COPY builds/shared/plugins/fd_rdna2 /app/plugins/fd_rdna2
COPY builds/shared/plugins/ar_rdna2 /app/plugins/ar_rdna2
RUN pip install --no-deps /app/plugins/fd_rdna2 /app/plugins/ar_rdna2 \
 && python3 - <<'EOF'
import importlib.metadata as m, inspect, pathlib
import fd_rdna2, ar_rdna2
eps = {e.name for e in m.entry_points(group="vllm.general_plugins")}
assert {"fd_rdna2", "ar_rdna2"} <= eps, eps
assert "GQA = q.shape[1] // n_kv" in inspect.getsource(fd_rdna2), "fd_rdna2 GQA generalization missing"
assert (pathlib.Path(ar_rdna2.__file__).parent / "csrc" / "ar_ext.hip").exists(), "ar_rdna2 csrc not packaged"
print("plugins registered:", sorted({"fd_rdna2", "ar_rdna2"} & eps))
EOF

# 7. Reference material shipped with the image (small): the recipe's verification clients and docs.
COPY verify/ /app/recipe/verify/
COPY LICENSE README.md 02-VERSIONS.md TROUBLESHOOTING.md /app/recipe/

# 8. Runtime defaults for Navi 2x; every one is overridable with `docker run -e`.
#    HSA_OVERRIDE_GFX_VERSION lets the gfx1030 code objects run on the rest of the family
#    (gfx1031/1032/… — RX 6700/6600 class) and is a no-op on a real gfx1030.
#    hipBLASLt and AITER have no gfx1030 kernels; Triton flash-attention is the attention path.
ENV HSA_OVERRIDE_GFX_VERSION=10.3.0 \
    FLASH_ATTENTION_TRITON_AMD_ENABLE=TRUE \
    VLLM_ROCM_USE_AITER=0 \
    VLLM_ROCM_USE_AITER_MOE=0 \
    VLLM_WORKER_MULTIPROC_METHOD=spawn \
    HIP_FORCE_DEV_KERNARG=1 \
    HSA_NO_SCRATCH_RECLAIM=1 \
    TOKENIZERS_PARALLELISM=false \
    SAFETENSORS_FAST_GPU=1 \
    PYTORCH_NVML_BASED_CUDA_CHECK=1 \
    HEALTHCHECK_PORT=8000

ARG BUILD_DATE=unknown
ARG VCS_REF=unknown
ARG IMAGE_VERSION=0.27.1-rocm7.2.3-gfx1030
LABEL org.opencontainers.image.title="vLLM for AMD RDNA2 (gfx1030 / Radeon PRO V620)" \
      org.opencontainers.image.description="vLLM ${VLLM_VERSION} with the vllm-rdna2-recipe patches and plugins on ROCm 7.2.3; PyTorch built from source for ${PYTORCH_ROCM_ARCH}." \
      org.opencontainers.image.source="https://github.com/leapdragon/vllm-rdna2-recipe" \
      org.opencontainers.image.url="https://github.com/leapdragon/vllm-rdna2-recipe" \
      org.opencontainers.image.documentation="https://github.com/leapdragon/vllm-rdna2-recipe/blob/main/containers/README.md" \
      org.opencontainers.image.licenses="GPL-3.0-or-later AND Apache-2.0" \
      org.opencontainers.image.vendor="Aron Hsiao" \
      org.opencontainers.image.version="${IMAGE_VERSION}" \
      org.opencontainers.image.revision="${VCS_REF}" \
      org.opencontainers.image.created="${BUILD_DATE}" \
      org.opencontainers.image.base.name="${BASE_IMAGE}"

# A cold first boot compiles for 10–15 min (warm ~3–4 min with a persisted /compile-cache):
# the start period is sized for the cold case.
HEALTHCHECK --interval=30s --timeout=10s --start-period=30m --retries=3 \
  CMD python3 -c 'import os, urllib.request as u; u.urlopen("http://127.0.0.1:%s/health" % os.environ.get("HEALTHCHECK_PORT", "8000"), timeout=5)' || exit 1

WORKDIR /app
ENTRYPOINT ["vllm", "serve"]
CMD ["--help"]
