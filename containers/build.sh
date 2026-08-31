#!/usr/bin/env bash
# containers/build.sh — build, tag and (optionally) push the recipe's container images.
#
#   ./containers/build.sh                  runtime image, FROM the published base image
#   ./containers/build.sh --base           first rebuild the PyTorch/ROCm base from scratch (hours)
#   ./containers/build.sh --push           push what this run built (`docker login ghcr.io` first)
#   ./containers/build.sh --asbuilt IMG    also tag (and push) a local image as "<VERSION>-asbuilt"
#
# Environment overrides:
#   REGISTRY_REPO   ghcr.io/leapdragon/vllm-rdna2-recipe      image repository (base: "<repo>-base")
#   VERSION         0.27.1-rocm7.2.3-gfx1030                  runtime tag; "latest" is added too
#   BASE_TAG        rocm7.2.3-torch2.11.0-gfx1030             base tag
#   BASE_IMAGE      ${REGISTRY_REPO}-base:${BASE_TAG}         FROM for the runtime image
#   PYTORCH_ROCM_ARCH gfx1030                                 arch list for both builds
#   MAX_JOBS        16                                        parallelism of the vLLM extension compile
set -euo pipefail
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
root="$(cd "$here/.." && pwd)"

REGISTRY_REPO="${REGISTRY_REPO:-ghcr.io/leapdragon/vllm-rdna2-recipe}"
VERSION="${VERSION:-0.27.1-rocm7.2.3-gfx1030}"
BASE_TAG="${BASE_TAG:-rocm7.2.3-torch2.11.0-gfx1030}"
BASE_IMAGE="${BASE_IMAGE:-${REGISTRY_REPO}-base:${BASE_TAG}}"
PYTORCH_ROCM_ARCH="${PYTORCH_ROCM_ARCH:-gfx1030}"
MAX_JOBS="${MAX_JOBS:-16}"
SOURCE_URL="https://github.com/leapdragon/vllm-rdna2-recipe"

do_base=0; do_push=0; asbuilt=""
while [ $# -gt 0 ]; do
  case "$1" in
    --base) do_base=1 ;;
    --push) do_push=1 ;;
    --asbuilt) asbuilt="${2:?--asbuilt needs a local image name}"; shift ;;
    -h|--help) sed -n '2,16p' "$0" | sed 's/^# \{0,1\}//'; exit 0 ;;
    *) echo "unknown argument: $1" >&2; exit 2 ;;
  esac
  shift
done

export DOCKER_BUILDKIT=1
vcs_ref="$(git -C "$root" rev-parse --short HEAD 2>/dev/null || echo unknown)"
build_date="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
built=()
log() { printf '\n==> %s\n' "$*"; }

if [ "$do_base" = 1 ]; then
  log "base image $BASE_IMAGE — PyTorch from source for $PYTORCH_ROCM_ARCH (hours, all cores)"
  docker buildx build --load --progress=plain \
    -f "$here/Dockerfile.rocm_base" --target final \
    --build-arg "PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH" \
    --label "org.opencontainers.image.title=PyTorch/ROCm base for vllm-rdna2-recipe ($PYTORCH_ROCM_ARCH)" \
    --label "org.opencontainers.image.source=$SOURCE_URL" \
    --label "org.opencontainers.image.revision=$vcs_ref" \
    --label "org.opencontainers.image.created=$build_date" \
    --label "org.opencontainers.image.licenses=Apache-2.0 AND BSD-3-Clause AND MIT" \
    -t "$BASE_IMAGE" "$here"
  built+=("$BASE_IMAGE")
fi

log "runtime image $REGISTRY_REPO:$VERSION (FROM $BASE_IMAGE, MAX_JOBS=$MAX_JOBS)"
docker buildx build --load --progress=plain \
  -f "$here/Dockerfile" \
  --build-arg "BASE_IMAGE=$BASE_IMAGE" \
  --build-arg "PYTORCH_ROCM_ARCH=$PYTORCH_ROCM_ARCH" \
  --build-arg "MAX_JOBS=$MAX_JOBS" \
  --build-arg "BUILD_DATE=$build_date" \
  --build-arg "VCS_REF=$vcs_ref" \
  --build-arg "IMAGE_VERSION=$VERSION" \
  -t "$REGISTRY_REPO:$VERSION" -t "$REGISTRY_REPO:latest" \
  "$root"
built+=("$REGISTRY_REPO:$VERSION" "$REGISTRY_REPO:latest")

if [ -n "$asbuilt" ]; then
  log "as-built snapshot: $asbuilt -> $REGISTRY_REPO:$VERSION-asbuilt"
  docker tag "$asbuilt" "$REGISTRY_REPO:$VERSION-asbuilt"
  built+=("$REGISTRY_REPO:$VERSION-asbuilt")
fi

if [ "$do_push" = 1 ]; then
  for t in "${built[@]}"; do log "push $t"; docker push "$t"; done
fi

printf '\nbuilt%s:\n' "$([ "$do_push" = 1 ] && echo ' and pushed')"
printf '  %s\n' "${built[@]}"
