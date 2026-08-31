#!/bin/bash
# recipe-serve — entrypoint of the vllm-rdna2-recipe image.
#
#   docker run … IMAGE <model-tag> [vllm serve flags]     exactly `vllm serve …` (as before)
#   docker run … IMAGE preset:<name> [extra vllm flags]   a build's tuned configuration
#   docker run … IMAGE list-presets                       what <name> can be
#
# Preset knobs (env): MTP=<0|1|2|3> (default per preset), PORT=<n> (default 8000; pass
# -e HEALTHCHECK_PORT=<n> too — the healthcheck cannot see this process's exports),
# DRYRUN=1 (print the resolved env + command and exit). Extra flags are appended after
# the preset's, and vLLM's argparse lets the last occurrence win.
#
# Device selection is standard ROCm: -e ROCR_VISIBLE_DEVICES=1,3 (plus the /dev/kfd + /dev/dri
# mounts). Mounted cache dirs are picked up when present: /tuning (TunableOp CSVs — seeded from
# the preset's shipped rows if empty; see TROUBLESHOOTING 5c), /compile-cache, /triton-cache,
# /ext-cache.
set -euo pipefail
BUILDS=/app/recipe/builds

list_presets() {
  echo "presets (docker run … IMAGE preset:<name>):"
  for f in "$BUILDS"/*/preset.env; do
    [ -f "$f" ] || continue
    name=$(sed -n 's/^PRESET_NAME="\{0,1\}\([^"]*\)"\{0,1\}$/\1/p' "$f" | head -1)
    note=$(sed -n 's/^# summary: //p' "$f" | head -1)
    printf "  %-24s %s\n" "$name" "$note"
  done
}

case "${1:-}" in
  list-presets) list_presets; exit 0 ;;
  preset:*) PRESET="${1#preset:}"; shift ;;
  *) exec vllm serve "$@" ;;
esac

PRESET_DIR=""
for f in "$BUILDS"/*/preset.env; do
  [ -f "$f" ] || continue
  if grep -q "^PRESET_NAME=\"\{0,1\}$PRESET\"\{0,1\}$" "$f" || [ "$(basename "$(dirname "$f")")" = "$PRESET" ]; then
    PRESET_DIR="$(dirname "$f")"; break
  fi
done
if [ -z "$PRESET_DIR" ]; then
  echo "recipe-serve: unknown preset '$PRESET'" >&2; list_presets >&2; exit 2
fi

# shellcheck disable=SC1091
source "$PRESET_DIR/preset.env"
if [ -n "${PRESET_UNSUPPORTED:-}" ]; then
  echo "recipe-serve: preset '$PRESET' cannot run turnkey: $PRESET_UNSUPPORTED" >&2; exit 2
fi
[ -n "${PRESET_NOTE:-}" ] && echo "recipe-serve: NOTE: $PRESET_NOTE" >&2

# TunableOp: always on (lookup-only unless TUNEOP_TUNING=1); seed /tuning from the shipped rows.
mkdir -p /tuning
if [ -n "${CSV_DIR:-}" ] && ! ls /tuning/tunableop_results*.csv >/dev/null 2>&1; then
  if ls "$CSV_DIR"/tunableop_results*.csv >/dev/null 2>&1; then
    cp "$CSV_DIR"/tunableop_results*.csv /tuning/ && echo "recipe-serve: seeded /tuning from $CSV_DIR (load-bearing lm_head rows — TROUBLESHOOTING 5c)" >&2
  else
    echo "recipe-serve: WARNING: no TunableOp CSVs shipped for this preset and /tuning is empty — decode will run ~40% slow (TROUBLESHOOTING 5c)" >&2
  fi
fi
export PYTORCH_TUNABLEOP_ENABLED=1
export PYTORCH_TUNABLEOP_TUNING="${TUNEOP_TUNING:-0}"
export PYTORCH_TUNABLEOP_HIPBLASLT_ENABLED=0
export PYTORCH_TUNABLEOP_FILENAME=/tuning/tunableop_results.csv

# Mounted caches, when present.
[ -d /compile-cache ] && export VLLM_CACHE_ROOT=/compile-cache
[ -d /triton-cache ] && export TRITON_CACHE_DIR=/triton-cache
[ -d /ext-cache ] && export TORCH_EXTENSIONS_DIR=/ext-cache

export GPU_MAX_HW_QUEUES="${HW_QUEUES:-4}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
export PYTORCH_ALLOC_CONF="${PYTORCH_ALLOC_CONF:-expandable_segments:True}"
export VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS="${VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS:-300}"
export VLLM_LOG_STATS_INTERVAL="${VLLM_LOG_STATS_INTERVAL:-10}"

MTP="${MTP:-${MTP_DEFAULT:-2}}"
SPEC_ARG=()
[ "$MTP" != "0" ] && SPEC_ARG=(--speculative-config "{\"method\":\"qwen3_5_mtp\",\"num_speculative_tokens\":$MTP}")

# shellcheck disable=SC2086  # PRESET_ARGS is a deliberate word-split of space-free tokens
CMD=(vllm serve "$MODEL" --served-model-name "$SERVED" --port "${PORT:-8000}" $PRESET_ARGS "${SPEC_ARG[@]}" "$@")
if [ "${DRYRUN:-0}" = "1" ]; then
  echo "# resolved preset '$PRESET' from $PRESET_DIR"
  env | grep -E "^(FD_|AR_|VLLM_|PYTORCH_|GPU_MAX|OMP_|ROCR_|HSA_)" | sort
  printf '%q ' "${CMD[@]}"; echo
  exit 0
fi
exec "${CMD[@]}"
