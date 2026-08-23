#!/usr/bin/env python3
# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
"""Convert the Intel 122B auto-round checkpoint config to vLLM-loadable GPTQ.

The tensors are already GPTQ-packed (packing_format "auto_round:auto_gptq");
only the config metadata is foreign to vLLM 0.27.1, which has no "auto-round"
quant method. This rewrites the snapshot's config.json in place:

  quant_method: "auto-round"  ->  "gptq"  (vLLM's AutoGPTQConfig then takes
  over via override_quantization_method and reads autoround_version)

  extra_config per-layer table  ->  GPTQ "dynamic" regex overrides:
    {"bits": 16, "data_type": "fp"}  ->  "-:" skip-quantization entries
    {"bits": 8}                      ->  "+:" {"bits": 8} entries

Patterns target vLLM's FUSED module names (in_proj_qkv+in_proj_z ->
in_proj_qkvz, in_proj_b+in_proj_a -> in_proj_ba) and match by suffix so the
serving-side root prefix doesn't matter. The exporter's choices are
fusion-consistent (verified: both qkvz halves int8 on the same layer, both
ba halves fp16 everywhere), so the fused-name translation is lossless.

Idempotent; safe to re-run. Re-run after any re-download of the snapshot.
"""
import glob
import json
import re
import sys

import os

CACHE = os.environ.get(
    "HF_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hf-cache"),
) + "/hub"
SNAP = glob.glob(
    f"{CACHE}/models--Intel--Qwen3.5-122B-A10B-int4-AutoRound/snapshots/*"
)

# The MTP head ships dense bf16 (outside the exporter's quantize list), but the
# global gptq quant_method would make vLLM build the draft's modules quantized
# and then fail loading dense tensors into them. vLLM's Qwen3.5 MTP model has a
# purpose-built escape: any "-:" dynamic key containing "mtp" makes it build
# the whole MTP layer stack unquantized, and the draft's fc layer (prefix
# "mtp.fc", built outside that branch) is unquantized by this same pattern
# through the ordinary dynamic-skip path. re.match anchors at the string start,
# so "mtp\." can never touch the target model's "model.*" / "lm_head" modules.
MTP_SKIP = r"-:mtp\..*"

FUSE = {"in_proj_qkv": "in_proj_qkvz", "in_proj_z": "in_proj_qkvz",
        "in_proj_b": "in_proj_ba", "in_proj_a": "in_proj_ba",
        # shared expert: vLLM fuses gate+up into gate_up_proj
        "gate_proj": "gate_up_proj", "up_proj": "gate_up_proj"}


def to_pattern(name: str) -> str:
    """Checkpoint module name -> suffix regex against vLLM module names."""
    parts = name.split(".")
    if parts[-1] in FUSE:
        parts[-1] = FUSE[parts[-1]]
    # keep from "layers.N" onward when present, else the last two components
    if "layers" in parts:
        parts = parts[parts.index("layers"):]
        return r".*\." + r"\.".join(re.escape(p) for p in parts) + "$"
    # no layers.N segment (e.g. "mtp.fc"): don't require a leading dot, the
    # serving-side prefix may be exactly this name
    parts = parts[-2:]
    return r".*" + r"\.".join(re.escape(p) for p in parts) + "$"


def _mtp_is_packed() -> bool:
    """True once quantize_mtp.py has int4-packed the MTP head (the skip must
    then stay absent so vLLM builds the draft quantized)."""
    idx = json.load(open(f"{SNAP[0]}/model.safetensors.index.json"))
    return any(k.startswith("mtp.") and k.endswith(".qweight")
               for k in idx["weight_map"])


def main() -> None:
    if not SNAP:
        sys.exit("snapshot not downloaded yet")
    cfg_path = f"{SNAP[0]}/config.json"
    cfg = json.load(open(cfg_path))
    q = cfg["quantization_config"]
    if q.get("quant_method") == "gptq" and "dynamic" in q:
        if MTP_SKIP not in q["dynamic"] and not _mtp_is_packed():
            q["dynamic"][MTP_SKIP] = {}
            json.dump(cfg, open(cfg_path, "w"), indent=2)
            print("already converted; added MTP skip pattern")
        else:
            print("already converted")
        return
    assert q["quant_method"] == "auto-round" and q["sym"] is True

    dynamic: dict[str, dict] = {}
    for name, override in q["extra_config"].items():
        pat = to_pattern(name)
        if override.get("bits") == 16:
            existing = dynamic.setdefault("-:" + pat, {})
            assert existing == {}
        elif override.get("bits") == 8:
            entry = dynamic.setdefault("+:" + pat, {"bits": 8})
            assert entry == {"bits": 8}
        else:
            sys.exit(f"unhandled override {name}: {override}")
    # this checkpoint leaves the whole MTP head out of block_name_to_quantize,
    # so no mtp entries come from extra_config; the skip is our addition (see
    # MTP_SKIP above) so vLLM builds the draft unquantized for MTP-under-PP
    dynamic[MTP_SKIP] = {}

    cfg["quantization_config"] = {
        "quant_method": "gptq",
        "bits": q["bits"],
        "group_size": q["group_size"],
        "sym": True,
        "desc_act": False,
        "lm_head": False,
        "autoround_version": q.get("autoround_version", ""),
        "dynamic": dynamic,
    }
    json.dump(cfg, open(cfg_path, "w"), indent=2)
    neg = sum(1 for k in dynamic if k.startswith("-:"))
    pos = len(dynamic) - neg
    print(f"converted: {pos} int8 overrides, {neg} skip patterns "
          f"({len(q['extra_config'])} source entries)")


if __name__ == "__main__":
    main()
