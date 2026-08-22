# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
"""Convert the Pilcothink auto-round checkpoint config to vLLM-loadable GPTQ.

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
import os
import re
import sys

# HF hub cache: env override first, then the standard location (which is also
# where the recipe launcher's HF_CACHE mount appears inside the container).
CACHE = os.environ.get("HF_HUB_CACHE") or os.path.expanduser(
    "~/.cache/huggingface/hub")
SNAP = glob.glob(
    f"{CACHE}/models--Pilcothink--Qwen3.8-27B-MixedInt4-AutoRound/snapshots/*"
)

FUSE = {"in_proj_qkv": "in_proj_qkvz", "in_proj_z": "in_proj_qkvz",
        "in_proj_b": "in_proj_ba", "in_proj_a": "in_proj_ba"}


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


def main() -> None:
    if not SNAP:
        sys.exit("snapshot not downloaded yet")
    cfg_path = f"{SNAP[0]}/config.json"
    cfg = json.load(open(cfg_path))
    q = cfg["quantization_config"]
    if q.get("quant_method") == "gptq" and "dynamic" in q:
        print("already converted"); return
    assert q["quant_method"] == "auto-round" and q["sym"] is True

    dynamic: dict[str, dict] = {}
    for name, override in q["extra_config"].items():
        pat = to_pattern(name)
        if override.get("bits") == 16:
            dynamic["-:" + pat] = {}
        elif override.get("bits") == 8:
            entry = dynamic.setdefault("+:" + pat, {"bits": 8})
            assert entry == {"bits": 8}
        else:
            sys.exit(f"unhandled override {name}: {override}")
    # mtp.fc has no layers.N segment; make its skip explicit and tight
    assert any("mtp" in k for k in dynamic), "expected mtp.fc skip entry"

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
