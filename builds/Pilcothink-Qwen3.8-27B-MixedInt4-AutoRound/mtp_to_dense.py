# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
"""Step 1: dequantize the mtp.* GPTQ tensors of the Pilcothink snapshot to bf16.

vLLM's Qwen3_5MultiTokenPredictor builds unquantized layers, so the draft
weights must be dense. GPTQ g32 sym: true zero is 8; the stored qzeros nibble
(7 = legacy minus-one convention, 8 = direct) is checked but the dequant always
subtracts 8. New dense tensors land in one extra shard; index updated.
"""
import glob, json, os
import torch
from safetensors.torch import safe_open, save_file

SNAP = glob.glob("/root/.cache/huggingface/hub/models--Pilcothink--Qwen3.8-27B-MixedInt4-AutoRound/snapshots/*")[0]
idx_path = os.path.join(SNAP, "model.safetensors.index.json")
idx = json.load(open(idx_path))
wm = idx["weight_map"]

bases = sorted({k[:-8] for k in wm if k.startswith("mtp.") and k.endswith(".qweight")})
print("quantized mtp linears:", bases)

def load(name):
    with safe_open(os.path.join(SNAP, wm[name]), framework="pt") as f:
        return f.get_tensor(name)

new_tensors = {}
for b in bases:
    qw, qz, sc = load(b + ".qweight"), load(b + ".qzeros"), load(b + ".scales")
    Kp, N = qw.shape          # [K/8, N] int32, 4-bit packed along K
    K = Kp * 8
    shifts = torch.arange(0, 32, 4)
    q = ((qw.unsqueeze(-1) >> shifts) & 0xF)          # [K/8, N, 8]
    q = q.permute(0, 2, 1).reshape(K, N)              # [K, N] in K order
    z_nib = ((qz[0, 0] >> shifts) & 0xF)
    assert z_nib[0].item() in (7, 8), f"unexpected zero nibble {z_nib[0]}"
    G = K // sc.shape[0]
    w = (q.float() - 8.0) * sc.float().repeat_interleave(G, dim=0)
    new_tensors[b + ".weight"] = w.T.contiguous().to(torch.bfloat16)
    print(f"  {b}: [{N}, {K}] zero_nibble={z_nib[0].item()} group={G}")

shard = "model-mtp-dense.safetensors"
save_file(new_tensors, os.path.join(SNAP, shard))
for b in bases:
    for suf in (".qweight", ".qzeros", ".scales", ".g_idx"):
        wm.pop(b + suf, None)
    wm[b + ".weight"] = shard
idx["metadata"]["total_size"] = idx.get("metadata", {}).get("total_size", 0)
json.dump(idx, open(idx_path, "w"))
print(f"wrote {shard} with {len(new_tensors)} tensors; index updated")

# ----------------------------------------------------------------------
# Step 2: strip the stale mtp quant tensors from their shard files.
# vLLM's loader streams every tensor inside each shard, ignoring the index,
# so the packed originals must physically leave the files.

import glob, json, os
from safetensors.torch import safe_open, save_file

SNAP = glob.glob("/root/.cache/huggingface/hub/models--Pilcothink--Qwen3.8-27B-MixedInt4-AutoRound/snapshots/*")[0]
BAD_SUFFIX = (".qweight", ".qzeros", ".scales", ".g_idx")

shards = set()
for f in glob.glob(os.path.join(SNAP, "model*.safetensors")):
    with safe_open(f, framework="pt") as sf:
        if any(k.startswith("mtp.") and k.endswith(BAD_SUFFIX) for k in sf.keys()):
            shards.add(f)
print("shards to rewrite:", [os.path.basename(s) for s in shards])
for f in shards:
    keep, dropped = {}, 0
    with safe_open(f, framework="pt") as sf:
        meta = sf.metadata()
        for k in sf.keys():
            if k.startswith("mtp.") and k.endswith(BAD_SUFFIX):
                dropped += 1; continue
            keep[k] = sf.get_tensor(k)
    tmp = f + ".tmp"
    save_file(keep, tmp, metadata=meta)
    os.replace(tmp, f)
    print(f"  {os.path.basename(f)}: dropped {dropped}, kept {len(keep)}")
