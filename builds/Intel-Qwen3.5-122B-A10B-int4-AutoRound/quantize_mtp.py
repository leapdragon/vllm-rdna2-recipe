#!/usr/bin/env python3
# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
"""RTN-pack the dense bf16 MTP head to GPTQ int4, in the exporter's own format.

Why: the MTP draft (needed for speculative decoding, which patch 0009 enables
under PP) ships as 5.05 GiB of dense bf16 — all of which lands on the LAST PP
stage, where it starves the KV cache out of existence at max_model_len 131072.
Packed to the checkpoint's own int4 scheme it is ~1.3 GiB, the last stage fits
again at the standard partition, and the draft's expert GEMVs ride the same
moe_wna16/skinny kernels as the target instead of a penalized dense path.

What gets packed: ONLY the 256 routed experts' gate/up/down projections
(4.83 of the 5.05 GiB). Everything else stays dense bf16, because it must
match what vLLM actually builds:
  - self_attn q/k/v/o and fc: vLLM's AutoGPTQ linear gate decides
    quantized-vs-dense from the checkpoint's safetensors METADATA — fetched
    from the HF HUB for hub-style model names (get_safetensors_params_metadata
    only reads local headers as an offline fallback). The hub copy has a dense
    MTP head, so these modules build dense no matter what the local tensors
    say. Keeping them dense is also the only state consistent across
    online and offline (HF_HUB_OFFLINE=1) runs.
  - experts are exempt from that gate: RoutedExperts dispatch to the MoE
    quant path (MoeWNA16), which follows the gptq config directly — so packed
    expert tensors load quantized and ride the moe_wna16/skinny kernels.
  - shared_expert gate/up/down + shared_expert_gate: the config's layer-0
    "-:" skip patterns start with ".*" and so also match mtp.layers.0.*
  - the router (mlp.gate), all norms, 1-D tensors

Format (byte-verified against the target's packed linears):
  qweight I32 [K/8, N]  k-sequential nibbles, low nibble = lowest k
  scales  F16 [K/128, N]
  qzeros  I32 [K/128, N/8] = 0x77777777  (legacy zeros-1 convention: stored 7,
          effective zero 8; dequant w = (q - 8) * s)
RTN with s = absmax/7: max |error| <= s/2, no clipping on either side. The
draft only proposes tokens — quantization error costs acceptance rate, never
output correctness (verification is exact).

Also removes the "-:mtp\\..*" dynamic skip from config.json (the packed draft
must now build QUANTIZED; convert.py only re-adds the skip for dense-MTP
checkpoints). Original shard kept as model_extra_tensors.safetensors.dense-bak
— the non-.safetensors suffix keeps it out of the loader's shard glob.

Idempotent. Re-run convert.py first after any fresh download.
"""
import glob
import json
import re
import struct
import sys

import numpy as np

import os

CACHE = os.environ.get(
    "HF_CACHE",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "..", "hf-cache"),
) + "/hub"
SNAP = glob.glob(
    f"{CACHE}/models--Intel--Qwen3.5-122B-A10B-int4-AutoRound/snapshots/*"
)
SHARD = "model_extra_tensors.safetensors"
GROUP = 128
MTP_SKIP = r"-:mtp\..*"

QUANT_RE = re.compile(
    r"^mtp\.layers\.0\.mlp\.experts\.\d+\.(gate|up|down)_proj\.weight$"
)


def read_safetensors(path):
    with open(path, "rb") as f:
        n = struct.unpack("<Q", f.read(8))[0]
        header = json.loads(f.read(n))
        data = f.read()
    return header, data


def bf16_to_f32(raw: bytes, shape) -> np.ndarray:
    u16 = np.frombuffer(raw, dtype=np.uint16).astype(np.uint32)
    return (u16 << 16).view(np.float32).reshape(shape)


def pack_int4(w: np.ndarray):
    """w [out N, in K] f32 -> (qweight I32 [K/8, N], scales F16 [K/128, N])."""
    n_out, k_in = w.shape
    assert k_in % GROUP == 0, (n_out, k_in)
    wt = np.ascontiguousarray(w.T)                      # [K, N]
    g = wt.reshape(k_in // GROUP, GROUP, n_out)
    absmax = np.abs(g).max(axis=1, keepdims=True)       # [K/G, 1, N]
    scales = (absmax / 7.0).astype(np.float32)
    scales[scales == 0] = 1.0
    q = np.rint(g / scales).astype(np.int32) + 8
    np.clip(q, 0, 15, out=q)
    q = q.reshape(k_in, n_out).astype(np.uint32)
    shifts = (np.arange(8, dtype=np.uint32) * 4)[None, :, None]
    qweight = (q.reshape(k_in // 8, 8, n_out) << shifts).sum(
        axis=1, dtype=np.uint32
    )
    return qweight.view(np.int32), scales.reshape(
        k_in // GROUP, n_out
    ).astype(np.float16)


def dequant_check(name, w, qweight, scales):
    q = np.repeat(qweight.view(np.uint32), 8, axis=0)
    shifts = np.tile(np.arange(8, dtype=np.uint32) * 4, qweight.shape[0])
    q = (q >> shifts[:, None]) & 0xF
    deq = (q.astype(np.float32) - 8.0) * np.repeat(
        scales.astype(np.float32), GROUP, axis=0
    )
    err = np.abs(deq - w.T)
    rel = err.max() / (np.abs(w).max() + 1e-30)
    print(f"  verify {name}: max_abs_err={err.max():.3e} "
          f"mean={err.mean():.3e} rel_max={rel:.4f}")
    # bound: err <= s/2 = group_absmax/14 <= global_absmax/14 ~= 0.0715
    assert rel < 0.072, "quantization error out of band"


def restore_dense(snap, idx):
    """Roll back to the dense shard (kept as .dense-bak) so packing always
    starts from the pristine tensors — makes re-runs with a different
    QUANT_RE self-healing."""
    import os
    bak = f"{snap}/{SHARD}.dense-bak"
    if not os.path.exists(bak):
        return False
    os.replace(bak, f"{snap}/{SHARD}")
    header, _ = read_safetensors(f"{snap}/{SHARD}")
    wm = idx["weight_map"]
    for k in [k for k in wm if k.startswith("mtp.")]:
        del wm[k]
    for k in header:
        if k != "__metadata__":
            wm[k] = SHARD
    return True


def main():
    if not SNAP:
        sys.exit("snapshot not downloaded yet")
    snap = SNAP[0]
    idx_path = f"{snap}/model.safetensors.index.json"
    idx = json.load(open(idx_path))
    if any(k.startswith("mtp.") and k.endswith(".qweight")
           for k in idx["weight_map"]):
        if not restore_dense(snap, idx):
            print("already quantized")
            return
        print("restored dense shard from backup; repacking")

    header, data = read_safetensors(f"{snap}/{SHARD}")
    names = [k for k in header if k != "__metadata__"]

    out = {}            # name -> (dtype str, shape, bytes)
    checked = 0
    for name in sorted(names):
        meta = header[name]
        a, b = meta["data_offsets"]
        raw = data[a:b]
        if QUANT_RE.match(name):
            assert meta["dtype"] == "BF16", (name, meta["dtype"])
            w = bf16_to_f32(raw, meta["shape"])
            qweight, scales = pack_int4(w)
            if checked < 2 or name.endswith("fc.weight"):
                dequant_check(name, w, qweight, scales)
                checked += 1
            base = name[: -len(".weight")]
            kg, n_out = scales.shape
            qzeros = np.full((kg, n_out // 8), 0x77777777, dtype=np.uint32)
            out[base + ".qweight"] = ("I32", list(qweight.shape),
                                      qweight.tobytes())
            out[base + ".scales"] = ("F16", list(scales.shape),
                                     scales.tobytes())
            out[base + ".qzeros"] = ("I32", list(qzeros.shape),
                                     qzeros.view(np.int32).tobytes())
        else:
            out[name] = (meta["dtype"], meta["shape"], raw)

    # write the new shard
    new_header = {}
    off = 0
    order = sorted(out)
    for name in order:
        dtype, shape, raw = out[name]
        new_header[name] = {"dtype": dtype, "shape": shape,
                            "data_offsets": [off, off + len(raw)]}
        off += len(raw)
    hdr_bytes = json.dumps(new_header).encode()
    import os
    os.replace(f"{snap}/{SHARD}", f"{snap}/{SHARD}.dense-bak")
    with open(f"{snap}/{SHARD}", "wb") as f:
        f.write(struct.pack("<Q", len(hdr_bytes)))
        f.write(hdr_bytes)
        for name in order:
            f.write(out[name][2])

    # index: quantized tensors change names; everything stays in this shard
    wm = idx["weight_map"]
    for name in names:
        del wm[name]
    for name in order:
        wm[name] = SHARD
    idx["metadata"]["total_size"] = (
        idx["metadata"].get("total_size", 0)
        - sum(header[k]["data_offsets"][1] - header[k]["data_offsets"][0]
              for k in names)
        + off
    )
    json.dump(idx, open(idx_path, "w"), indent=2)

    # config: the packed draft must build quantized -> drop the dense-MTP skip
    cfg_path = f"{snap}/config.json"
    cfg = json.load(open(cfg_path))
    dyn = cfg["quantization_config"].get("dynamic", {})
    if MTP_SKIP in dyn:
        del dyn[MTP_SKIP]
        json.dump(cfg, open(cfg_path, "w"), indent=2)

    n_q = sum(1 for k in order if k.endswith(".qweight"))
    print(f"packed {n_q} linears; shard {len(data)/1e9:.2f} GB -> "
          f"{off/1e9:.2f} GB; dense original kept as {SHARD}.dense-bak")


if __name__ == "__main__":
    main()
