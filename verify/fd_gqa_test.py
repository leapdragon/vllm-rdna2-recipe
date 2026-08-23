# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
#!/usr/bin/env python3
"""Correctness + timing for the fd_rdna2 kernels across GQA geometries.

Runs the SAME kernel entry points the plugin calls, with synthetic
int8-per-token-head KV in the exact 260-byte row layout the kernel addresses
(K bytes, f32 k-scale, V bytes, f32 v-scale), against an fp32 torch reference.

Covers the 27B geometry (GQA 6 / KVH 4, PAD 8 — regression guard) and the
122B geometry (GQA 16 / KVH 2, PAD 16 — the generalization), q positions 1,
2 (batched single-pass) and 3 (per-position loop, exercised via seq_delta).

Run inside the serving container (needs triton + one GPU):
  docker run --rm --entrypoint python3 --device /dev/kfd --device /dev/dri \
    --group-add render --group-add video \
    -e HSA_OVERRIDE_GFX_VERSION=10.3.0 -e ROCR_VISIBLE_DEVICES=0 \
    -v <recipe-root>:/repo <your-vllm-gfx1030-image> \
    /repo/verify/fd_gqa_test.py
"""
import sys
import time

import torch

sys.path.insert(0, "/repo/builds/shared/plugins/fd_rdna2")  # recipe root mounted at /repo
from fd_rdna2.fd_kernel2 import (fd2_decode, fd2_decode_mq, permute_q,
                                 permute_q_mq)

DEV = "cuda"
HS = 256


def build_kv2(nblocks, bs, kvh, seed):
    """(nblocks, bs, kvh, 520) int8: [K bytes(256), kscale f32(4), V bytes(256),
    vscale f32(4)] — 520 bytes = 130 i32, matching the kernel's +64/+65/+129
    i32/f32 offsets within one (token, head) record."""
    g = torch.Generator(device="cpu").manual_seed(seed)
    kv = torch.zeros(nblocks, bs, kvh, 520, dtype=torch.int8, device=DEV)
    kb = torch.randint(-100, 100, (nblocks, bs, kvh, HS), generator=g,
                       dtype=torch.int8).to(DEV)
    vb = torch.randint(-100, 100, (nblocks, bs, kvh, HS), generator=g,
                       dtype=torch.int8).to(DEV)
    ks = (torch.rand(nblocks, bs, kvh, generator=g) * 0.02 + 0.001).to(DEV)
    vs = (torch.rand(nblocks, bs, kvh, generator=g) * 0.02 + 0.001).to(DEV)
    kv[..., :HS] = kb
    kv[..., 260:260 + HS] = vb
    kv.view(torch.float32).view(nblocks, bs, kvh, 130)[..., 64] = ks
    kv.view(torch.float32).view(nblocks, bs, kvh, 130)[..., 129] = vs
    return kv, kb, vb, ks, vs


def reference(q, kb, vb, ks, vs, bt, ctx, nq, gqa, kvh):
    """fp32 reference; q: (nq, kvh*gqa, HS) UNSCALED (softmax scale applied
    by caller before both paths)."""
    outs = []
    nblocks, bs = kb.shape[0], kb.shape[1]
    flat_k = kb.float().view(nblocks * bs, kvh, HS)
    flat_v = vb.float().view(nblocks * bs, kvh, HS)
    flat_ks = ks.view(nblocks * bs, kvh)
    flat_vs = vs.view(nblocks * bs, kvh)
    for qi in range(nq):
        n = ctx - (nq - 1) + qi
        pos = torch.arange(n, device=DEV)
        rows = bt[pos // bs].long() * bs + (pos % bs)
        o_heads = []
        for h in range(kvh * gqa):
            kvhh = h // gqa
            K = flat_k[rows, kvhh] * flat_ks[rows, kvhh, None]   # (n, HS)
            V = flat_v[rows, kvhh] * flat_vs[rows, kvhh, None]
            s = (K @ q[qi, h].float())                            # (n,)
            p = torch.softmax(s, dim=0)
            o_heads.append(p @ V)                                 # (HS,)
        outs.append(torch.stack(o_heads))
    return torch.stack(outs)                                      # (nq, heads, HS)


def run_case(gqa, kvh, nq, ctx, bs=1552, chunk=512, tile=32, warps=8, seed=0):
    pad = 8 if gqa <= 8 else 16
    nblocks = (ctx + bs - 1) // bs + 1
    kv, kb, vb, ks, vs = build_kv2(nblocks, bs, kvh, seed)
    st = kv.untyped_storage()
    f8 = torch.empty(0, dtype=torch.int8, device=DEV)
    f8.set_(st, 0, (st.nbytes(),))
    i32 = f8.view(torch.int32)
    f32 = f8.view(torch.float32)
    s_blk = kv.stride(0) // 4
    s_tok = kv.stride(1) // 4
    s_kvh = kv.stride(2) // 4
    bt = torch.randperm(nblocks, device=DEV).to(torch.int32)
    scale = HS ** -0.5
    q = (torch.randn(nq, kvh * gqa, HS, device=DEV) * 0.3).to(torch.float16)
    su = torch.tensor([ctx], dtype=torch.int32, device=DEV)
    gchunks = (nblocks * bs + chunk - 1) // chunk

    m = torch.empty(gchunks * kvh * 32, dtype=torch.float32, device=DEV)
    ws = (m, torch.empty_like(m),
          torch.empty(gchunks * kvh * 32 * HS, dtype=torch.float32, device=DEV))
    out_buf = torch.empty(4 * kvh * gqa, HS, dtype=torch.float16, device=DEV)

    def kernel_pass():
        got = torch.empty(nq, kvh * gqa, HS, dtype=torch.float16, device=DEV)
        if nq > 1 and nq * pad <= 32:
            nqp = 16 if nq * pad <= 16 else 32
            qbuf = torch.zeros(kvh, 4, 64, nqp, dtype=torch.float16, device=DEV)
            qp = permute_q_mq((q * scale).to(torch.float16), nq, GQA=gqa,
                              KVH=kvh, PAD=pad, out=qbuf, NQP=nqp)
            fd2_decode_mq(qp, i32, f32, bt, su, gchunks, ws, out_buf, nq,
                          BS=bs, CHUNK=chunk, TILE=tile, GQA=gqa, KVH=kvh,
                          PAD=pad, NQP=nqp, num_warps=warps,
                          strides=(s_blk, s_kvh, s_tok))
            got.copy_(out_buf[: nq * kvh * gqa].view(nq, kvh * gqa, HS))
        else:
            qbuf = torch.zeros(kvh, 4, 64, pad, dtype=torch.float16, device=DEV)
            for qi in range(nq):
                qp = permute_q((q[qi] * scale).to(torch.float16), GQA=gqa,
                               KVH=kvh, out=qbuf, PAD=pad)
                fd2_decode(qp, i32, f32, bt, 0, BS=bs, CHUNK=chunk, TILE=tile,
                           GQA=gqa, KVH=kvh, num_warps=warps, workspace=ws,
                           strides=(s_blk, s_kvh, s_tok), seq_ptr=su,
                           grid_chunks=gchunks, out=out_buf,
                           seq_delta=qi - (nq - 1), PAD=pad)
                got[qi].copy_(out_buf[: kvh * gqa])
        return got

    got = kernel_pass()
    ref = reference(q * scale, kb, vb, ks, vs, bt, ctx, nq, gqa, kvh)
    err = (got.float() - ref).abs()
    denom = ref.abs().max().clamp_min(1e-6)
    rel = (err.max() / denom).item()

    torch.cuda.synchronize()
    t0 = time.perf_counter()
    REPS = 50
    for _ in range(REPS):
        kernel_pass()
    torch.cuda.synchronize()
    ms = (time.perf_counter() - t0) / REPS * 1e3
    mode = ("batched" if nq > 1 and nq * pad <= 32 else
            ("loop" if nq > 1 else "single"))
    ok = rel < 0.03
    print(f"GQA={gqa:2d} KVH={kvh} nq={nq} ctx={ctx:6d} [{mode:7s}] "
          f"rel_err={rel:.4f} {'OK ' if ok else 'FAIL'} {ms:7.3f} ms/layer-pass")
    return ok


def main():
    torch.set_grad_enabled(False)
    all_ok = True
    for gqa, kvh in ((6, 4), (16, 2)):
        for nq in (1, 2, 3, 4):
            for ctx in (4096, 40960):
                all_ok &= run_case(gqa, kvh, nq, ctx)
    print("ALL OK" if all_ok else "FAILURES PRESENT")
    sys.exit(0 if all_ok else 1)


if __name__ == "__main__":
    main()
