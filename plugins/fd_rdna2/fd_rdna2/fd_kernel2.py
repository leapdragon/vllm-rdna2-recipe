# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
# fd_kernel4.py — variant G: variant E with the q-padding narrowed from 16 to PAD (=8).
#
# GQA is 6, but tl.dot was assumed to need N>=16, so Q was zero-padded to 16 and every
# downstream tensor -- m_i, l_i, alpha, the exps, both softmax reductions and all four (16,64)
# accumulators -- ran 16 wide with 10 columns of zeros. A probe (m1b_padding_probe.py) shows
# this backend accepts N=8, and that narrowing is worth ~25% on the QK+softmax stage:
#   tl.dot 16 wide 0.291 ms/layer | tl.dot 8 wide 0.217 | plain FMA 8 wide 0.792 (4x worse)
# Plain FMA is NOT the answer -- the dot units dominate. Narrower dots are.
# The v0 kernel unpacked int8->fp16 then reshaped (TILE,64,4)->(TILE,256), which
# forced Triton to materialise dot operands through LDS (ISA: 256x ds_write_b16).
# Here each byte-slice keeps its load-native (TILE,64) shape and we do 4 dots
# against a byte-permuted Q; the d-dimension lives in permuted order (j = b*64+w,
# d = 4w+b) all the way through the partials, un-permuted only at the final store.
import torch
import triton
import triton.language as tl


@triton.jit
def fd2_phase1(
    q_ptr,            # (KVH*GQA_pad16 rows? no:) Q permuted: (KVH, 4, 64, 16) fp16, see wrapper
    kv_i32, kv_f32, bt_ptr,
    m_ptr, l_ptr, acc_ptr,   # acc in PERMUTED d-order: (chunks,KVH,16,256)
    seq_len, seq_ptr, seq_delta,
    s_blk, s_kvh, s_tok,
    BS: tl.constexpr, CHUNK: tl.constexpr, TILE: tl.constexpr,
    GQA: tl.constexpr, KVH: tl.constexpr, SEQ_PTR: tl.constexpr, PAD: tl.constexpr,
):
    # Under CUDA-graph capture the grid is frozen, so the true length cannot be a
    # host scalar: SEQ_PTR mode reads it from device memory (vLLM's seqused_k) and
    # over-range programs retire immediately. Combine recomputes the same bound, so
    # the chunks these programs skip are never read.
    chunk = tl.program_id(0)
    kvh = tl.program_id(1)
    # seq_delta shifts the effective length per query position. Speculative verification
    # passes q>1 tokens, and token i attends to (total - (q-1) + i) keys; delta carries
    # that offset as a kernel argument, which is safe under graph capture because it is
    # fixed for a captured shape.
    n = seq_len
    if SEQ_PTR:
        n = tl.load(seq_ptr) + seq_delta
    start = chunk * CHUNK
    if start >= n:
        return

    offs_q = tl.arange(0, PAD)
    offs_w = tl.arange(0, 64)
    qbase = q_ptr + kvh * (4 * 64 * PAD)
    Q0 = tl.load(qbase + 0 * 64 * PAD + offs_w[:, None] * PAD + offs_q[None, :])  # (64,16)
    Q1 = tl.load(qbase + 1 * 64 * PAD + offs_w[:, None] * PAD + offs_q[None, :])
    Q2 = tl.load(qbase + 2 * 64 * PAD + offs_w[:, None] * PAD + offs_q[None, :])
    Q3 = tl.load(qbase + 3 * 64 * PAD + offs_w[:, None] * PAD + offs_q[None, :])

    m_i = tl.full((PAD,), float("-inf"), tl.float32)
    l_i = tl.zeros((PAD,), tl.float32)
    a0 = tl.zeros((PAD, 64), tl.float32)   # acc for d = 4w+0
    a1 = tl.zeros((PAD, 64), tl.float32)
    a2 = tl.zeros((PAD, 64), tl.float32)
    a3 = tl.zeros((PAD, 64), tl.float32)

    for t in range(0, CHUNK, TILE):
        pos = start + t + tl.arange(0, TILE)
        tmask = pos < n
        blk = tl.load(bt_ptr + pos // BS, mask=tmask, other=0)
        row = blk * s_blk + kvh * s_kvh + (pos % BS) * s_tok

        Ki = tl.load(kv_i32 + row[:, None] + offs_w[None, :],
                     mask=tmask[:, None], other=0)               # (TILE,64) i32
        kscale = tl.load(kv_f32 + row + 64, mask=tmask, other=1.0)

        K0 = ((Ki << 24) >> 24).to(tl.float16)                    # byte 0
        K1 = ((Ki << 16) >> 24).to(tl.float16)
        K2 = ((Ki << 8) >> 24).to(tl.float16)
        K3 = (Ki >> 24).to(tl.float16)
        S = tl.dot(K0, Q0) + tl.dot(K1, Q1) + tl.dot(K2, Q2) + tl.dot(K3, Q3)  # (TILE,16)
        S = S * kscale[:, None]
        S = tl.where(tmask[:, None], S, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(S, axis=0))
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
        P = tl.exp(S - m_new[None, :])
        P = tl.where(tmask[:, None], P, 0.0)
        l_i = l_i * alpha + tl.sum(P, axis=0)
        a0 = a0 * alpha[:, None]
        a1 = a1 * alpha[:, None]
        a2 = a2 * alpha[:, None]
        a3 = a3 * alpha[:, None]

        Vi = tl.load(kv_i32 + row[:, None] + 65 + offs_w[None, :],
                     mask=tmask[:, None], other=0)
        vscale = tl.load(kv_f32 + row + 129, mask=tmask, other=1.0)
        Pv = (P * vscale[:, None]).to(tl.float16)
        Pt = tl.trans(Pv)                                         # (16,TILE)
        a0 += tl.dot(Pt, ((Vi << 24) >> 24).to(tl.float16))
        a1 += tl.dot(Pt, ((Vi << 16) >> 24).to(tl.float16))
        a2 += tl.dot(Pt, ((Vi << 8) >> 24).to(tl.float16))
        a3 += tl.dot(Pt, (Vi >> 24).to(tl.float16))
        m_i = m_new

    base = (chunk * KVH + kvh) * PAD
    tl.store(m_ptr + base + offs_q, m_i)
    tl.store(l_ptr + base + offs_q, l_i)
    ap = acc_ptr + (base + offs_q)[:, None] * 256
    tl.store(ap + 0 * 64 + offs_w[None, :], a0)     # permuted: j = b*64 + w
    tl.store(ap + 1 * 64 + offs_w[None, :], a1)
    tl.store(ap + 2 * 64 + offs_w[None, :], a2)
    tl.store(ap + 3 * 64 + offs_w[None, :], a3)


@triton.jit
def fd2_combine(
    m_ptr, l_ptr, acc_ptr, out_ptr, nchunks, seq_ptr, seq_delta,
    GQA: tl.constexpr, KVH: tl.constexpr,
    CHUNK: tl.constexpr, SEQ_PTR: tl.constexpr, PAD: tl.constexpr,
):
    qh = tl.program_id(0)
    kvh = qh // GQA
    qi = qh % GQA
    offs_j = tl.arange(0, 256)                       # permuted index
    d_out = (offs_j % 64) * 4 + offs_j // 64         # un-permute at the end

    nc = nchunks
    if SEQ_PTR:
        nc = (tl.load(seq_ptr) + seq_delta + CHUNK - 1) // CHUNK

    m_g = float("-inf")
    for c in range(0, nc):
        m_g = tl.maximum(m_g, tl.load(m_ptr + (c * KVH + kvh) * PAD + qi))
    l_g = 0.0
    o = tl.zeros((256,), tl.float32)
    for c in range(0, nc):
        idx = (c * KVH + kvh) * PAD + qi
        w = tl.exp(tl.load(m_ptr + idx) - m_g)
        l_g += tl.load(l_ptr + idx) * w
        o += tl.load(acc_ptr + idx * 256 + offs_j) * w
    tl.store(out_ptr + qh * 256 + d_out, (o / l_g).to(tl.float16))


def permute_q(q_scaled, GQA=6, KVH=2, out=None, PAD=8):
    """(KVH*GQA,256) fp16 -> (KVH,4,64,PAD) fp16: Q[kvh,b,w,qpad] = q[kvh*GQA+qi, 4w+b]."""
    dev = q_scaled.device
    if out is None:
        out = torch.zeros(KVH, 4, 64, PAD, dtype=torch.float16, device=dev)
    # a persistent `out` was zeroed at allocation and we always write the same
    # slice, so the padding columns stay zero without a per-call memset.
    qv = q_scaled.view(KVH, GQA, 64, 4)              # d = 4w+b -> [.., w, b]
    out[:, :, :, :GQA] = qv.permute(0, 3, 2, 1)      # (KVH,4,64,GQA)
    return out


def fd2_decode(qperm, kv_i32, kv_f32, block_table, seq_len,
               BS=1552, CHUNK=1024, TILE=32, GQA=6, KVH=2,
               num_warps=4, workspace=None, strides=None,
               seq_ptr=None, grid_chunks=None, out=None, PAD=8, seq_delta=0):
    # strides: (s_blk, s_kvh, s_tok) in int32 units. Default derives them from a
    # 4-D (blocks,kvh,BS,130) harness tensor; the vLLM plugin passes them explicitly
    # because its handles are flat 1-D views over the cache storage.
    # seq_ptr set => graph-safe mode: fixed grid of grid_chunks, length read on device.
    use_ptr = seq_ptr is not None
    nchunks = grid_chunks if use_ptr else (seq_len + CHUNK - 1) // CHUNK
    dev = qperm.device
    if workspace is None or workspace[0].shape[0] < nchunks * KVH * PAD:
        m = torch.empty(nchunks * KVH * PAD, dtype=torch.float32, device=dev)
        l = torch.empty_like(m)
        a = torch.empty(nchunks * KVH * PAD * 256, dtype=torch.float32, device=dev)
    else:
        m, l, a = workspace
    if out is None:
        out = torch.empty(KVH * GQA, 256, dtype=torch.float16, device=dev)
    sp = seq_ptr if use_ptr else m          # unused when SEQ_PTR is False
    fd2_phase1[(nchunks, KVH)](
        qperm, kv_i32, kv_f32, block_table, m, l, a,
        0 if use_ptr else seq_len, sp, seq_delta,
        *(strides if strides is not None else
          (kv_i32.stride(0), kv_i32.stride(1), kv_i32.stride(2))),
        BS=BS, CHUNK=CHUNK, TILE=TILE, GQA=GQA, KVH=KVH, SEQ_PTR=use_ptr, PAD=PAD,
        num_warps=num_warps)
    fd2_combine[(KVH * GQA,)](m, l, a, out, nchunks, sp, seq_delta,
                              GQA=GQA, KVH=KVH, CHUNK=CHUNK, SEQ_PTR=use_ptr, PAD=PAD)
    return out, (m, l, a)


# ---------------------------------------------------------------------------
# Variant H: batched multi-query verification (MTP).
#
# The host-loop fix (one full KV pass per query position) made MTP correct but
# multiplied KV traffic by nq. Here all nq positions ride in the dot's column
# dimension -- column c = qi*PAD + head -- so each K/V tile is loaded ONCE and
# the only extra cost is arithmetic, which T-C4/T25 showed has multiples of
# headroom. Structurally this is variant G with PAD widened to NQP and the
# causal mask made per-column: position qi attends to (n_last - (NQ-1) + qi)
# keys, so n varies by column instead of being uniform.
# ---------------------------------------------------------------------------

@triton.jit
def fd2_phase1_mq(
    q_ptr,            # (KVH, 4, 64, NQP) fp16; column qi*PAD+g = head g of position qi
    kv_i32, kv_f32, bt_ptr,
    m_ptr, l_ptr, acc_ptr,   # (chunks, KVH, NQP) and (chunks, KVH, NQP, 256), permuted d
    seq_ptr,
    s_blk, s_kvh, s_tok,
    BS: tl.constexpr, CHUNK: tl.constexpr, TILE: tl.constexpr,
    KVH: tl.constexpr, PAD: tl.constexpr, NQ: tl.constexpr, NQP: tl.constexpr,
):
    chunk = tl.program_id(0)
    kvh = tl.program_id(1)
    n_last = tl.load(seq_ptr)              # length seen by the LAST query position
    start = chunk * CHUNK
    if start >= n_last:
        return
    offs_w = tl.arange(0, 64)
    offs_q = tl.arange(0, NQP)
    # per-column effective length; padding columns clamp to n_last (their output
    # is garbage that combine never reads, but their loads stay bounded)
    n_col = n_last - (NQ - 1) + tl.minimum(offs_q // PAD, NQ - 1)

    qbase = q_ptr + kvh * (4 * 64 * NQP)
    Q0 = tl.load(qbase + 0 * 64 * NQP + offs_w[:, None] * NQP + offs_q[None, :])
    Q1 = tl.load(qbase + 1 * 64 * NQP + offs_w[:, None] * NQP + offs_q[None, :])
    Q2 = tl.load(qbase + 2 * 64 * NQP + offs_w[:, None] * NQP + offs_q[None, :])
    Q3 = tl.load(qbase + 3 * 64 * NQP + offs_w[:, None] * NQP + offs_q[None, :])

    m_i = tl.full((NQP,), float("-inf"), tl.float32)
    l_i = tl.zeros((NQP,), tl.float32)
    a0 = tl.zeros((NQP, 64), tl.float32)
    a1 = tl.zeros((NQP, 64), tl.float32)
    a2 = tl.zeros((NQP, 64), tl.float32)
    a3 = tl.zeros((NQP, 64), tl.float32)

    for t in range(0, CHUNK, TILE):
        pos = start + t + tl.arange(0, TILE)
        tmask = pos < n_last                              # widest load mask
        blk = tl.load(bt_ptr + pos // BS, mask=tmask, other=0)
        row = blk * s_blk + kvh * s_kvh + (pos % BS) * s_tok
        Ki = tl.load(kv_i32 + row[:, None] + offs_w[None, :],
                     mask=tmask[:, None], other=0)        # (TILE,64) i32
        kscale = tl.load(kv_f32 + row + 64, mask=tmask, other=1.0)

        K0 = ((Ki << 24) >> 24).to(tl.float16)
        K1 = ((Ki << 16) >> 24).to(tl.float16)
        K2 = ((Ki << 8) >> 24).to(tl.float16)
        K3 = (Ki >> 24).to(tl.float16)
        S = tl.dot(K0, Q0) + tl.dot(K1, Q1) + tl.dot(K2, Q2) + tl.dot(K3, Q3)
        S = S * kscale[:, None]
        smask = pos[:, None] < n_col[None, :]             # per-column causal bound
        S = tl.where(smask, S, float("-inf"))

        m_new = tl.maximum(m_i, tl.max(S, axis=0))
        alpha = tl.where(m_i == float("-inf"), 0.0, tl.exp(m_i - m_new))
        P = tl.exp(S - m_new[None, :])
        P = tl.where(smask, P, 0.0)
        l_i = l_i * alpha + tl.sum(P, axis=0)
        a0 = a0 * alpha[:, None]
        a1 = a1 * alpha[:, None]
        a2 = a2 * alpha[:, None]
        a3 = a3 * alpha[:, None]

        Vi = tl.load(kv_i32 + row[:, None] + 65 + offs_w[None, :],
                     mask=tmask[:, None], other=0)
        vscale = tl.load(kv_f32 + row + 129, mask=tmask, other=1.0)
        Pv = (P * vscale[:, None]).to(tl.float16)
        Pt = tl.trans(Pv)
        a0 += tl.dot(Pt, ((Vi << 24) >> 24).to(tl.float16))
        a1 += tl.dot(Pt, ((Vi << 16) >> 24).to(tl.float16))
        a2 += tl.dot(Pt, ((Vi << 8) >> 24).to(tl.float16))
        a3 += tl.dot(Pt, (Vi >> 24).to(tl.float16))
        m_i = m_new

    base = (chunk * KVH + kvh) * NQP
    tl.store(m_ptr + base + offs_q, m_i)
    tl.store(l_ptr + base + offs_q, l_i)
    ap = acc_ptr + (base + offs_q)[:, None] * 256
    tl.store(ap + 0 * 64 + offs_w[None, :], a0)
    tl.store(ap + 1 * 64 + offs_w[None, :], a1)
    tl.store(ap + 2 * 64 + offs_w[None, :], a2)
    tl.store(ap + 3 * 64 + offs_w[None, :], a3)


@triton.jit
def fd2_combine_mq(
    m_ptr, l_ptr, acc_ptr, out_ptr, nchunks, seq_ptr,
    GQA: tl.constexpr, KVH: tl.constexpr,
    CHUNK: tl.constexpr, PAD: tl.constexpr, NQ: tl.constexpr, NQP: tl.constexpr,
):
    # grid: (NQ * KVH * GQA,). Chunks past a position's own length carry m=-inf,
    # l=0 and weight out to zero, so every position can reduce over the LAST
    # position's chunk count.
    pid = tl.program_id(0)
    qi = pid // (KVH * GQA)
    rem = pid % (KVH * GQA)
    kvh = rem // GQA
    head = rem % GQA
    col = qi * PAD + head
    offs_j = tl.arange(0, 256)
    d_out = (offs_j % 64) * 4 + offs_j // 64

    nc = (tl.load(seq_ptr) + CHUNK - 1) // CHUNK
    m_g = float("-inf")
    for c in range(0, nc):
        m_g = tl.maximum(m_g, tl.load(m_ptr + (c * KVH + kvh) * NQP + col))
    l_g = 0.0
    o = tl.zeros((256,), tl.float32)
    for c in range(0, nc):
        idx = (c * KVH + kvh) * NQP + col
        w = tl.exp(tl.load(m_ptr + idx) - m_g)
        l_g += tl.load(l_ptr + idx) * w
        o += tl.load(acc_ptr + idx * 256 + offs_j) * w
    tl.store(out_ptr + pid * 256 + d_out, (o / l_g).to(tl.float16))


def permute_q_mq(q_scaled, nq, GQA=6, KVH=2, PAD=8, out=None, NQP=32):
    """(nq, KVH*GQA, 256) fp16 -> (KVH, 4, 64, NQP), column qi*PAD+g = head g of pos qi."""
    dev = q_scaled.device
    if out is None:
        out = torch.zeros(KVH, 4, 64, NQP, dtype=torch.float16, device=dev)
    qv = q_scaled.view(nq, KVH, GQA, 64, 4)
    # (KVH, 4, 64, nq, GQA); a persistent `out` was zeroed at allocation and we
    # always write the same slice, so padding columns stay zero
    out.view(KVH, 4, 64, NQP)[..., : nq * PAD].view(KVH, 4, 64, nq, PAD)[..., :GQA] = \
        qv.permute(1, 4, 3, 0, 2)
    return out


def fd2_decode_mq(qperm, kv_i32, kv_f32, block_table, seq_ptr, grid_chunks,
                  workspace, out, nq, BS=1552, CHUNK=512, TILE=32, GQA=6, KVH=2,
                  PAD=8, NQP=32, num_warps=8, strides=None):
    m, l, a = workspace
    fd2_phase1_mq[(grid_chunks, KVH)](
        qperm, kv_i32, kv_f32, block_table, m, l, a, seq_ptr,
        *(strides if strides is not None else
          (kv_i32.stride(0), kv_i32.stride(1), kv_i32.stride(2))),
        BS=BS, CHUNK=CHUNK, TILE=TILE, KVH=KVH, PAD=PAD, NQ=nq, NQP=NQP,
        num_warps=num_warps)
    fd2_combine_mq[(nq * KVH * GQA,)](
        m, l, a, out, grid_chunks, seq_ptr,
        GQA=GQA, KVH=KVH, CHUNK=CHUNK, PAD=PAD, NQ=nq, NQP=NQP)
    return out
