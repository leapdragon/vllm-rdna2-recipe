# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
"""WS1 flash-decode attention for gfx1030 — vLLM general plugin.

Routes the pure-decode, int8-per-token-head, head_size-256 case to our kernel;
everything else (prefill, chunked prefill, fp16 KV, spec decode, batch>1)
delegates unchanged to vLLM's unified_attention.

Patch point: vllm.v1.attention.backends.triton_attn.unified_attention.
Chosen over register_backend deliberately — it inherits all cache-write, scale
and metadata setup from the stock impl, so the override surface is one function
instead of a backend + impl pair. Enable with FD_RDNA2=1.
"""
import os

_stats = {"fast": 0, "slow": 0, "reason": {}}
_log = None


def _reject(why):
    _stats["slow"] += 1
    n = _stats["reason"].get(why, 0) + 1
    _stats["reason"][why] = n
    if _log is not None and n == 1:
        # first occurrence only: enough to see which cases delegate, without spam
        _log.info("fd_rdna2: delegating to stock kernel (%s)", why)
    return False


def register():
    if os.getenv("FD_RDNA2", "0") != "1":
        return
    import torch
    from vllm.logger import init_logger
    import vllm.v1.attention.backends.triton_attn as ta
    from vllm.v1.kv_cache_interface import KVQuantMode
    from .fd_kernel2 import fd2_decode, permute_q, fd2_decode_mq, permute_q_mq

    global _log
    logger = _log = init_logger("vllm.fd_rdna2")
    if getattr(ta, "_fd_rdna2_patched", False):
        return
    orig = ta.unified_attention

    CHUNK = int(os.getenv("FD_CHUNK", "512"))
    TILE = int(os.getenv("FD_TILE", "32"))
    WARPS = int(os.getenv("FD_WARPS", "8"))
    MAXQ = int(os.getenv("FD_MAXQ", "4"))   # batched path packs nq*8 columns into NQP=32, so 4 max
    _ws = {}

    def patched(**kw):
        q = kw.get("q")
        try:
            ok = (
                # q == 1 is plain decode; q up to MAXQ is a speculative verification pass
                # (MTP drafts + 1 bonus token). Before this, q>1 delegated to the stock kernel,
                # which made MTP a 3.6x regression at long context -- the stock attention slope
                # is ~6x ours, and with speculation it ran on every step that matters.
                1 <= (kw.get("max_seqlen_q") or 0) <= MAXQ
                and kw.get("kv_quant_mode") == KVQuantMode.INT8_PER_TOKEN_HEAD
                and q is not None and q.dim() == 3
                and q.shape[0] == kw.get("max_seqlen_q")          # single sequence
                and q.shape[0] <= MAXQ
                and q.shape[2] == 256                             # head_size
                and bool(kw.get("causal"))
                and kw.get("sinks") is None
                and kw.get("alibi_slopes") is None
                and not kw.get("softcap")
                and kw.get("window_size", (-1, -1))[0] < 0
                and kw.get("chunk_lookback", -1) < 0
                and kw.get("rswa_window") is None
                and kw.get("mm_prefix_range") is None
            )
        except Exception:
            ok = False
        if not ok:
            _reject("shape/mode")
            return orig(**kw)

        k = kw["k"]                    # (blocks, block_size, n_kv, 260) int8 view
        n_kv = q.shape[1] // 6
        if k.dtype != torch.int8 or q.shape[1] % 6 != 0 or n_kv != k.shape[2]:
            _reject("layout")
            return orig(**kw)

        # int32/fp32 handles over the same storage
        st = k.untyped_storage()
        f8 = torch.empty(0, dtype=torch.int8, device=k.device)
        f8.set_(st, 0, (st.nbytes(),))
        off32 = k.storage_offset() // 4
        i32 = f8.view(torch.int32)[off32:]
        f32 = f8.view(torch.float32)[off32:]
        if any(s % 4 for s in (k.stride(0), k.stride(1), k.stride(2))):
            _reject("stride align")
            return orig(**kw)
        s_blk, s_tok, s_kvh = (k.stride(0) // 4, k.stride(1) // 4, k.stride(2) // 4)

        bt_all = kw["block_table"]
        if bt_all.dtype != torch.int32:
            _reject("block_table dtype")
            return orig(**kw)
        bt = bt_all[0]
        bs = k.shape[1]

        # Grid must be constant across capture and replay, so size it for the
        # longest sequence the block table can address, not the current one.
        max_ctx = bt.shape[-1] * bs
        gchunks = (max_ctx + CHUNK - 1) // CHUNK
        buf = _ws.get((gchunks, n_kv))
        if buf is None:
            if torch.cuda.is_current_stream_capturing():
                # Allocating here would bind the buffers to one graph's private
                # pool; warm-up runs outside capture normally create them first.
                _reject("capture before warmup")
                return orig(**kw)
            dev = q.device
            PAD = 8   # GQA is 6; tl.dot accepts N=8 on this backend and 8 beats 16 by 1.6x
            NQP = 32  # widest column count: batched verification packs nq*PAD columns
            m = torch.empty(gchunks * n_kv * NQP, dtype=torch.float32, device=dev)
            buf = (m, torch.empty_like(m),
                   torch.empty(gchunks * n_kv * NQP * 256, dtype=torch.float32, device=dev),
                   torch.empty(4 * q.shape[1], 256, dtype=torch.float16, device=dev),
                   torch.zeros(n_kv, 4, 64, PAD, dtype=torch.float16, device=dev),
                   torch.zeros(n_kv, 4, 64, 16, dtype=torch.float16, device=dev),
                   torch.zeros(n_kv, 4, 64, 32, dtype=torch.float16, device=dev))
            _ws[(gchunks, n_kv)] = buf
            logger.info("fd_rdna2: buffers for max_ctx=%d (%d chunks of %d)",
                        max_ctx, gchunks, CHUNK)
        m, l, a, out_buf, qbuf, qbuf_mq16, qbuf_mq32 = buf

        nq = q.shape[0]
        if nq > 1:
            # Batched verification: every position rides in the dot's column dimension,
            # so KV is read ONCE per tile instead of once per position. NQP=16 covers
            # nq=2; 32 covers nq=3..4.
            NQP = 16 if nq == 2 else 32
            qs = (q * kw["softmax_scale"]).to(torch.float16)       # (nq, n_heads, 256)
            qp = permute_q_mq(qs, nq, GQA=6, KVH=n_kv, PAD=8,
                              out=(qbuf_mq16 if NQP == 16 else qbuf_mq32), NQP=NQP)
            fd2_decode_mq(qp, i32, f32, bt, kw["seqused_k"], gchunks, (m, l, a),
                          out_buf, nq, BS=bs, CHUNK=CHUNK, TILE=TILE, GQA=6, KVH=n_kv,
                          PAD=8, NQP=NQP, num_warps=WARPS, strides=(s_blk, s_kvh, s_tok))
            kw["out"][:nq].copy_(out_buf[: nq * q.shape[1]].view(nq, q.shape[1], 256))
            _stats["fast"] += 1
            key = f"fast batched q={nq}"
            _stats["reason"][key] = _stats["reason"].get(key, 0) + 1
            if _log is not None and _stats["reason"][key] == 1:
                _log.info("fd_rdna2: batched fast path for q=%d (one KV pass)", nq)
            return None
        for qi in range(nq):
            # token qi of the verification block attends to (total - (nq-1) + qi) keys.
            # Passing that as an offset keeps the launch graph-safe: it is constant for a
            # captured shape, unlike reading the length on the host.
            qs = (q[qi] * kw["softmax_scale"]).to(torch.float16)   # (n_q_heads, 256)
            qp = permute_q(qs, GQA=6, KVH=n_kv, out=qbuf)
            fd2_decode(qp, i32, f32, bt, 0, BS=bs, CHUNK=CHUNK,
                       TILE=TILE, GQA=6, KVH=n_kv, num_warps=WARPS,
                       workspace=(m, l, a), strides=(s_blk, s_kvh, s_tok),
                       seq_ptr=kw["seqused_k"], grid_chunks=gchunks, out=out_buf,
                       seq_delta=qi - (nq - 1))
            kw["out"][qi].copy_(out_buf[: q.shape[1]])
        _stats["fast"] += 1
        if nq > 1:
            _stats["reason"][f"fast q={nq}"] = _stats["reason"].get(f"fast q={nq}", 0) + 1
            if _log is not None and _stats["reason"][f"fast q={nq}"] == 1:
                _log.info("fd_rdna2: taking the fast path for q=%d (speculative verification)", nq)
        return None

    ta.unified_attention = patched
    ta._fd_rdna2_patched = True
    ta._fd_rdna2_stats = _stats
    logger.info("fd_rdna2: flash-decode override active (CHUNK=%d TILE=%d warps=%d); "
                "decode/int8/hs256/batch1 -> ours, all else -> stock", CHUNK, TILE, WARPS)
