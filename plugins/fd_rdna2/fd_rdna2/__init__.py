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
    from .fd_kernel2 import fd2_decode, permute_q

    global _log
    logger = _log = init_logger("vllm.fd_rdna2")
    if getattr(ta, "_fd_rdna2_patched", False):
        return
    orig = ta.unified_attention

    CHUNK = int(os.getenv("FD_CHUNK", "512"))
    TILE = int(os.getenv("FD_TILE", "32"))
    WARPS = int(os.getenv("FD_WARPS", "8"))
    _ws = {}

    def patched(**kw):
        q = kw.get("q")
        try:
            ok = (
                kw.get("max_seqlen_q") == 1                       # pure decode
                and kw.get("kv_quant_mode") == KVQuantMode.INT8_PER_TOKEN_HEAD
                and q is not None and q.dim() == 3
                and q.shape[0] == 1                               # batch 1 (v1 scope)
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
            m = torch.empty(gchunks * n_kv * PAD, dtype=torch.float32, device=dev)
            buf = (m, torch.empty_like(m),
                   torch.empty(gchunks * n_kv * PAD * 256, dtype=torch.float32, device=dev),
                   torch.empty(q.shape[1], 256, dtype=torch.float16, device=dev),
                   torch.zeros(n_kv, 4, 64, PAD, dtype=torch.float16, device=dev))
            _ws[(gchunks, n_kv)] = buf
            logger.info("fd_rdna2: buffers for max_ctx=%d (%d chunks of %d)",
                        max_ctx, gchunks, CHUNK)
        m, l, a, out_buf, qbuf = buf

        qs = (q[0] * kw["softmax_scale"]).to(torch.float16)       # (n_q, 256)
        qp = permute_q(qs, GQA=6, KVH=n_kv, out=qbuf)
        fd2_decode(qp, i32, f32, bt, 0, BS=bs, CHUNK=CHUNK,
                   TILE=TILE, GQA=6, KVH=n_kv, num_warps=WARPS,
                   workspace=(m, l, a), strides=(s_blk, s_kvh, s_tok),
                   seq_ptr=kw["seqused_k"], grid_chunks=gchunks, out=out_buf)
        kw["out"][0].copy_(out_buf.view(q.shape[1], 256))
        _stats["fast"] += 1
        return None

    ta.unified_attention = patched
    ta._fd_rdna2_patched = True
    ta._fd_rdna2_stats = _stats
    logger.info("fd_rdna2: flash-decode override active (CHUNK=%d TILE=%d warps=%d); "
                "decode/int8/hs256/batch1 -> ours, all else -> stock", CHUNK, TILE, WARPS)
