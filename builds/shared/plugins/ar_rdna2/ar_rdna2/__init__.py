# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
"""WS2 push-based all-reduce for RDNA2 TP=2 — vLLM general plugin.

Replaces RCCL for small tensor-parallel all-reduces on gfx1030. Anything this kernel cannot
handle (world_size != 2, wrong dtype, oversized, non-contiguous) delegates to the stock path.

Patch point: GroupCoordinator._all_reduce_out_place — the single funnel both the custom-op
path and the direct path go through. Enable with AR_RDNA2=1.

Why this works where vLLM's own custom all-reduce does not (T18, PROFILE §9): its barrier
spins on ordinary coarse-grained device memory, which a peer's write never makes visible, so
the workers hang. Here the staging buffers are uncached and the flags are host-resident.
"""
import os

_stats = {"fast": 0, "slow": 0, "reason": {}}
_log = None


def _reject(why):
    _stats["slow"] += 1
    n = _stats["reason"].get(why, 0) + 1
    _stats["reason"][why] = n
    if _log is not None and n == 1:
        _log.info("ar_rdna2: delegating to stock all-reduce (%s)", why)
    return None


def _build(rank):
    """Compile the HIP extension. Per-rank build dirs: concurrent ninja builds into one
    directory race, and the compile is ~1 minute either way."""
    from torch.utils.cpp_extension import load
    here = os.path.dirname(os.path.abspath(__file__))
    csrc = os.path.join(here, "csrc")
    build = os.path.join(os.environ.get("TORCH_EXTENSIONS_DIR", "/tmp/ar_rdna2"), f"rank{rank}")
    os.makedirs(build, exist_ok=True)
    return load(name="ar_ext", sources=[os.path.join(csrc, "ar_ext.hip")],
                extra_cflags=["-O3"], extra_include_paths=[csrc],
                extra_cuda_cflags=["-O3", "--offload-arch=gfx1030"],
                build_directory=build, verbose=False)


class _Chan:
    """One initialised channel for a 2-rank group."""

    def __init__(self, group, max_bytes):
        import torch
        import torch.distributed as dist
        self.ext = _build(group.rank_in_group)
        rank = group.rank_in_group
        peer_dev = 1 - rank            # TP ranks map to local device ordinals 0..n-1
        name = f"/vllm_ar_{group.unique_name}".replace("/", "_", 1)[:200]
        handle = None
        # Ordered init: rank 0 recreates the shm object; rank 1 must not open a stale one.
        for r in range(2):
            if r == rank:
                handle = self.ext.init(rank, peer_dev, max_bytes, name)
            dist.barrier(group=group.cpu_group)
        gathered = [None, None]
        dist.all_gather_object(gathered, handle, group=group.cpu_group)
        self.ext.connect(gathered[1 - rank])
        dist.barrier(group=group.cpu_group)
        if _log:
            _log.info("ar_rdna2: channel up (rank %d, peer dev %d, max %d KB)",
                      rank, peer_dev, max_bytes // 1024)


def register():
    if os.getenv("AR_RDNA2", "0") != "1":
        return
    import torch
    from vllm.logger import init_logger
    import vllm.distributed.parallel_state as ps

    global _log
    # vLLM only configures handlers under the "vllm" namespace; a bare name logs into the void.
    logger = _log = init_logger("vllm.ar_rdna2")
    if getattr(ps, "_ar_rdna2_patched", False):
        return

    MAX_BYTES = int(os.getenv("AR_MAX_KB", "512")) * 1024
    orig = ps.GroupCoordinator._all_reduce_out_place
    chans = {}

    def patched(self, input_):
        if self.world_size != 2:
            return orig(self, input_)
        key = self.unique_name
        ch = chans.get(key)
        if ch is None:
            if key in chans:                       # a previous attempt failed; do not retry
                return orig(self, input_)
            try:
                ch = chans[key] = _Chan(self, MAX_BYTES)
            except Exception as e:
                chans[key] = None
                logger.warning("ar_rdna2: channel init failed (%s); using stock all-reduce", e)
                return orig(self, input_)
        if not ch.ext.can(input_):
            _reject("shape/dtype/size")
            return orig(self, input_)
        out = ch.ext.all_reduce(input_)
        _stats["fast"] += 1
        return out

    ps.GroupCoordinator._all_reduce_out_place = patched
    ps._ar_rdna2_patched = True
    ps._ar_rdna2_stats = _stats
    logger.info("ar_rdna2: push-based all-reduce active (max %d KB); "
                "TP=2 small all-reduces -> ours, everything else -> stock", MAX_BYTES // 1024)
