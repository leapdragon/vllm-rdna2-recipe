# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
"""Offline sweep of fused_moe int4_w4a16 configs at the 122B's exact shapes.

E=256 experts, N=1024 (moe_intermediate), K=3072 (hidden), topk=8, group 128.
Measures the full expert MLP (gate_up w13 + down w2) per config for decode-like
M and prefill-like M. Weights sized to exceed Infinity Cache via layer rotation.
"""
import itertools, time, torch
from vllm.model_executor.layers.fused_moe.fused_moe import fused_experts
from vllm.model_executor.layers.fused_moe.config import int4_w4a16_moe_quant_config
import vllm.model_executor.layers.fused_moe.fused_moe as fm

E, N, K, TOPK, G = 256, 1024, 3072, 8, 128
DEV = "cuda:0"
torch.manual_seed(0)

def make_layer():
    w13 = torch.randint(0, 255, (E, 2*N, K//2), dtype=torch.uint8, device=DEV)
    w2  = torch.randint(0, 255, (E, K, N//2),  dtype=torch.uint8, device=DEV)
    s13 = torch.rand(E, 2*N, K//G, dtype=torch.float16, device=DEV)*0.01
    s2  = torch.rand(E, K, N//G,  dtype=torch.float16, device=DEV)*0.01
    qc = int4_w4a16_moe_quant_config(w1_scale=s13, w2_scale=s2, block_shape=[0, G])
    return w13, w2, qc

LAYERS = [make_layer() for _ in range(4)]   # rotate to defeat cache residency

def run(M, cfg, iters=20):
    orig = fm.try_get_optimal_moe_config
    def forced(*a, **k):
        c = dict(cfg)
        return c
    fm.try_get_optimal_moe_config = forced
    try:
        x = torch.randn(M, K, dtype=torch.float16, device=DEV)
        tw = torch.randn(M, E, device=DEV)
        topk_w, topk_ids = torch.topk(torch.softmax(tw.float(), -1), TOPK)
        topk_w = topk_w.to(torch.float32); topk_ids = topk_ids.to(torch.int32)
        for i in range(3):
            w13, w2, qc = LAYERS[i % len(LAYERS)]
            fused_experts(x, w13, w2, topk_w, topk_ids, quant_config=qc)
        torch.cuda.synchronize()
        t0 = time.time()
        for i in range(iters):
            w13, w2, qc = LAYERS[i % len(LAYERS)]
            fused_experts(x, w13, w2, topk_w, topk_ids, quant_config=qc)
        torch.cuda.synchronize()
        return (time.time() - t0) / iters * 1000
    finally:
        fm.try_get_optimal_moe_config = orig

def lds_ok(c):
    a = c["BLOCK_SIZE_M"] * c["BLOCK_SIZE_K"] * 2
    b = c["BLOCK_SIZE_N"] * c["BLOCK_SIZE_K"] * 2
    return (a + b) * max(1, c["num_stages"]) <= 60000

results = {}
for M in (8, 64, 8192):   # decode (1 tok x topk8), small batch, prefill chunk
    cands = []
    for bm, bn, bk, w, st in itertools.product(
            (16, 32, 64), (32, 64, 128, 256), (32, 64, 128), (2, 4, 8), (1, 2)):
        if M >= 1024 and bm < 32: continue
        if M <= 8 and bm > 16: continue
        c = {"BLOCK_SIZE_M": bm, "BLOCK_SIZE_N": bn, "BLOCK_SIZE_K": bk,
             "GROUP_SIZE_M": 1, "SPLIT_K": 1, "num_warps": w, "num_stages": st}
        if lds_ok(c): cands.append(c)
    best = None
    print(f"== M={M}: {len(cands)} candidates")
    for c in cands:
        try:
            ms = run(M, c, iters=10 if M >= 1024 else 20)
        except Exception as e:
            if not results.get(('err', M)):
                results[('err', M)] = 1; print('   first error:', type(e).__name__, str(e)[:90])
            continue
        if best is None or ms < best[0]:
            best = (ms, c)
            print(f"   {ms:7.2f} ms  {c}")
    results[M] = best
print("\n=== winners ===")
for M, v in results.items():
    if isinstance(M, tuple): continue
    if v: print(f"M={M}: {v[0]:.2f} ms  {v[1]}")
