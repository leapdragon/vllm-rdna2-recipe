#!/usr/bin/env python3
# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
"""Decode rate that is honest under speculative decoding.

Two prior methods were both wrong for MTP:
  - median inter-token gap: deltas can carry multiple tokens, so counting deltas
    understates tokens and overstates ms/token;
  - total wall / completion_tokens: includes the partial-block re-prefill, which
    at 3.4k context is ~2.4 s of GEMMs charged to decode.
This streams, times first->last delta (excluding prefill), and takes the token
count from usage, not from delta count."""
import argparse, json, statistics, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="qwen38-27b-gptq")
ap.add_argument("--ctx", default="200,4000,16000,43000")
ap.add_argument("--out", type=int, default=120)
ap.add_argument("--trials", type=int, default=2)
a = ap.parse_args()

def ntok(t):
    r = urllib.request.Request(a.base+"/tokenize",
        data=json.dumps({"model": a.model, "prompt": t}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=300))["count"]

def run(target):
    u = uuid.uuid4().hex[:6]
    unit = " ".join(f"{u}{i} record {i} value {i*3}" for i in range(200)) + "\n"
    per = ntok(unit)
    filler = unit * max(1, int(target/per))
    body = json.dumps({"model": a.model,
        "messages": [{"role": "user", "content": filler + "\n\nWrite a long essay about indexing."}],
        "max_tokens": a.out, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    def once():
        r = urllib.request.Request(a.base+"/v1/chat/completions", data=body,
                                   headers={"Content-Type": "application/json"})
        t_first = t_last = None; usage = None; t0 = time.perf_counter()
        with urllib.request.urlopen(r, timeout=3600) as resp:
            for raw in resp:
                l = raw.decode().strip()
                if not l.startswith("data: "): continue
                d = l[6:]
                if d == "[DONE]": break
                j = json.loads(d)
                if j.get("usage"): usage = j["usage"]
                c = j.get("choices") or []
                if c and c[0].get("delta", {}).get("content"):
                    now = time.perf_counter()
                    if t_first is None: t_first = now
                    t_last = now
        n = usage["completion_tokens"]
        return usage["prompt_tokens"], (t_last - t_first)/(n - 1)*1000, t_first - t0
    once()                                     # warm: prefix-cache + JIT for this shape
    res = [once() for _ in range(a.trials)]
    pt = res[0][0]
    ms = statistics.median(r[1] for r in res)
    ttft = statistics.median(r[2] for r in res)
    return pt, ms, ttft

print(f"  context   ms/token  decode t/s   TTFT s")
print("  " + "-"*42)
for tgt in (int(x) for x in a.ctx.split(",")):
    pt, ms, ttft = run(tgt)
    print(f"  {pt:>7} {ms:>10.2f} {1000/ms:>11.2f} {ttft:>8.1f}")
