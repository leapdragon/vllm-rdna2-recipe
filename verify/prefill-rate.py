#!/usr/bin/env python3
# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
"""Prompt-processing (prefill) rate on COLD prompts.

Every request uses a unique filler so the prefix cache can never serve it; TTFT is
the wall from request start to the first streamed content delta. Rate = prompt
tokens / TTFT (the one decode step inside TTFT is <100 ms, noise at these sizes)."""
import argparse, json, statistics, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8000")
ap.add_argument("--model", default="qwen38-27b-gptq")
ap.add_argument("--ctx", default="4000,8000,16000,32000,43000")
ap.add_argument("--trials", type=int, default=2)
a = ap.parse_args()

def ntok(t):
    r = urllib.request.Request(a.base+"/tokenize",
        data=json.dumps({"model": a.model, "prompt": t}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=300))["count"]

unit_tok = None
def once(target):
    global unit_tok
    u = uuid.uuid4().hex[:6]                       # fresh -> never prefix-cached
    unit = " ".join(f"{u}{i} record {i} value {i*3}" for i in range(200)) + "\n"
    if unit_tok is None: unit_tok = ntok(unit)
    filler = unit * max(1, int(target/unit_tok))
    body = json.dumps({"model": a.model,
        "messages": [{"role": "user", "content": filler + "\n\nSay OK."}],
        "max_tokens": 4, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(a.base+"/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    t0 = time.perf_counter(); ttft = None; usage = None
    with urllib.request.urlopen(r, timeout=3600) as resp:
        for raw in resp:
            l = raw.decode().strip()
            if not l.startswith("data: "): continue
            d = l[6:]
            if d == "[DONE]": break
            j = json.loads(d)
            if j.get("usage"): usage = j["usage"]
            c = j.get("choices") or []
            if c and c[0].get("delta", {}).get("content") and ttft is None:
                ttft = time.perf_counter() - t0
    return usage["prompt_tokens"], ttft

print(f"  {'prompt tok':>10} {'TTFT s':>8} {'prefill tok/s':>14}")
print("  " + "-"*36)
for tgt in (int(x) for x in a.ctx.split(",")):
    res = [once(tgt) for _ in range(a.trials)]
    pt = res[0][0]
    ttft = statistics.median(r[1] for r in res)
    print(f"  {pt:>10} {ttft:>8.1f} {pt/ttft:>14.0f}", flush=True)
