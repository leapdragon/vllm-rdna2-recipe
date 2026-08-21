#!/usr/bin/env python3
# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
"""Steady-state decode rate at a given context length (TTFT excluded).

Builds the prompt against the server's own tokenizer so the context length is
exact, then reports the median inter-token gap during pure decode.
"""
import argparse, json, statistics, time, urllib.request, uuid

def ntok(base, model, t):
    r = urllib.request.Request(base+"/tokenize",
        data=json.dumps({"model": model, "prompt": t}).encode(),
        headers={"Content-Type": "application/json"})
    return json.load(urllib.request.urlopen(r, timeout=180))["count"]

def run(base, model, target_ctx, out_tokens):
    u = uuid.uuid4().hex[:6]
    unit = " ".join(f"{u}{i} record {i} value {i*3}" for i in range(200)) + "\n"
    per = ntok(base, model, unit)
    filler = unit * max(1, int(target_ctx/per))
    body = json.dumps({"model": model,
        "messages": [{"role": "user", "content": filler + "\n\nWrite a long essay about indexing."}],
        "max_tokens": out_tokens, "temperature": 0.0, "stream": True,
        "stream_options": {"include_usage": True},
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(base+"/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    ts = []; usage = None; t0 = time.perf_counter(); ttft = None
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
                if ttft is None: ttft = now - t0
                ts.append(now)
    gaps = sorted(ts[i]-ts[i-1] for i in range(1, len(ts)))
    med = statistics.median(gaps)
    return usage["prompt_tokens"], med*1000, 1.0/med, ttft

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8000")
    ap.add_argument("--model", default="qwen38-27b-gptq")
    ap.add_argument("--ctx", default="200,43000")
    ap.add_argument("--out", type=int, default=120)
    a = ap.parse_args()
    print(f"{'context':>9} {'ms/token':>10} {'decode t/s':>11} {'TTFT s':>8}")
    print("-"*42)
    for c in [int(x) for x in a.ctx.split(",")]:
        p, ms, tps, ttft = run(a.base, a.model, c, a.out)
        print(f"{p:>9} {ms:>10.2f} {tps:>11.2f} {ttft:>8.1f}")
