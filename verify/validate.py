#!/usr/bin/env python3
# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
"""Greedy-decode a fixed prompt set and save/compare outputs.

Used as the correctness gate when changing the collective path: a broken
all-reduce usually does NOT crash, it quietly corrupts activations. Comparing
temperature-0 outputs against a known-good run catches that.

  validate.py --save baseline.json
  validate.py --compare baseline.json
"""
import argparse, json, sys, time, urllib.request, difflib

PROMPTS = [
    "What is 17 times 23? Reply with just the number.",
    "List the first 10 prime numbers, comma separated, nothing else.",
    "Name the capital of Australia in one word.",
    "Write a Python function that reverses a linked list. Code only.",
    "Explain in exactly three sentences why the sky appears blue.",
    "Translate to French: 'The quick brown fox jumps over the lazy dog.'",
    "What is the derivative of x^3 * sin(x)? Show the result only.",
    "Summarise the causes of the 1929 Wall Street Crash in four bullet points.",
]

def run(url, model, p, maxtok=220):
    body = json.dumps({"model": model, "messages": [{"role": "user", "content": p}],
                       "max_tokens": maxtok, "temperature": 0.0, "seed": 0,
                       "chat_template_kwargs": {"enable_thinking": False}}).encode()
    req = urllib.request.Request(url, data=body, headers={"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=900) as r:
        j = json.load(r)
    return j["choices"][0]["message"]["content"]

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--url", default="http://127.0.0.1:8000/v1/chat/completions")
    ap.add_argument("--model", default="qwen38-27b-gptq")
    ap.add_argument("--save"); ap.add_argument("--compare")
    a = ap.parse_args()

    outs = {}
    for i, p in enumerate(PROMPTS):
        t0 = time.perf_counter()
        outs[p] = run(a.url, a.model, p)
        print(f"  [{i+1}/{len(PROMPTS)}] {time.perf_counter()-t0:5.1f}s  {p[:52]}")

    if a.save:
        json.dump(outs, open(a.save, "w"), indent=1)
        print(f"\nsaved {len(outs)} outputs -> {a.save}")

    if a.compare:
        base = json.load(open(a.compare))
        ident = diff = 0
        for p, txt in outs.items():
            b = base.get(p)
            if b is None: continue
            if b.strip() == txt.strip():
                ident += 1
            else:
                diff += 1
                print(f"\n--- DIVERGED: {p[:60]}")
                for line in list(difflib.unified_diff(
                        b.split(), txt.split(), lineterm="", n=1))[:14]:
                    print("   ", line)
        print(f"\nidentical={ident}  diverged={diff}  of {ident+diff}")
        # Greedy decode may differ in fp rounding, so judge on the checkable ones.
        hard = {"What is 17 times 23? Reply with just the number.": "391",
                "Name the capital of Australia in one word.": "canberra"}
        bad = [p for p, want in hard.items()
               if p in outs and want.lower() not in outs[p].lower()]
        print("factual spot-checks:", "PASS" if not bad else f"FAIL -> {bad}")
        sys.exit(1 if bad else 0)

if __name__ == "__main__":
    main()
