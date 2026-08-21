#!/usr/bin/env python3
# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
"""Sustained mixed-traffic soak, written to survive a hard crash.

The previous run's verdict was lost because it printed only at the end, to tmpfs. This one
appends a timestamped line every interval to runlogs/, flushed and fsync'd, so an abrupt power
loss leaves the last known-good state on disk. GPU power/temp/clock are sampled alongside the
request counters, because the leading crash hypothesis is a marginal supply under multi-GPU
load — if the machine dies, the final line shows what the cards were drawing when it did.
"""
import argparse, json, os, subprocess, sys, threading, time, urllib.request, uuid

ap = argparse.ArgumentParser()
ap.add_argument("--minutes", type=float, default=20)
ap.add_argument("--conc", type=int, default=3)
ap.add_argument("--interval", type=float, default=15, help="seconds between progress lines")
ap.add_argument("--base", default="http://127.0.0.1:8000")
ap.add_argument("--devices", default="0,1,3")
ap.add_argument("--tag", default="")
a = ap.parse_args()

REPO = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.makedirs(os.path.join(REPO, "runlogs"), exist_ok=True)
stamp = time.strftime("%Y%m%d-%H%M%S")
path = os.path.join(REPO, "runlogs", f"soak-{stamp}{('-'+a.tag) if a.tag else ''}.log")
fh = open(path, "a", buffering=1)

def emit(line):
    """Write, flush and fsync: a crash must not cost us the last line."""
    fh.write(line + "\n"); fh.flush(); os.fsync(fh.fileno())
    print(line, flush=True)

def gpu():
    """power/temp/sclk per device, one rocm-smi call."""
    out = []
    try:
        r = subprocess.run(["rocm-smi", "--showpower", "--showtemp", "--showgpuclocks"],
                           capture_output=True, text=True, timeout=15).stdout
        import re
        cur = {}
        for ln in r.splitlines():
            m = ln.split("GPU[")
            if len(m) < 2: continue
            d = m[1].split("]")[0].strip()
            if "Power" in ln:
                g = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*$", ln.strip())
                if g: cur.setdefault(d, {})["W"] = g.group(1)
            elif "Temperature" in ln and "junction" not in ln.lower() and "mem" not in ln.lower():
                g = re.search(r"([0-9]+(?:\.[0-9]+)?)\s*$", ln.strip())
                if g: cur.setdefault(d, {})["C"] = g.group(1)
            elif "sclk" in ln.lower():
                g = re.search(r"\(?([0-9]+)\s*Mhz\)?", ln, re.I)   # "level: 7: (2460Mhz)"
                if g: cur.setdefault(d, {})["MHz"] = g.group(1)
        for d in a.devices.split(","):
            c = cur.get(d, {})
            out.append(f"g{d}={c.get('W','?')}W/{c.get('C','?')}C/{c.get('MHz','?')}MHz")
    except Exception as e:
        out.append(f"gpu-sample-failed:{type(e).__name__}")
    return " ".join(out)

stop_at = time.time() + a.minutes * 60
st = {"ok": 0, "err": 0, "empty": 0, "tok": 0, "inflight": 0}
lock = threading.Lock(); errs = []; done = threading.Event()

def one(words, mt):
    u = uuid.uuid4().hex[:6]
    filler = " ".join(f"{u}{i} record {i} value {i*3}" for i in range(words))
    body = json.dumps({"model": "qwen38-27b-gptq",
        "messages": [{"role": "user", "content": filler + "\n\nSummarise the above in one sentence, then count to five."}],
        "max_tokens": mt, "temperature": 0.0,
        "chat_template_kwargs": {"enable_thinking": False}}).encode()
    r = urllib.request.Request(a.base + "/v1/chat/completions", data=body,
                               headers={"Content-Type": "application/json"})
    d = json.load(urllib.request.urlopen(r, timeout=900))
    return (d["choices"][0]["message"].get("content") or ""), d["usage"]["completion_tokens"]

def worker(w):
    sizes = [(50, 32), (400, 64), (1500, 48), (3000, 32)]
    i = 0
    while time.time() < stop_at:
        words, mt = sizes[i % len(sizes)]; i += 1
        with lock: st["inflight"] += 1
        try:
            c, n = one(words, mt)
            with lock:
                if not c.strip(): st["empty"] += 1; errs.append(f"w{w} empty at words={words}")
                else: st["ok"] += 1; st["tok"] += n
        except Exception as e:
            with lock: st["err"] += 1; errs.append(f"w{w} {type(e).__name__}: {str(e)[:80]}")
        finally:
            with lock: st["inflight"] -= 1

def reporter():
    t0 = time.time()
    while not done.wait(a.interval):
        with lock: s = dict(st)
        emit(f"{time.strftime('%H:%M:%S')} +{(time.time()-t0)/60:5.1f}m "
             f"ok={s['ok']:<4} err={s['err']} empty={s['empty']} inflight={s['inflight']} "
             f"tok={s['tok']:<6} {gpu()}")

emit(f"# soak start {time.strftime('%Y-%m-%d %H:%M:%S')} minutes={a.minutes} conc={a.conc} "
     f"devices={a.devices} pid={os.getpid()}")
emit(f"# power caps: " + (subprocess.run(['rocm-smi','--showmaxpower'],capture_output=True,text=True).stdout
                          .replace('\n',' ').split('Power Cap')[-1][:160] if True else ''))
ts = [threading.Thread(target=worker, args=(i,), daemon=True) for i in range(a.conc)]
rp = threading.Thread(target=reporter, daemon=True); rp.start()
t0 = time.time()
for t in ts: t.start()
for t in ts: t.join()
done.set(); rp.join(timeout=5)
el = time.time() - t0
verdict = "CLEAN" if st["err"] == 0 and st["empty"] == 0 else "FAILED"
emit(f"# soak end {el/60:.1f} min: ok={st['ok']} err={st['err']} empty={st['empty']} tokens={st['tok']}")
for e in errs[:8]: emit(f"#   {e}")
emit(f"SOAK {verdict}")
print(f"\n  log: {path}")
sys.exit(0 if verdict == "CLEAN" else 1)
