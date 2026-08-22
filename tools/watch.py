#!/usr/bin/env python3
# Copyright (C) 2026 Aron Hsiao
# SPDX-License-Identifier: GPL-3.0-or-later
# Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
"""llama-server-style runtime reporting for vLLM, derived from /metrics deltas.

WHAT THIS NEEDS TO WORK — assumptions and scope:

  * A vLLM V1 OpenAI-compatible server whose Prometheus endpoint is reachable
    at --base (default http://127.0.0.1:8000/metrics). Docker is NOT required:
    the tool speaks plain HTTP. It works with this recipe's container because
    the launcher runs --network=host; for other setups, point --base at
    wherever the port is published.
  * vLLM 0.27-era V1 metric names (vllm:prompt_tokens_total,
    vllm:generation_tokens_total, vllm:iteration_tokens_total_sum,
    vllm:prefix_cache_{queries,hits}_total, vllm:num_requests_{running,waiting},
    vllm:kv_cache_usage_perc). Different vLLM versions may rename these; if
    every column reads "--" while the server is clearly busy, check the names
    in `curl /metrics` against the WANT/GAUGE tables below.
  * One engine per endpoint. Values are summed across label sets, so a
    multi-engine (data-parallel) server would blend its engines together.
  * The PREFILL-state inference assumes decode publishes tokens every few
    engine steps (true for normal serving). The `dp < 100` floor that filters
    the speculative-decoding residual assumes MTP-style drafting of a few
    tokens per step (this recipe's MTP=2); it is harmless when speculation
    is off.
  * Python 3 standard library only. The self-overwriting idle/status line
    needs a real terminal; when stdout is piped, it degrades to one plain
    heartbeat line per --heartbeat interval.

A vLLM V1 quirk shapes this tool: ALL /metrics values (counters and gauges alike)
publish only alongside output processing, and a request that is mid-prefill produces
no output — so every metric freezes for the duration of a prefill, then lands in one
batch with the first generated token. Two consequences handled here:
  - "is it prefilling?" is inferred live: a running request that published zero
    tokens over a full tick can only be inside a prefill chunk (decode publishes
    every ~20 ms). Shown as PREFILL with a running clock.
  - the prefill RATE is only knowable when the batch lands; it is shown as the
    average over the just-finished burst (tokens / burst wall time), never as the
    bogus one-tick spike the raw counters would suggest.

Prints one line per interval whenever the engine has work, answering at a glance:
what is it doing right now (PREFILL / DECODE / P+D / stalled), how far the current
prefill burst has progressed, and the current prefill/decode token rates. Also shows
the interval's prefix-cache hit rate (not lifetime) and queue/KV state.

Complements two native mechanisms this launcher already enables:
  - the engine's own periodic line (docker logs; STATS_INTERVAL env to tune)
  - per-request cached-token counts in every API response
    (usage.prompt_tokens_details.cached_tokens, via --enable-prompt-tokens-details)

Per-request progress percentages are not derivable from /metrics (vLLM exposes
prompt-size histograms only for *finished* requests), so prefill progress is shown
as tokens accumulated in the current burst — with one running request, that IS the
progress through its prompt.
"""
import argparse, re, sys, time, urllib.request

ap = argparse.ArgumentParser()
ap.add_argument("--base", default="http://127.0.0.1:8000")
ap.add_argument("--interval", type=float, default=1.0)
ap.add_argument("--heartbeat", type=float, default=60.0,
                help="print an idle line at least this often")
a = ap.parse_args()

# the prometheus client appends _total to counters in exposition
WANT = {"vllm:prompt_tokens_total": "prompt", "vllm:generation_tokens_total": "gen",
        "vllm:iteration_tokens_total_sum": "iter",
        "vllm:prefix_cache_queries_total": "cq", "vllm:prefix_cache_hits_total": "ch"}
GAUGE = {"vllm:num_requests_running": "run", "vllm:num_requests_waiting": "wait",
         "vllm:kv_cache_usage_perc": "kv", "vllm:gpu_cache_usage_perc": "kv"}

def scrape():
    out = {}
    with urllib.request.urlopen(a.base + "/metrics", timeout=30) as r:
        for ln in r.read().decode().splitlines():
            if ln.startswith("#"): continue
            m = re.match(r"([a-z_:0-9]+)(?:\{[^}]*\})? ([0-9.e+-]+)", ln)
            if not m: continue
            name, val = m.group(1), float(m.group(2))
            if name in WANT: out[WANT[name]] = out.get(WANT[name], 0.0) + val
            if name in GAUGE: out[GAUGE[name]] = val
    return out

def fmt_tok(n):
    return f"{n/1000:.1f}k" if n >= 1000 else f"{int(n)}"

IS_TTY = sys.stdout.isatty()
_status_open = False

def emit_status(text):
    """Transient bottom line: overwrites itself, never scrolls history away."""
    global _status_open
    print("\r\x1b[K" + text, end="", flush=True)
    _status_open = True

def emit_line(text):
    """Permanent scrolling line; clears any open status line first."""
    global _status_open
    if _status_open:
        print("\r\x1b[K", end="")
        _status_open = False
    print(text, flush=True)

def fmt_dur(sec):
    sec = int(sec)
    if sec < 60: return f"{sec}s"
    if sec < 3600: return f"{sec//60}m{sec%60:02d}s"
    return f"{sec//3600}h{(sec%3600)//60:02d}m"

print(f"  watching {a.base} every {a.interval:g}s (ctrl-c to stop)", flush=True)
prev = None; last_print = 0.0
while prev is None:
    try: prev = scrape()
    except Exception as e:
        msg = f"  {time.strftime('%H:%M:%S')} | waiting for server ({type(e).__name__})"
        if IS_TTY: emit_status(msg)
        elif time.time() - last_print > a.heartbeat:
            emit_line(msg); last_print = time.time()
        time.sleep(max(a.interval, 2.0))
last_activity = time.time()
burst = 0.0; burst_t0 = None  # tokens landed in, and start of, the current prefill burst
while True:
    t0 = time.time()
    time.sleep(a.interval)
    try: cur = scrape()
    except Exception as e:
        msg = f"  {time.strftime('%H:%M:%S')} | metrics unreachable ({type(e).__name__}) — retrying"
        if IS_TTY: emit_status(msg)
        elif time.time() - last_print > a.heartbeat:
            emit_line(msg); last_print = time.time()
        continue
    dt = time.time() - t0
    dg = cur.get("gen", 0) - prev.get("gen", 0)
    # Live prefill rate from per-iteration tokens (updates every engine step,
    # unlike prompt_tokens_total which lands only when a prefill completes).
    # Subtracting gen leaves prefill; the floor absorbs the MTP residual
    # (scheduled draft tokens minus accepted ones, ~half the gen rate).
    di = cur.get("iter", 0) - prev.get("iter", 0)
    dp = max(0.0, di - dg)
    if dp / dt < 100 and dg > 0: dp = 0.0
    dq = cur.get("cq", 0) - prev.get("cq", 0)
    dh = cur.get("ch", 0) - prev.get("ch", 0)
    run, wait = int(cur.get("run", 0)), int(cur.get("wait", 0))
    now = time.time()

    in_chunk = run > 0 and di == 0        # running but nothing published: mid-prefill
    if in_chunk and burst_t0 is None:
        burst_t0 = now - dt               # burst began roughly when this tick started
    if dp > 0 and burst_t0 is None:
        burst_t0 = now - dt               # short prefill landed within one tick
    if dp > 0:
        burst += dp
    if run == 0 and wait == 0 and not in_chunk and dp == 0:
        burst = 0.0; burst_t0 = None      # engine drained: close the burst

    if in_chunk:          state = "PREFILL"
    elif dp > 0 and dg > 0: state = "P+D    "
    elif dp > 0:          state = "PREFILL"
    elif dg > 0:          state = "DECODE "
    elif wait > 0:        state = "QUEUED "
    else:                 state = "idle   "

    busy = dp > 0 or dg > 0 or run > 0 or wait > 0
    if not busy:
        idle_msg = (f"  {time.strftime('%H:%M:%S')} | idle {fmt_dur(now - last_activity)}"
                    f" (last activity above)" )
        if IS_TTY: emit_status(idle_msg)
        elif now - last_print > a.heartbeat:
            emit_line(idle_msg); last_print = now
        prev = cur
        continue
    last_activity = now
    if True:
        bits = [time.strftime("%H:%M:%S"), state]
        if in_chunk:
            bits.append(f"prefill  ...     ({now-burst_t0:4.0f}s in-chunk, stats on completion)")
        elif dp > 0 and burst_t0 is not None:
            avg = burst / max(now - burst_t0, dt)
            bits.append(f"prefill {avg:6.0f} t/s avg (≈{fmt_tok(burst)} this burst)")
        else:
            bits.append("prefill      --")
        bits.append(f"gen {dg/dt:6.1f} t/s" if dg > 0 else "gen     --")
        bits.append(f"cache {dh/dq*100:5.1f}% hit ({int(dh)}/{int(dq)})" if dq else "cache    --")
        bits.append(f"run {run} wait {wait}")
        if "kv" in cur: bits.append(f"kv {cur['kv']*100:.1f}%")
        emit_line("  " + " | ".join(bits))
        last_print = now
    prev = cur
