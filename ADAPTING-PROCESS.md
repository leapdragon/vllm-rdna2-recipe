# ADAPTING-PROCESS.md — instructions for adapting this recipe

Copyright © 2026 Aron Hsiao · GPL-3.0-or-later (see LICENSE)

**Audience: you are most likely an LLM coding assistant** that has been handed this repository
and asked to reproduce or adapt it — a different RDNA2 card, a different model, a newer vLLM.
This document is written as working instructions for you, verbose enough that your human can
follow the reasoning and check your work. The patches and numbers in this repo are artifacts
of one machine and one model; **what actually transfers is the method below.** Follow the
method and you will re-derive whatever the patches no longer give you. Follow the patches
without the method and you will ship something slow without knowing it.

## Read this first: PROFILE-NAVI21.md is the rosetta stone

Before anything else, read `PROFILE-NAVI21.md` end to end. It is the key that makes this whole
method executable, in two ways:

1. **It shows what "knowing your hardware" concretely means.** Every design decision in this
   repo — which kernel to write, which to not write, which knob to turn — traces to a measured
   number in that file: achievable memory bandwidth (not the spec sheet number), dot-product
   peak, the LDS ceiling, cache sizes, interconnect behavior, the coherence rules. When you
   read our patches and plugins, that file is the decoder ring for *why* each one is shaped
   the way it is.
2. **If your hardware differs from ours, your first substantial task is to produce its
   equivalent for your card.** You do not need all 37 tests on day one. You need, minimum:
   measured achievable DRAM bandwidth, measured fp16/dot-product throughput, the LDS/shared
   memory limit per workgroup, and (if multi-GPU) measured peer-to-peer bandwidth and the
   memory-coherence rules. Every "theoretical floor" in the loop below is computed from these,
   and the floors are what tell you when you are done. Numbers you did not measure yourself
   are guesses; on our machine, the gap between spec and measured was the difference between
   a correct plan and a wasted workstream.

## Phase -1 — check builds/ before adapting anything

Adaptation may already be done for you. Each directory under `builds/` is a model we brought
up and optimized end-to-end, named for its Hugging Face id (`/` → `-`): read its `BUILD.md`
for the measured numbers and working configuration, run its `serve.sh`, and you are at the
end of this document without executing the loop.

If your model is *not* there but is close to one that is (same architecture, different quant;
same quant family, different size), **start by copying the closest build's configuration and
adapting it** — that is the convention we use ourselves. The BUILD.md of the donor tells you
which assumptions you are inheriting; the quant format is the one to check first (symmetric
vs asymmetric W4 selects a different kernel route entirely — see patch 0006 and the kernel
pitfall in 01-PATCHES.md). When your adaptation works, record it as a new `builds/<model-id>/`
directory with its own `serve.sh` and `BUILD.md`, so the next reader starts where you finished.

## Phase 0 — get to a boot before optimising anything

With `00-HARDWARE.md`, `01-PATCHES.md`, `02-VERSIONS.md`, and `PROFILE-NAVI21.md` ingested:

1. **Build the base image** with your GPU architecture in the compile target list. Patch 0005
   is the one-line shape of this. Verify with the device attached (`torch.cuda.get_arch_list()`
   returns an empty list *silently* when no GPU is visible — do not debug a healthy build).
2. **Apply the six patches.** When one fails to apply — expected on any vLLM newer than
   0.27.1 — do not force fuzzy hunks. Each patch's section in `01-PATCHES.md` states its
   *intent* and a verification line: re-implement the intent in the moved code, then run the
   verification. A fuzzy hunk that lands in the wrong place fails silently, which is worse
   than failing loudly.
3. **Boot the launcher** (`config/serve-rdna2-tp2.sh`, paths adapted) and get coherent text
   from a request. Consult the pitfalls section of `01-PATCHES.md` for the failure modes that
   look like something else — the wrong-kernel-dispatched case in particular serves correct
   text at a quarter of the proper speed and logs nothing.
4. **Immediately save your correctness baseline:** `verify/validate.py --save`. This file is
   the referee for every change you make afterward. Greedy outputs on a fixed prompt set are
   your ground truth; any numerics-lossless change must reproduce them byte-for-byte.

Also set up your record-keeping now, because the loop generates knowledge faster than memory
holds it. Maintain three living documents, modeled on this project's discipline:

- a **state file** — current config, current measured numbers, what is adopted vs experimental;
- a **results log** — one entry per experiment: question, method, numbers, verdict. Record
  negative results with the same care as wins; they are what make later iterations cheap;
- a **dead-end register** — refuted approaches *with the evidence*, so neither you nor a
  future session re-attempts them. Our `01-PATCHES.md` dead-end table is the template, and on
  different hardware some of our entries deserve re-testing — but know they failed here first.

## The optimisation loop

Run the loop **twice, as separate campaigns: once for decode (generation) and once for
prefill (prompt processing).** They are different physical regimes and the same idea can win
in one and lose in the other — on our hardware, the custom-kernel approach that gave decode a
6× better context slope was built for prefill too, validated correct, and measured *slower*
than stock, because decode is bandwidth-bound with zero cache reuse while prefill is
compute-bound with heavy reuse. Do decode first; it is the simpler regime and it is where a
serving token spends most of its life.

### Step 1 — profile performance and stability, in the deployed configuration

This is the load-bearing rule of the entire method: **measure the running server, not a
microbenchmark.** Out-of-server measurements repeatedly overstated or invented deficits during
this project (worst case 4×, which cost an entire workstream). The tools in `verify/` exist so
you can do this correctly on the first try:

- `verify/decode-rate.py` — decode tokens/sec, honestly. It streams, excludes prefill from the
  timing, and takes token counts from the API's `usage` field. (Two simpler methods each
  mis-measure under speculative decoding; this replaced them.)
- `verify/prefill-rate.py` — prefill tokens/sec on *cold* prompts (unique text per request, so
  the prefix cache can never serve it). Run 3–4 context sizes and fit
  `avg_ms_per_token = a + b·context/2`: `a` is your linear (weights) term, `b` — the quadratic
  coefficient — is prefill's real health number. If trials at one size are wildly non-uniform,
  something is intruding on your measurement (see the TunableOp pitfall).
- `verify/validate.py --compare` — the correctness gate. 8/8 byte-identical for any
  numerics-lossless change; a single divergence means a bug (or a numerics change you did not
  intend to make).
- `verify/soak.py` — sustained concurrent load with per-interval progress and GPU
  power/temperature/clock *fsync'd to disk*, so if the machine hard-crashes you still have the
  last known state as evidence. Run it before calling anything production-ready.
- **Attribution** — vLLM's built-in profiler: boot with `--profiler-config.profiler=torch
  --profiler-config.torch_profiler_dir=<dir>`, then POST `/start_profile`, run one warm
  request, POST `/stop_profile`, and parse the per-kernel GPU-time table it writes. The
  acceptance test for any profile you produce: **the components must sum to the measured
  token time.** If they do not, your attribution is wrong and everything you size from it
  will be wrong too.

While profiling, compute the **theoretical floors** from your rosetta-stone numbers, because
the floors define "done":

- **decode floor** = bytes streamed per token ÷ measured achievable bandwidth. Get the bytes
  by reading the model's safetensors headers and summing tensor sizes — count, never estimate;
  remember embeddings are *not* streamed (one row is read) while an untied lm_head *is*.
- **prefill floor** = attention FLOPs per token (2 × 2 × query_heads × head_dim × context,
  for QK plus PV) ÷ measured dot-product peak.

### Step 2 — identify optimisation potential and paths

Subtract the floor from each profiled component: the difference is the *ceiling* of any
optimisation aimed at that component. A component running at 91% of its floor is finished no
matter how clever your kernel idea is — we cancelled a planned workstream on exactly that
arithmetic. For components with real ceilings, research candidates: read the dispatch code to
see which kernel actually runs (do not assume), read what llama.cpp does for the same
operation on your hardware class, search upstream vLLM PRs for your architecture. Check the
dead-end register — ours and yours — before adding a candidate.

### Step 3 — rank candidates, best first, in performance-stability terms

Order by expected value = component ceiling × plausibility of reaching it, discounted by risk:

1. **Config-only changes** (environment variables, launch parameters, compilation modes):
   reversible by flag, numerics-lossless, one boot to test. Do not assume these are minor —
   the two largest single wins of this entire project were one launch parameter
   (`num_stages=1`, +35% prefill at 37k) and one config mode (compilation, +20% decode).
2. **Numerics-lossless code changes** (kernel scheduling, tiling, dispatch routing):
   gate with byte-identical `validate.py`.
3. **Numerics-changing options last** (quantising a layer, replacing a numeric path,
   speculative-decoding drafts): these cost you the byte-identical gate, require a quality
   argument, and force a re-baselining decision. **Re-baselining is your human's call, not
   yours** — present the evidence and wait.

Estimate cost in **machine time** — boots (~9 minutes each on our machine) plus measurement
runs — not in engineer-days. At a few hours per attempt, any candidate whose ceiling exceeds
~2% is worth trying: the cost is paid once, the gain applies to every token served afterward.

### Step 4 — test-implement the top candidate

- **Probe before building.** For anything larger than a config flag, first write a ~30-minute
  harness probe that measures the candidate's *mechanism* in isolation. Honour a negative
  result. And read the probe's whole output, not just its headline: one of ours confirmed its
  narrow question while showing, in the same output, that the mechanism it validated was only
  5% of the target's cost — we built anyway, and the build lost.
- **Change one variable at a time.** When a new result contradicts an earlier one, suspect the
  measurement environment first — thermal state, autotuner churn, cache warmth, prompt-length
  variance — and re-run the control before blaming or crediting the code.
- **Make harnesses match the deployed regime,** or they will lie to you: working sets larger
  than your GPU's last-level cache (rotate buffers), CUDA-graph-captured timing wherever the
  server uses graphs, the real paged KV-cache layout, the real strides. Every
  harness-vs-server discrepancy we chased traced back to violating one of these.

### Step 5 — re-profile, gate, adopt, and loop

Measure the change in-server, same-boot A/B where possible. Adopt behind a flag; make it the
default only after `validate.py` passes and a soak survives. Log the result — especially if
it failed — update the dead-end register, and return to Step 1. Each iteration is cheaper
than the last precisely because the registers accumulate.

### Termination — when to stop

Stop a campaign when either condition holds:

- the dominant component measures at **>85% of its theoretical floor** (our decode campaign
  ended with weight streaming at 91% of achievable bandwidth — at that point a *perfect*
  replacement kernel had a 1.8 ms ceiling, and we stopped on that arithmetic), **or**
- **the candidate list is empty**: every remaining lever is measured dead or priced above its
  ceiling. Write that state down explicitly — the difference between "unexplored" and
  "explored and closed" is most of the value your successor inherits.

## Calibration: what the loop produced here

**Decode campaign:** 18.4 t/s at 42k → profile: attention was 100% of context-dependent cost →
custom decode kernel (bandwidth regime: packed loads + fused dequant, 6× better slope) →
all-reduce (premise half-wrong; +1.9% kept) → compilation +20% → queue depth +3.8% → MTP with
a batched-verification kernel +24% → **41.4 t/s, ≈76% of the absolute weight-streaming floor.**

**Prefill campaign:** decay curve 816→386 t/s → profile: one kernel at 6% of dot peak → the
obvious fix (wider tiles, upstream's own fix for NVIDIA) measured *worse* → the real fix was
one launch parameter (+35% at 37k) → two custom kernel structures built, validated correct,
and rejected at 0.70× and 0.58× → chunk size swept (default was optimal), speculative-decoding
prefill cost measured at nil → **closed at 834/747/521 t/s with the one remaining route priced
and filed, not built.**

Across both campaigns, nine load-bearing premises — several of them ours — were refuted by
measurement, at a typical cost of one probe or one boot each. That is the point of the loop:
it assumes your premises are wrong, and makes each one cheap to kill.
