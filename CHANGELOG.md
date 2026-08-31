# Changelog

**Copyright © 2026 Aron Hsiao. Licensed under the GNU General Public License v3.0 or later.**

Newest first. Companion to [README.md](README.md); per-model detail lives in each
`builds/*/BUILD.md`, and measured configs in [02-VERSIONS.md](02-VERSIONS.md).

**2026-08-31 — the recipe ships as a container image.** New [containers/](containers/README.md):
`Dockerfile.rocm_base` (upstream vLLM's `v0.27.1` base recipe, vendored verbatim with patch 0005
applied) and `Dockerfile` (clone the pinned tag *and* commit, `git apply` patches 0001–0009,
compile every extension for gfx1030, install both plugins, verify code-object targets / patch ops /
patch markers / stock libhsa at build time, OCI labels, healthcheck, `vllm serve` entrypoint), plus
`build.sh` to build, tag and push `ghcr.io/leapdragon/vllm-rdna2-recipe`. All nine patches apply
cleanly to pristine `v0.27.1` (`6e448d0ea9`). The from-scratch base build — the step 02-VERSIONS
said nobody had run end to end — was started the same day; result in containers/README.md "Build
status" and 02-VERSIONS. `config/serve-rdna2-tp2.sh` now derives the `render`/`video` GIDs from the
host instead of hardcoding this machine's 991/44, and passes `HEALTHCHECK_PORT` to the image.
Validation build of the runtime Dockerfile against the old hand-built base: PASS (27.5 GB, ~5.5 min,
GPU smoke test OK — on the RX 6700 XT through the baked `HSA_OVERRIDE_GFX_VERSION`). The first
from-scratch base attempt died 57 min in at flash-attention (`['gfx1030']` unsupported): **patch 0005
grew** — it now also skips the CDNA-only flash-attention/AITER/MORI stages when the arch list has no
`gfx9xx`, which is what the measured base always was (torch/vision/audio/triton/amdsmi, nothing else).
Attempt 2 passed: base 24.2 GB (~65 min on 32 cores), package-for-package identical to the hand-built
one; runtime image 26.9 GB on it, all checks and the GPU smoke test green. The 02-VERSIONS warning that
nobody had rebuilt the base from scratch is retired. Serving test from the published image (27B GPTQ,
TP=2, MTP=2): outputs 8/8 byte-identical to the recorded baseline; decode equal to the newest
hand-built image in a same-day A/B and far ahead of the pre-plugin Aug-18 one — the shortfall vs
BUILD.md's table tracked to the host itself (170 W power caps, TP pair root links at x8 Gen3), not
the image. A pair test on the x16-Gen3-rooted pair (`DEVICES=2,4`) decoded the same within noise —
link width is not the bottleneck — and surfaced a real trap: the torch.compile/AOT cache is
device-set-specific, and reusing one across pairs crashes boot with an aperture violation
(TROUBLESHOOTING 5b; the wrapper now scopes `compile-cache-<devices>` per device set). Follow-up
tests killed the remaining environmental hypotheses one by one: raising the cap 170→220 W changed
nothing (cards draw 208–214 W for the same decode — bandwidth-bound, as documented), and **v3 — the
exact image BUILD.md's numbers came from — measures 27–31 t/s on today's host** with identical
outputs. That platform hypothesis died the same day: **the regression was the lost TunableOp results CSV.**
A torch profile (record_shapes) showed the MTP step paying three ~10.5 ms lm_head GEMMs
(`tn_124160_{1,3}_5120` fp16 at 115 GB/s — rocBLAS's heuristic macro-tile for a skinny shape);
an isolated micro-benchmark reproduced 11.08 ms default vs 2.96 ms TunableOp-tuned. One offline
tuning session later, lookup-only decode from the published container image measures
**49.0/45.3/41.8 t/s at 3.5k/13k/42k — the recorded table, reproduced** (outputs 8/8 identical).
The per-rank CSVs are now shipped in `builds/btbtyler09-…/tunableop/` with restore/retune
instructions in BUILD.md ("load-bearing rows"); the symptom is TROUBLESHOOTING 5c; the stability
cmdline, power caps, link width, HW queues, P2P latency and the checkpoint were all exonerated
by direct A/B along the way. `GPU_MAX_HW_QUEUES=4` is restored in the wrapper (dropped in the
Aug-22 refactor; measured as a no-op — the container default is already 4).

**2026-08-31 (later) — the image became self-configuring.** New entrypoint shim (`recipe-serve`):
`docker run … IMAGE preset:<name>` serves a build's tuned configuration — flags, kernel disable
list, plugin knobs, MTP default, and the shipped TunableOp lm_head rows auto-seeded into `/tuning`
— with no repo clone; `list-presets` enumerates, `DRYRUN=1` prints the resolved command,
`MTP=`/`PORT=` and appended vLLM flags override, plain arguments pass through to `vllm serve`
unchanged, and device choice stays standard `ROCR_VISIBLE_DEVICES`. Presets ship for the three
27B builds; the 122B preset refuses with instructions (it needs one-time weight conversions and
the repo wrapper). `builds/*/preset.env` is the format; verified end to end with a bare
`docker run` (no mounts beyond the HF cache): tuned rows auto-seeded, outputs 8/8 identical,
decode 36.5–48.5 t/s — the tuned band, from one command.

**2026-08-30 (later) — the ~100 t/s was measured with a handshake that never waited.** On
this ROCm, `hipStreamWaitValue32` is accepted during stream capture but not recorded into the
HIP graph, so the n-gram (PLE) offload's in-graph wait was a no-op on every CUDA-graph decode
step: the GPU read whatever lookup was in the buffer. MTP's per-step host round trip hid it by
keeping the CPU worker in lock-step; with MTP off, async scheduling ran the host a step ahead,
the worker dropped "duplicate" requests, and long generations garbled. Moving the wait outside
the graph is not the fix here — a pending WAIT_REG_MEM cannot be preempted by KFD queue
eviction, and the reset lost a card from the PCIe bus. The fork now synchronises on the host
(shared-memory counter, the model thread blocks before enqueueing the forward), moved the
one-shot all-reduce's barrier flags out of a host-coherent page that four GPUs polled over
PCIe (also an ordering race between payload and flag — greedy runs were not reproducible),
and recovered the cost with a fused numpy lookup, a prefaulted sidecar and rank-side result
copies. Honest numbers: **MTP=0 61–65 t/s**, byte-identical greedy runs, soak with no PCIe
events; the MTP=3 98–106 figure awaits a re-measure. Fork `CHANGES.md` §8a, TROUBLESHOOTING §5a.

**2026-08-30 — Qwen3.8-Flash-Next at ~100 t/s, in its own fork.** The 176 B / 512-expert
model with a 51 B-row n-gram table runs on 4× V620 at 98–106 t/s (llama.cpp: 29–30) via
https://github.com/leapdragon/vllm-rdna2-qwen — upstream vLLM main + the unmerged Flash-Next
model branch + 22 ported commits, built from source against TheRock ROCm 7.14 on the host
(PyTorch too: TheRock ships no torch for gfx103X). The campaign (T42–T46) in one line each:
PLE n-gram table served from a 30 GB int4 sidecar by a CPU worker over a HIP stream-wait shim;
CUDA graphs 3×; MTP 1.9×; every dense fp16 projection had been on rocBLAS at 35 % of bandwidth
because vLLM's skinny-GEMV gate is gfx9/gfx11-only and its RDNA kernels miscompute on gfx10 —
own wave-per-row GEMV, 2.1×; the T38 int4 MoE GEMV had never run under expert parallelism;
a push-based one-shot P2P all-reduce (33 µs vs RCCL 156) — but a barrier shows rank skew as
its own time; int8 shadows of the dense weights; ~1,100 launches fused out of a 2,900-kernel
decode step. New traps: torch.compile freezes Python-level decode/prefill branches at trace
time (put the choice inside a custom op); a stale compile cache served an old graph; a
`cuda-bindings` package shadows the HIP shim on ROCm; TheRock's LLVM on `CMAKE_PREFIX_PATH`
breaks Triton's configure. See TROUBLESHOOTING.md §5a.

**2026-08-25 — four cards, TP=4: 56–59 t/s on the 122B (+43–124% over PP=3).** The
config every earlier attempt said was impossible: flat tensor-parallel across four V620s,
MTP=2, near-FLAT decode from 3.8k to 41k (attention sharded 4-way collapses the context
term), cards bandwidth-bound at ~180 W — cool and quiet at full speed. Fresh prefill
830/537 t/s at 3.5k/45k — long-context prefill trails PP=3 pending the TP-shape tuning
pass (see the build's 4-GPU section; and never trust prefix-cached TTFT for prefill claims).
What changed: the platform-personality mitigation stack — kernel line
(`amdgpu.pcie_gen_cap=0x00070007` Gen3 link cap, `aspm=0`, `runpm=0`, `gpu_recovery=1`),
`HSA_NO_SCRATCH_RECLAIM=1`, `NCCL_P2P_LEVEL=PXB`, and `--max-num-batched-tokens 2048`
(batch size is a TIMING knob on graphics silicon: it sets unpreemptible dispatch length,
scratch-crossing odds, DMA burst duration, and power-ramp width all at once). Validated
warm, cold-cache, and soaked: ~2.5 h sustained, zero gpu events. Details and prerequisites
in the 122B build's 4-GPU section; updated TROUBLESHOOTING §4. **Still untuned for TP=4
shapes** (MoE N=256 sweep + lm_head TunableOp rows pending) — these numbers are a floor.
Credit where due: `HSA_NO_SCRATCH_RECLAIM` and the P2P level came from strip-mining
[edwinbrowwn/llama.cpp-rdna2](https://github.com/edwinbrowwn/llama.cpp-rdna2), a parallel
4×V620 effort on the llama.cpp side.

**2026-08-24 — the Polaris day: three ways an unsupported GPU poisons a supported fleet.**
A pre-Vega display card broke ROCm enumeration for every healthy card (all-or-nothing agent
discovery), contaminated device identity through amdsmi (which ignores visibility filters),
and — beyond any userspace fix — its kernel-side KFD node poisoned VMM/graph-capture memory
paths, producing arithmetically impossible "out of memory" errors on 64-byte allocations.
Card removed; every diagnostic and the one-byte vendor-binary doorbell patch documented in
the new **TROUBLESHOOTING.md** — failure modes that lie to you, symptom-first.

**2026-08-23 (later still) — MTP pays: 39.7–41 t/s on the 122B.** Two finishing moves on
top of patch 0009: `fd_rdna2` generalized from the 27B's GQA 6 to any GQA ≤ 16 (the Triton
kernels were already parameterized; the wrapper wasn't), and offline TunableOp rows for the
fp16 lm_head shapes (the ROCm skinny-gemm fast path is CDNA-gated; the default Tensile pick
cost 10.6 ms per verification step). 122B decode with MTP=2: 39.7–41 / 33.4 / 21.9 t/s at
3.5k/13k/40k — vLLM now beats llama.cpp on this model on these cards. Verification harness
for the plugin geometries in `verify/fd_gqa_test.py`.

**2026-08-23 (later) — patch 0009: MTP under pipeline parallelism (V2 runner).** Backport
of upstream PR #46994 plus our own V1-runner guards. The finding that matters: MTP under PP
requires `VLLM_USE_V2_MODEL_RUNNER=1` (the V1 drafter path page-faults under PP — upstream
never tested it), and the V2 runner measures identical to V1 on gfx1030 at MTP=0, so the
switch is free. On the 122B, speculation is *correct* (81% acceptance at K=2, draft head
RTN-packed to int4 by `quantize_mtp.py`) but currently a decode regression — verification
attention (q_len=3) rides the context-proportional prefill path; production stays MTP=0
until a batched-MQ kernel covers those shapes. Launcher gains `PP_PARTITION`, `ASYNC_SCHED`,
`SPEC_EAGER`, `EXTRA_ENV`.

**2026-08-23 (night) — patch 0008: the MoE decode skinny GEMV.** The kernel the 122B was
waiting for: wave-per-row expert GEMV at 432 GB/s effective (the tile-based MoE kernels
stream weights across the wrong axis and sit 10×+ off bandwidth at batch-1). Decode on the
Intel 122B: 15.6 → **26.9 t/s** — short-context parity with llama.cpp on the same three
cards, and the day's cumulative on this model is 7.1 → 26.9. The launcher also gains
`EXTRA_MOUNT` (comma-separated bind overlays) — born as a debugging tool for the
twin-apply-method pitfall documented in 01-PATCHES, kept because rebuild-free iteration on
any baked file is generally useful.


**2026-08-23 (evening) — 122B build replaced: backbone-quantized checkpoint doubles decode.**
`builds/Intel-Qwen3.5-122B-A10B-int4-AutoRound/` supersedes the official-GPTQ build. The
lesson that matters: the official Qwen GPTQ quantizes only the *experts*, leaving ~18 GB of
bf16 backbone streaming every token (extra-painful on gfx1030, which has no native bf16).
Intel's AutoRound covers the backbone (shared expert fp16, MTP untouched) — verified from
config + tensor-index metadata alone, before downloading a byte. Decode 7.1 → **15.7 t/s**,
same serving config. Read a MoE checkpoint's exclusion map before trusting its name.


**2026-08-23 (later) — patch 0007: the CUDA moe_wna16 MoE kernel ported to gfx1030.**
An afternoon-scale port (portable lop3/prmt/bf16 substitutions, CAS fp16 atomicAdd, four
un-gating layers). Correct in-server; ~1.1× the Triton MoE path at decode sizes. Its main
value was diagnostic: on the official 122B quant, decode splits roughly evenly between MoE
kernels, the checkpoint's unquantized bf16 backbone, and PP overhead — so this patch alone
does not move end-to-end numbers, and the 122B build's BUILD.md now carries the corrected
token budget. Lesson shipped with it: benchmark MoE kernels at the batch size the server
actually runs (M = tokens × top-k at decode is still tiny), not a convenient synthetic M.


**2026-08-23 — first model beyond two cards: Qwen3.5-122B-A10B at PP=3.**
`builds/Qwen-Qwen3.5-122B-A10B-GPTQ-Int4/` — the official GPTQ 122B MoE across three V620s
via pipeline parallelism (TP=3 is arithmetically impossible: 2 KV heads). New launcher
knobs: `PP` (pipeline stages) and `MOE_CFG` (mounts a tuned fused-MoE config JSON; the
build dir carries one swept offline on gfx1030, worth +75% prefill). Working and validated:
prefill 558–721 t/s; decode 7.1 t/s, bound by vLLM's Triton fused-MoE overhead at batch-1
(~45× off bandwidth — the CUDA moe_wna16 kernel that handles this regime is is_cuda()-gated
out of ROCm builds; porting it is the open campaign). Read the BUILD.md's quantization
section before extrapolating: only the experts are int4, the bf16 backbone caps decode
at ~28 t/s.


**2026-08-22 (later) — third build: AutoRound mixed-precision.**
`builds/Pilcothink-Qwen3.8-27B-MixedInt4-AutoRound/` — W4 g32 symmetric with int8 on 17
sensitivity-selected projections. Needs two scripted one-time checkpoint conversions (in the
build dir): vLLM has no "auto-round" quant method, so the config is rewritten to GPTQ with
the mixed-bits table as `dynamic` overrides, and the int4 MTP head is dequantized to dense
(vLLM builds MTP predictors unquantized). Decode 47.0/49.8/41.6 t/s, prefill 856/827/670 —
reference parity or better. A worked example of adapting a checkpoint whose *metadata*, not
tensors, is the incompatibility.


**2026-08-22 — per-model builds/ tree; second model (AWQ) at full speed; patch 0006.**
- The repo now mirrors our working layout: `builds/<model-id>/` holds a ready-made optimized
  configuration per model (`BUILD.md` + `serve.sh`); plugins moved to `builds/shared/plugins/`.
- **`cyankiwi/Qwen3.8-27B-AWQ-INT4`** (compressed-tensors, *asymmetric* W4 g32 — same bytes as
  the GPTQ reference) is brought up at parity or better: decode 49.8/49.0/43.9 t/s at
  3.8k/15k/41k, prefill 753/736/524 at 3.8k/7.5k/35k. Asymmetric uint4 is outside Exllama's
  types, which forces a different kernel route entirely — new **patch 0006** extends vLLM's HIP
  skinny int4 GEMV from its gfx11/gfx12 guard down to gfx1030 (wave32/LDS/v_dot2 envelope is
  identical) and adds a dequant-to-dense + rocBLAS prefill route. Two transferable lessons in
  01-PATCHES: check arch guards before assuming a "gfx11+" kernel is off-limits, and a fused-
  dequant Triton GEMM loses ~3× to dequantize-then-rocBLAS at prefill shapes.
- The shared launcher gained a `DISABLED_KERNELS` override so per-model builds can pick their
  weights kernel.

**2026-08-21 (later) — prefill +35% at long context.** Patch 0002 now also sets
`num_stages=1` for the gfx10x attention branch: pipelining was halving occupancy at
head_dim 256. Prefill 834/747/521 tok/s at 3.5k/15k/37k (was 816/608/386), outputs
byte-identical. The wider-query-tile route (upstream's Blackwell fix) was measured and is
*worse* on this chip — see 01-PATCHES.

**2026-08-21 — MTP enabled; prefill stall fixed.**
- `MTP=2` (the checkpoint's own multi-token-prediction head, `qwen3_5_mtp`) is now the default:
  **41.4 t/s @41k (+24%), 50.4 @14k (+36%)**, output-lossless. This required extending the
  `fd_rdna2` plugin to **batched multi-query verification** — without it, verification passes
  silently fall to the stock attention kernel and MTP measures as a **3.6× regression**. The
  plugin and MTP must be deployed together.
- **TunableOp now runs lookup-only** (`PYTORCH_TUNABLEOP_TUNING=0`). Tuning mode autotunes every
  never-seen GEMM shape mid-request, and prefill M is prompt-length-dependent — fresh prompts
  paid minutes-long stalls (bimodal 771 vs 37 tok/s at identical sizes). See pitfalls in 01.
- `verify/decode-rate.py` and `verify/prefill-rate.py` added; `longctx-decode.py` removed (its
  median-gap method mis-measures under speculative decoding).

**2026-08-21 (later still) — prefill campaign closed at the practical ceiling.** Final:
834/747/521 tok/s at 3.5k/15k/37k. Verified optimal and now documented as measured dead ends
(see 01-PATCHES): two custom prefill-attention kernel structures (0.70× and 0.58× — at prefill,
KV is cache-served and the stock kernel wins), chunk sizes other than 8192, and wider query
tiles. The reusable rule that came out of it: **pass `num_stages=1` in every Triton kernel on
this chip.** Full patch series re-verified: applied to pristine vLLM 0.27.1 it reproduces the
running tree exactly.

**2026-08-20 — initial publication.** Five patches, two plugins, 33.5 t/s @42k, no speculation.
