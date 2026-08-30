# PROFILE-NAVI21.md — Navi 21 / gfx1030 (Radeon PRO V620): architectural & performance profile for inference kernel work

> ## ⭐ SINGLE SOURCE OF TRUTH FOR HARDWARE FACTS
> **Every number describing what this GPU is or can do lives here and only here** — bandwidths,
> latencies, cache sizes, instruction rates, ISA capabilities, occupancy limits, interconnect
> behaviour. No other document in this project may restate a hardware figure; they link here.
>
> **Maintenance rule.** If a hardware measurement changes, it changes *in this file*, and nothing
> else needs touching. If you find a hardware number quoted elsewhere, that is a bug — replace it
> with a pointer to the relevant section here.
>
> **Scope boundary.** Application-level results (tokens/sec, cache hit rates, model configs, which
> vLLM patch does what) belong in `STATE.md` and `TESTS_RESULTS.md`, *not* here. This file is about
> the silicon. Where §11 cites application numbers, it does so only to express them as a percentage
> of a hardware ceiling, and marks them as such.

**Status:** living document. Compiled 2026-08-19. **All 37 profiling tests complete; no open TBDs.**
**Purpose:** a complete, on-the-ground spec of what this silicon can actually do, so that inference kernels (attention decode, GEMV/dequant, collectives) can be written *to* the hardware rather than ported *around* it — on any OSS inference stack (vLLM, llama.cpp, or our own).
**Provenance tags:** every claim is marked
`[published]` (vendor/press/third-party docs) · `[probed]` (verified on this machine) · `[derived]` (arithmetic from other facts) · `[TBD → T-nn]` (unknown; filled by test T-nn in `PROFILE-NAVI21-TESTS-NEEDED.md`).
**Companions:** `PROFILE-NAVI21-TESTS-NEEDED.md` (what we still need to measure), `PROFILE-NAVI21-TESTS-PLAN.md` (how).

---

## 1. Device identity

| Item | Value | Src |
|---|---|---|
| GPU | AMD Radeon PRO V620 | [probed] |
| Silicon | Navi 21, RDNA2, 7 nm, 26.8 B transistors, 520 mm² | [published] |
| LLVM/ROCm target | `gfx1030` | [probed] |
| VRAM | 32 GB GDDR6 (31.98 GiB visible to HIP) | [probed] |
| Board power | 300 W TBP spec; **our cards report a 232 W cap** and sustain 186–197 W under decode load | [published]/[probed] |
| Host interface | PCIe 4.0 x16 on the card; **negotiates Gen4 x16 to onboard switches, but the Threadripper 2950X root ports are Gen3** (x16 for devices 1/3, x8 for devices 0/2) — platform-limited, not card-limited | [probed] |
| Fleet on this machine | 3× V620 installed (devices 0/1/3, gfx1030; a fourth on hand, not yet installed) + 1× RX 6700 XT (device 2, Navi 22/gfx1031, 12 GB, display; runs gfx1030 binaries via `HSA_OVERRIDE_GFX_VERSION=10.3.0`) | [probed] |
| Availability for profiling | **All local inference is switched off for the duration of this profiling program** (user decision, 2026-08-19; server stopped and verified 15:0x — 0 KFD client processes, all V620s at 0% VRAM). Every device, including the 1+3 pair, is available. Multi-GPU tests (T-N1/T-N2) no longer need scheduling around a live server. | — |

## 2. Compute organization

| Item | Value | Src |
|---|---|---|
| Compute Units | **72 CUs = 36 WGPs** (HIP reports "36 multiprocessors" = WGPs; 2 CUs per WGP) | [published]+[probed] |
| Stream processors | 4608 (64 per CU; 2× SIMD32 per CU, 4× SIMD32 per WGP) | [published] |
| Wavefront | **wave32 native** (`warpSize=32`); wave64 executed as two wave32 halves. **Measured [T-C2]: no difference on memory-bound work (505.6 vs 505.5 GB/s), but wave64 is ~36% FASTER on an ALU-bound loop** (12.65 → 17.20 TFLOPS) — one instruction stream drives 64 lanes, halving per-lane loop/branch overhead. *Caveat: measured on a loop with real overhead; the gain may shrink in a tightly-unrolled kernel. Worth trying `-mwavefrontsize64` on any compute-heavy inner loop.* Note `hipDeviceProp.warpSize` reports 32 regardless of the compile flag. | [probed] |
| Max threads/block | 1024 | [probed] |
| Max threads per WGP | 2048 → **64 wave32 per WGP → 16 waves per SIMD32** | [probed]/[derived] |
| Clocks | max boost 2570 MHz (HIP `clockRate`); spec game clock ~2200 MHz; **observed sustained 2400 MHz at 186–197 W during decode** | [probed] |
| Memory clock | 1000 MHz controller = GDDR6 16 Gbps effective | [probed]/[published] |

**Throughput — MEASURED [probed, T-C1]**, 8 independent register-resident accumulator chains, normalised to SCLK sampled across the timed region:

| Instruction | ops/instr | issue/clk/SIMD | **measured** | theoretical @2.4 GHz | of peak |
|---|---:|---:|---:|---:|---:|
| `v_fma_f32` | 2 | 0.754 | **16.7 TFLOPS** | 22.1 | 75% |
| `v_pk_fma_f16` | 4 | 0.767 | **33.9 TFLOPS** | 44.2 | 77% |
| `v_dot2_f32_f16` | 4 | **0.860** | **38.0 TFLOPS** | 44.2 | 86% |
| `v_dot4_i32_i8` | 8 | 0.726 | **64.2 TOPS** | 88.5 | 73% |
| `v_dot8_i32_i4` | 16 | 0.757 | **133.7 TOPS** | 177 | 76% |

**The dot instructions are full-rate.** Every one issues at essentially the same ~0.73–0.86 instructions per clock per SIMD as a plain `v_fma_f32` — there is no penalty for using them, so their higher ops-per-instruction translates directly into throughput. `v_dot2_f32_f16` is actually the *best* issuer of the five. The uniform ~75% shortfall against theoretical is loop overhead in the microbenchmark, not an instruction-specific cost; the derived table this replaces was correct in shape and ~1.33× optimistic in absolute terms.

**For quantised inference this is the headline compute fact:** int8 dot delivers **3.8× the op rate of fp32** and int4 dot **8×**, at no issue-rate penalty.

**And the real GPTQ inner loop confirms arithmetic is not the constraint [probed, T-C4].** The full W4 chain — extract 8 nibbles from a packed `uint32`, subtract the zero-point, apply the g32 scale, FMA — sustains **1907 G weights/s = 953.6 GB/s of 4-bit weight throughput, 1.88× the 506 GB/s memory ceiling.** Shift+mask and `__builtin_amdgcn_ubfe` compile to the same speed (0.63 ms both), so there is no reason to hand-write the extract.

**Design fork resolved:** dequantise-to-float and FMA is affordable — there is no need to stay in integer domain with requantised activations to keep up. A W4 GEMV on this chip should be fully memory-bound with ~1.9× compute headroom. Conversion costs for reference: int8→fp16 is **0.53×** the cost of a plain fp32 FMA (cheaper than the baseline), int8→fp32 is 1.65×.

## 2a. Compute — synthesis and kernel design rules

*(C-series complete. Companion to §4a.)*

| What | Measured | Rule |
|---|---|---|
| fp32 FMA | 16.7 TFLOPS | the baseline |
| packed fp16 FMA | 33.9 TFLOPS | 2× fp32, free |
| **`v_dot2_f32_f16`** | **38.0 TFLOPS** | best issuer of all five (0.86/clk/SIMD) |
| **`v_dot4_i32_i8`** | **64.2 TOPS** | 3.8× fp32 op rate, no penalty |
| `v_dot8_i32_i4` | 133.7 TOPS | 8× fp32 |
| **GPTQ W4 dequant chain** | **953.6 GB/s of 4-bit weights** | **1.88× the memory ceiling — dequant is free** |
| `__expf` | ¼ rate (0.19/clk/SIMD) | ~4 µs/token of softmax — irrelevant |
| int8→fp16 convert | 0.53× an fp32 FMA | cheaper than the baseline |
| bf16 | no native opcodes, no dot | **avoid; cast to fp16** |
| wave64 vs wave32 | +36% on ALU-bound loops, 0% on memory | try `-mwavefrontsize64` on compute-heavy inner loops |
| `__shfl_xor` wave32 reduction | **13.0 ns**, 2.01× faster than LDS | do softmax max/sum in registers |
| global atomics contended | **1100× penalty** | never for split-K |
| LDS atomics contended | 1.2× penalty | contend here instead |
| split-K combine | partials+combine **25.6× faster** than atomics | write partials, second kernel |
| kernel launch | 3.11 µs dependent, graphs only 1.09× | ⚠️ **Corrected 2026-08-20 [T25].** The old note read "~500 launches/token = 3% of a step; don't fuse to save launches". An in-server profile counts **2,553 tiny elementwise/reduce kernels per decode step** averaging **1.94 µs of work each** — 5× the assumed launch count — and **~26% of the token is GPU idle** between kernels. Raising `GPU_MAX_HW_QUEUES` 2→4 recovered 1.3 ms/token (3.8%) on its own. **Kernel count is a first-order cost here; fusion is worth pursuing.** (8 queues regresses 32% — 4 is the optimum.) |
| cooperative grid barrier | 0.43–0.53 µs | persistent single-device kernels viable |

**The one-line conclusion: arithmetic is never the constraint on this chip for quantised inference.** Every quantised path — int8 KV, W4 weights, fp16 accumulate, softmax — has multiples of headroom over the 506 GB/s memory ceiling. Design purely for memory behaviour (§4a) and spend compute freely to get it.

## 3. ISA capabilities [probed via hipcc compile tests on this machine, 2026-08-19]

| Instruction | Meaning | gfx1030 |
|---|---|---|
| `v_dot4_i32_i8` (`__builtin_amdgcn_sdot4`) | 4×int8 dot + i32 accum | ✅ **available** |
| `v_dot4_u32_u8` (`udot4`) | unsigned variant | ✅ |
| `v_dot8_i32_i4` (`sdot8`) | 8×int4 dot | ✅ |
| `v_dot2_i32_i16` (`sdot2`) | 2×int16 dot | ✅ |
| `v_dot2_f32_f16` (`fdot2`) | 2×fp16 dot + f32 accum | ✅ |
| WMMA (`wmma_f32_16x16x16_f16`) | RDNA3 matrix | ❌ compile fails |
| MFMA | CDNA matrix | ❌ not present |
| Packed fp16 (`v_pk_fma_f16` etc.) | 2×fp16/instr — **measured 33.9 TFLOPS, full issue rate** [probed, T-C1] | |
| **Triton reaches BOTH dot families** | fp16 `tl.dot` → `v_dot2c_f32_f16` with **zero** FMA fallback at **every** tested shape (16×16×16 → 16 dots; 64×64×64 → 1024 dots) [probed, T-I1]. **int8 `tl.dot` → `v_dot4_i32_i8`** (16 instrs, 0 FMA) [probed, T-I2]. **Triton is fully viable as the implementation language.** |
| **But `tl.sum(A*B)` does NOT reach them** | the hand-rolled vector/GEMV form emits no dot instructions [probed, T-I2b]. If you want the dot units in Triton you must use `tl.dot`, which has a 16-row minimum — so a literal M=1 decode GEMV would pad to M=16. Given W4 dequant already has 1.88× compute headroom and attention is memory-bound, that padding is likely affordable, but it is the key structural constraint on a Triton implementation. |
| Cross-lane reductions | **`__shfl_xor` ladder: 13.0 ns per full wave32 sum, 13.4 ns for max — 2.01× faster than an LDS round-trip (26.1 ns)** [probed, T-C8]. **Do softmax running max/sum in registers via `__shfl_xor`, not through LDS.** |
| bf16 | **Arithmetic compiles** (`hip_bfloat16`, native `__bf16`, packed `__bf16` vectors) **but no native bf16 opcodes appear in the generated ISA** [probed, T-C6] — treat it as emulated. **The bf16 dot builtin (`__builtin_amdgcn_fdot2_f32_bf16`) is NOT available.** Reinforces this project's standing guidance: use fp16, cast bf16 checkpoints. | |
| fp8 anything | | ❌ none. Software emulation only; measured Triton conversion throughput: e4m3 199 Gconv/s, e5m2 921, **int8+scale 1581** (int8 8× cheaper than e4m3) [probed] |

**Design consequence:** there is no matrix unit of any kind, so a GEMM-shaped formulation buys no hardware advantage — but it does pay the LDS-staging and tiling costs. The hardware's strength for quantized inference is the **dot-product instruction family, which matches int8 (q8_0-style) KV and int4/int8 weights exactly**.
**Correction (2026-08-19):** an earlier working assumption — that Triton could not reach these units and therefore the performance kernel *must* be HIP — is **wrong for fp16**: `tl.dot` emits `v_dot2c_f32_f16` cleanly. The language choice is now an open question decided by T-I2 (int8 lowering) and T-I3, not a foregone conclusion.

## 4. Memory hierarchy

| Level | Size | Latency | Bandwidth | Src |
|---|---|---|---|---|
| L0 vector D$ | 16 KiB per CU | **20.3 ns** [probed, T-M3] | — | [published] |
| Scalar K$ / I$ | ~16 KiB / 32 KiB per WGP | — | — | [published, low confidence] |
| GL1 | 128 KiB per shader array | **39.6 ns** [probed, T-M3] | — | [published] |
| L2 | **4 MiB** (`l2CacheSize`) | **116.6 ns** [probed, T-M3] | **~9340 GB/s** at a 1 MB working set [probed, T-M2] | |
| **Infinity Cache (L3)** | **128 MB** — cliff measured between 120 and 160 MB, exactly as advertised | **150.1 ns (+33.5 ns over L2; DRAM is a further +30.2)** [probed, T-M3] | **~1890 GB/s** flat across 8–120 MB working sets = **3.72× DRAM** [probed, T-M2] (published peak ~1987 — we reach 95% of it) | |
| GDDR6 | 32 GB | **180.3 ns** [probed, T-M3] | **512 GB/s theoretical / 506 measured** | |
| **Achievable DRAM BW** | — | — | **read 506 GB/s = 98.8% of the 512 theoretical** · dwordx2 506 · dword 495 · write 447 · copy 415 [probed, T-M1 pass 1, 2 GiB buffers, mean SCLK 2450] | |
| Host↔device (PCIe Gen3 root) | — | — | **7.08–7.16 GB/s** both directions on device 0 [probed, T-M7]. Device 0 sits on an **x8** root port (7.88 GB/s ceiling) so this is **~90% of its link**, not 45% of x16 — the x16-rooted devices 1/3 should reach ~14 GB/s. **Pinned memory gives no advantage over pageable** (0.99×), unusually. | |

**This card streams at essentially spec bandwidth.** 506 GB/s read is 98.8% of the 512 GB/s theoretical — there is no meaningful "achievable vs theoretical" discount to hide behind, and reads are width-insensitive (dwordx4 ≈ dwordx2 > dword by only 2%). **Use 506 GB/s as the denominator for every efficiency claim.** Earlier claims in this project used ~244 GB/s (which was an *achieved* decode rate mistaken for a ceiling) and are corrected in §11.

**Infinity Cache is the sleeper.** 128 MB at ~4× DRAM bandwidth and lower latency. Per-layer KV working set at 43k ctx (int8, per GPU at TP=2) ≈ 44 MB — **fits**. **The cache ladder, measured end to end [T-M2 phase 1]:** 1 MB → 9340 GB/s (L2) · 8–120 MB → ~1890 GB/s (Infinity Cache, flat) · 160 MB → 575 · 256 MB → 523 · 1024 MB → 508 GB/s (DRAM). The cliff sits precisely between 120 and 160 MB. IC is worth **3.72×** over DRAM when your data fits.
*(An earlier single-pass measurement in T-M4 reported 1256 GB/s for a 44 MB buffer; that included compulsory cold misses. The warm steady-state figure of ~1890 GB/s from T-M2 supersedes it.)*

**But the obvious lever is dead: `__builtin_nontemporal_load` is a NO-OP on gfx1030** [probed, T-M4] — 507.3 vs 507.4 GB/s streaming, 1256.4 vs 1257.8 GB/s resident. It neither costs anything nor bypasses any cache. The "mark weights nontemporal so they stop evicting KV" strategy cannot be implemented this way. **Consequence: T-M2 phase 2 must rest on its phase-3 contention control** (the reviewer's addition — now load-bearing rather than belt-and-braces).

**ANSWERED [T-M2 phases 2–3], and the answer is no.** A 44 MB KV-sized buffer reads at 1873 GB/s alone, but only **145 GB/s (7.8% retained)** while a weights-sized 8 GiB stream runs concurrently. The >IC control (512 MB) retained 22.4% under the same load, so **14.7 percentage points of the loss is specifically Infinity-Cache eviction**, not raw bandwidth contention.

**Design consequence — a tempting direction is closed.** In real decode, a given layer's KV is read once per token, and the entire ~16.7 GB of model weights streams past between consecutive reads of it. IC cannot hold KV across that, and `__builtin_nontemporal_load` cannot protect it (T-M4: it is a no-op). **The attention kernel's realistic ceiling is DRAM bandwidth, ~506 GB/s — not the 1890 GB/s IC figure.** Plan against 506. The IC prize is real but unreachable for this access pattern without hardware cache partitioning, which RDNA2 does not expose.

**Fetch granularity ≈ 128 B [probed, T-M6].** Reading one 4 B float every S bytes: useful bandwidth falls in exact proportion from S=4 (475 GB/s) through S=128 (15.79), then **S=256 gives 15.62 — essentially unchanged**, which is the plateau signature of a 128 B fetch unit. *Caveat, recorded rather than smoothed over: S=512 halves again (7.82 GB/s, implying ~256 B), which a simple 128 B line model does not predict; likely a channel/TLB second-order effect. The design-relevant number is 128 B.*
**Rule: lay KV out so each wave consumes ≥128 contiguous bytes per access group.** Anything sparser wastes bandwidth in direct proportion (see stride data below).

**Access-pattern costs [probed, T-M5]:**
- **Stride penalty is exactly proportional.** Useful bandwidth = 505 / stride: 16 B apart → 505 GB/s, 32 B → 253, 64 B → 126, 128 B → 63. Penalties of 2.00× / 4.00× / 8.00× to three figures. A wave's *access span* sets the traffic, so partially-used lines waste bandwidth in direct proportion — pack KV so a wave consumes contiguous bytes.
- **PagedAttention-style block indirection is FREE.** Walking a 1 GiB pool through a shuffled block table retains **99.8–100%** of sequential bandwidth at every block size tested — even at **4 KB** blocks. vLLM's real blocks (~390–400 KB contiguous per layer/head at block_size 784–1552) are two orders of magnitude above where scatter would start to matter. **The block table is not a performance problem and needs no optimisation.** (Generalises and quantifies the negative result from the attention-harness scatter test on 2026-08-19.)


## 4a. Memory subsystem — synthesis and kernel design rules

*(M-series complete: 8/8 measured on this machine, 2026-08-19. This section is the distillation — if you are writing a kernel for this GPU, read this and you should not need to re-derive anything below. A companion compute section will appear here when the C-series completes.)*

### The full ladder, one table

| Level | Size | Latency | Bandwidth | Note |
|---|---|---:|---:|---|
| L0 vector D$ | 16 KiB / CU | 20.3 ns | — | |
| GL1 | 128 KiB / shader array | 39.6 ns | — | |
| L2 | 4 MiB | 116.6 ns | ~9340 GB/s | measured at a 1 MB working set |
| Infinity Cache | 128 MB | 150.1 ns | **~1890 GB/s** | flat 8–120 MB; cliff between 120 and 160 MB |
| DRAM | 32 GB | 180.3 ns | **506 GB/s** | 98.8% of the 512 theoretical |

Write 447 GB/s, copy 415 GB/s. Fetch granularity **128 B**. Host↔device 7.1 GB/s on an x8 root port (~14 expected on x16).

### The five rules that follow

1. **Budget against 506 GB/s, never 1890.** Infinity Cache is real and worth 3.72×, but it cannot hold KV during decode: a 44 MB KV-sized buffer keeps only **7.8%** of its standalone bandwidth while a weights-sized stream runs, against 22.4% for a >IC control — so ~15 points of that loss is eviction specifically. Structurally, a layer's KV is read once per token with all ~16.7 GB of weights streaming past before the next read. There is no access pattern that fixes this and no cache-partitioning control on RDNA2.
2. **Keep ≥2 independent loads in flight per lane.** Little's Law on the measured numbers: `506 GB/s × 180 ns ≈ 91 KB` must be in flight device-wide. A wave32 with one 16 B load per lane holds 512 B, so ~178 concurrent wave-loads — about **1.2 outstanding loads per SIMD32** (144 SIMDs). Streaming kernels meet this trivially; a kernel whose lanes sit on a dependent chain or a synchronised LDS tile does not, and can only compensate by multiplying workgroups. **This is the single most important design variable, and it is where the current vLLM attention kernel fails.**
3. **Give each wave ≥128 contiguous bytes per access group.** Stride penalty is exactly proportional — 32/64/128 B between consumed values costs 2.00×/4.00×/8.00×. A wave's *span* determines traffic, so partially-used lines waste bandwidth one-for-one. With int8 KV and head_dim 256, one head-token is exactly 256 B, which is comfortable **provided lanes cover it contiguously rather than striding across heads**.
4. **Do not tune workgroup count or size for bandwidth.** A well-formed streaming kernel reaches 96.5% of peak at **36 workgroups (1/WGP)** and 99.8% at **72 (2/WGP)**, flat to 1024. Block size 64→1024 threads moves it 0.22%. If a kernel needs many workgroups to go fast, rule 2 is being violated — fix the per-lane parallelism instead.
5. **64 KiB LDS per workgroup is a hard cap.** 72 KiB fails outright (`hipErrorInvalidValue` at both attribute-set and launch); the 128 KiB per WGP is not addressable by one workgroup. Any design needing more must restructure, not ask for more.

### Ruled out — do not spend time re-testing these

| Idea | Verdict |
|---|---|
| Mark weights nontemporal so they stop evicting KV | **Dead.** `__builtin_nontemporal_load` is a **no-op** on gfx1030 — 507.3 vs 507.4 GB/s, no bypass, no cost. |
| Keep KV resident in Infinity Cache | **Dead.** See rule 1; the weight stream evicts it and nothing can prevent that. |
| Optimise the PagedAttention block table | **Pointless.** Scattered block walks retain **99.8–100%** of sequential bandwidth down to 4 KB blocks; real blocks are ~390–400 KB. |
| Route the block table through the scalar unit | **Already automatic and irrelevant.** The compiler emits `s_load` for wave-uniform indices (0 vector loads), but timing is identical (0.4%) — the table is ~512 B against ~32 MB of data. |
| Widen the prefill tile by requesting more LDS | **Impossible.** See rule 5. |
| Use pinned host memory for transfers | **No benefit measured** (0.99× vs pageable). |

### Still open in memory

Nothing blocking. The remaining unknowns are compute-side (C-series), occupancy-vs-register (T-O1), launch overhead (T-O4/O5), Triton lowering (T-I2/I3), and the kernel-path interconnect (T-N2).

## 5. LDS (shared memory)

| Item | Value | Src |
|---|---|---|
| Physical | **128 KiB per WGP** | [published] |
| Modes | WGP mode (unified 128 KiB) vs CU mode (2×64 KiB) [published]. **`-mcumode` changes nothing measurable** [probed, T-L4]: memory 505.5 vs 505.5 GB/s, ALU 12.66 vs 12.66 TFLOPS, occupancy 16 waves/SIMD and LDS allocation identical in both. **Not a lever; leave it at the default.** |
| **Runtime reports** | `sharedMemPerBlock = 65536`, `maxSharedMemoryPerMultiProcessor = 65536` | [probed] |
| Triton enforces | 65536 B/workgroup (`OutOfResources: 139264 > 65536` killed head_dim-256 attention until our tile patch) | [probed] |
| Can a workgroup get >64 KiB? | **No — 64 KiB is a hard per-workgroup cap** [probed, T-L1]. Verified by sweep: 32/40/48/56/64 KiB all launch and round-trip a checked pattern; **72 KiB fails** `hipErrorInvalidValue` at both `hipFuncSetAttribute(MaxDynamicSharedMemorySize)` and launch. The 128 KiB per WGP is not addressable by a single workgroup. **Consequence: `gfx1030-lds-tile.patch` was NOT over-conservative — the prefill tile cannot be widened by asking for more LDS.** Any prefill recovery must come from using the 64 KiB better (or not staging through LDS at all). |
| Banks | 32 banks × 4 B. **Conflict penalty is ~linear in conflict degree** [probed, T-L2]: 2-way 1.41×, 4-way 2.60×, 8-way 5.09×, 16-way 10.1×, 32-way 20.0×. Pad LDS arrays to avoid power-of-two strides. |
| **LDS atomics** | **31.7 G/s uncontended, 26.4 G/s all-to-one — only a 1.2× penalty** [probed, T-L3]. Contrast with *global* atomics, which collapse 1100× under the same contention (T-C7). **If a reduction must contend, contend in LDS.** |
| Hazard | LLVM flags `lds-misaligned-bug` for gfx10 WGP mode. **Not reproduced** [probed, T-L5]: `float4` LDS accesses at byte offsets 0/4/8 all completed with zero errors and no fault, so the toolchain appears to guard it. *Absence of reproduction is not proof of safety — keep LDS accesses naturally aligned where convenient, but this is not a live hazard for ordinary code.* |

### 5a. Latency and the bandwidth-delay product — why the attention kernel stalls

Measured ladder [T-M3, dependent pointer chase, one lane, `clock64` calibrated at 2477 MHz vs SCLK 2460 — the counter does track SCLK on this chip, so the plan's fixed-reference caveat does not apply]:

| level | latency |
|---|---:|
| L0 (8 KB) | 20.3 ns |
| GL1 (64 KB) | 39.6 ns |
| L2 (2 MB) | 116.6 ns |
| Infinity Cache (64 MB) | 150.1 ns |
| DRAM (512 MB) | 180.3 ns |

**The number that explains the attention kernel.** Little's Law: to hold 506 GB/s at 180 ns of latency you need
`506 GB/s × 180 ns ≈ 91 KB of data in flight` at all times.
A wave32 issuing one 16 B load per lane has 512 B outstanding, so **~178 concurrent wave-loads** are required across the device — about **1.2 outstanding loads per SIMD32** (144 SIMDs = 36 WGP × 4).

That is trivially met by a streaming kernel whose lanes each have several independent loads in flight — hence T-O2's result that 2 workgroups/WGP saturates. It is *not* met by a kernel whose lanes sit on a dependent chain or a synchronised LDS tile, where each lane has **one** outstanding load and then waits 180 ns. Such a kernel can only buy memory-level parallelism by multiplying workgroups — which is exactly what raising the softmax segment count did, and why it still only reaches 17% of bandwidth.

**Measured directly [T-O1] — the quantity that matters is `waves/SIMD × in-flight loads per lane`, and either dimension can supply it:**

| occupancy | 1 load/lane | 2 | 4 | 8 |
|---|---:|---:|---:|---:|
| 0.5 waves/SIMD | **136 GB/s** | 254 | **433** | 140 ← collapses (not spilling — see below) |
| 2 waves/SIMD | 486 | 503 | 506 | 490 |
| 4 waves/SIMD | 503 | 506 | 506 | 503 |
| 12 waves/SIMD | 506 | 506 | 506 | 508 |

At normal occupancy (≥2 waves/SIMD) a single load per lane already saturates — parallelism is abundant and nothing needs tuning. **The interesting corner is low occupancy**, which is exactly where a register-hungry kernel lives: at 0.5 waves/SIMD, one load per lane reaches only **27% of peak**, but **four independent loads per lane reach 86%**.

**Design rule for a register-accumulator attention kernel:** it deliberately trades occupancy for registers, so it *must* carry **≥4 independent in-flight loads per lane** to compensate. But more is not better — 8 accumulators collapses to 140 GB/s, worse than 1. **The usable band is 2–4 in-flight loads per lane.**

*Mechanism note, stated honestly:* the collapse at 8 is **not** register spilling. The compiler reports VGPRs 17/25/41/45 for NACC 1/2/4/8, **ScratchSize 0 and occupancy 16 waves/SIMD in every case** — nothing spilled and occupancy never dropped. The likely cause is too many widely-separated concurrent streams per lane (at this grid the 8 loads are ~36 KB apart, so ~295 KB of span per lane per iteration) thrashing address translation or channel locality. Not established; recorded as an observed cliff to design around rather than an explained one. Register pressure is evidently *not* the binding constraint on this chip at these accumulator counts — 45 VGPRs still permits full 16-wave occupancy.

## 6. Registers & occupancy

| Item | Value | Src |
|---|---|---|
| VGPR file | 128 KiB per SIMD32 (1024 regs × 32 lanes × 4 B); HIP `regsPerBlock=131072` per WGP | [published]+[probed] |
| Max VGPRs per wave | 256 (wave32) | [published] |
| Waves per SIMD32 | **16 max** — confirmed by the compiler's own occupancy report on a 4-VGPR kernel (T-I5) | [probed] |
| Occupancy vs VGPR use | 1024/VGPRs-per-wave waves; e.g. 64 VGPRs → 16 waves, 128 → 8, 256 → 4 | [derived — verify + latency-hiding curve → T-O1] |
| Occupancy needed to saturate DRAM | **Almost none, for a well-formed streaming kernel** [probed, T-O2]: 36 WGs (1/WGP) already reaches 96.5% of peak; **72 WGs (2/WGP) reaches 99.8%**; flat from there to 1024 WGs. |
| Workgroup size sensitivity | **Negligible** [probed, T-O3]: 64→1024 threads spans only **0.22%** on a streaming read. Use 256 as the default; it is not a tuning knob for bandwidth-bound work. |

### 6a. The occupancy finding that matters most

T-O2 and T-O3 look boring in isolation — a streaming read saturates at 2 workgroups per WGP and does not care about block size. Their value is the **contrast** with the measured attention kernel:

| kernel | grid needed to plateau | at 32–36 WGs |
|---|---|---|
| simple `float4` streaming read (T-O2) | **72 WGs (2/WGP)** | already 96.5% of peak |
| vLLM Triton decode attention (measured 2026-08-19) | **128 WGs** | 2.6× slower than plateau |

The hardware does **not** need many workgroups to saturate memory. So the attention kernel's steep occupancy sensitivity is not "36 WGPs need feeding" — it is that **each of its workgroups issues too few outstanding memory requests**. Low memory-level parallelism per wave (dependent chains, LDS staging with `num_stages` synchronisation, one tile in flight) forces it to buy MLP by multiplying workgroups instead.

**Design consequence:** raising the split-K/segment count was a workaround, not the cure. The cure is more independent loads in flight *per wave* — which is exactly what a register-accumulator vector formulation gives (many independent per-lane loads) and what a `tl.dot` staged-tile formulation does not. This is now the primary argument for the rewrite, and it is stronger than the "no matrix unit" argument.

## 7. Scheduling & issue

- 1 instruction issue per SIMD32 per clock; no VOPD dual-issue (that is RDNA3) [published].
- Transcendental: **`__expf` measured at 0.191 issue/clk/SIMD vs 0.756 for `v_fma_f32` — exactly ¼ rate** [probed, T-C3], confirming the published expectation. **But it cannot bound the attention kernel:** ~8.3 M exps per token (12 q-heads × 43k ctx × 16 layers) at ~2.1×10¹² exp/s is **~4 µs against a 54 ms step**. Softmax arithmetic is free here; do not optimise it.
- Scalar unit executes address/control flow in parallel. **Block-table walks already ride the scalar path automatically** [probed, T-M8]: for a wave-uniform table index the compiler emits `s_load_dword` with **zero** `global_load_dword` for the table, versus 2 `global_load_dword` when the index is forced lane-varying. **But it makes no measurable difference** — 1.5651 ms vs 1.5710 ms (0.4%) when both variants walk the same blocks, because the table is ~512 B against ~32 MB of block data. **No action needed; this is already optimal and not a lever.**
- `cooperativeLaunch = 1` — device-wide sync grids are available if a persistent-kernel design is wanted [probed]; grid-wide barrier cost unmeasured [TBD → T-O6, only if a persistent-kernel design is pursued].
- Instruction cache ≈ 32 KiB per WGP [published, low confidence] — heavily unrolled dequant loops should watch I$ pressure via the T-I5 resource reports [design note].
- **Kernel launch overhead is small and graphs barely help it** [probed, T-O4]: free-running 2.81 µs/kernel, **dependent chain 3.11 µs**, graph-replayed 2.84 µs — only **1.09×**. A ~500-kernel decode step therefore costs **1.55 ms eager vs 1.42 ms in a graph**, about **3% of a 46.8 ms step**.
  **This corrects a standing assumption in this project.** The measured ~2× win from `FULL_DECODE_ONLY` graphs (10.3 → 19–20 t/s at TP=1) **cannot** be launch overhead — that is only 3%. It must come from eliminating *host-side framework* work per step (Python dispatch, tensor allocation, scheduler), which CUDA graphs also remove. Practical consequence: **a hand-written kernel need not fear launch cost**; ~3 µs per launch is affordable, and fusing kernels purely to reduce launch count is not worth doing.
- **Hardware queues: our inherited `GPU_MAX_HW_QUEUES=2` is actively limiting** [probed, T-O5]. Four concurrent streams overlap by 1.00× at q=1, **1.98× at q=2, 3.51× at q=4**, plateauing at q=8. **Set it to 4.** The value 2 was copied from community RDNA2 recipes and caps stream overlap at half what the hardware gives — relevant to any design overlapping collectives with compute, or serving multiple streams.

## 8. Atomics

| Item | Value | Src |
|---|---|---|
| Global fp32 atomic add | present (`atomic-fadd-*-insts`, buffer/global/flat variants) | [probed via LLVM features] |
| Peer (cross-GPU) atomics | **work exactly** — `atomicAdd`/`atomicCAS` into a peer's VRAM verified lossless, all directions, incl. the production pair | [probed] |
| LDS atomics throughput | **31.7 G/s uncontended, 26.4 G/s all-to-one — 1.2× penalty only** [probed, T-L3] |
| Global atomic throughput | **95.7 G/s to distinct addresses; 0.087 G/s all-to-one — a 1100× contention penalty** [probed, T-C7] |
| **Split-K combine strategy** | **Write partials + second combine kernel, never atomics: 0.207 ms vs 5.305 ms = 25.6× faster** [probed, T-C7]. Independently reproduces the Windows-PoC finding that switching their GEMV from `atomicAdd` to direct-store was the winning move. |

## 9. Interconnect / multi-GPU [probed 2026-08-18/19 — runtime path complete; kernel path TBD]

- Direct PCIe P2P: **real** (kernel-level peer reads/writes verified, not driver-staged); large-BAR 32 GB active; `amdgpu.pcie_p2p=Y`; `iommu=pt`.
- Bulk P2P: **14.04 GB/s** dev1↔dev3 (89% of Gen3 x16); 7.0 GB/s on x8-rooted pairs; scales with root-port width (link-limited, not host-limited).
- Small-message latency: **12.16 µs idle for a 10 KB `hipMemcpyPeer`, 14.08 µs while both GPUs are saturated (only 1.16× inflation)** [probed, T-N1]. *This supersedes the earlier 28.4 µs figure, which came from a torch-level copy with more per-call overhead.* Latency is flat with message size, so all-reduce cost is per-op, not per-byte. **128 all-reduces/token (TP=2) → a 1.56 ms/token hardware floor (1.80 ms under load).** ⚠️ **Corrected 2026-08-20 [T24].** This bullet previously read "against RCCL's measured ~82 µs/op ≈ 10.5 ms/token, so ~85% of TP=2 communication time is software overhead". That 82 µs was measured **outside the server**, with per-call host overhead included. Inside vLLM, RCCL runs as a node in the captured graph with no host round-trip and costs **~18 µs/op ≈ 2.35 ms/token** — about 6% of a decode token, not 29%. A custom push-based collective 3.7× faster standalone bought only **0.7 ms/token (1.9%)** end to end (T24). **Comms is not a lever on this machine; do not size work against the old figure.**
- Cross-process HIP IPC (`hipIpcGet/OpenMemHandle`): works — TP worker architecture supported.
- Cross-NUMA penalty: 1–5 µs — negligible.
- No XGMI. AMD validates P2P on Instinct only: *unsupported but functional*.
- vLLM custom all-reduce: kernels compile & register for gfx1030 but **workers crash at CUDA-graph capture** when enabled (gate bypassed via our patch). Unresolved [software, not silicon].
- **Kernel-initiated peer transport [probed, T-N2, devices 1↔3] — two results, one of them a blocker.**
  - **Push beats pull 2.5×: peer STORE 14.30 GB/s vs peer LOAD 5.70 GB/s.** The store figure matches the 14.04 GB/s SDMA copy, so kernel-initiated writes achieve full link bandwidth while reads pay PCIe round-trip latency. **Any custom collective on this hardware must be push-based.**
  - **Spin-wait flag polling across PCIe DOES NOT WORK.** A ping-pong handshake (peer writes payload + flag, partner spins on `volatile int*`) **timed out under all three fence variants** — none, `__threadfence()`, and `__threadfence_system()`. The waiting GPU never observes the peer's write: RDNA2 has no cross-device cache coherence, and `volatile` only forces a re-read from the local L2, which nothing invalidates.
  - **This is a strong candidate explanation for T18** (vLLM's custom all-reduce killing both TP workers at CUDA-graph capture): its one-shot algorithm is exactly peer-writes-plus-spin-barrier. A barrier that never observes its peer hangs, and a hung worker under graph capture is what we saw.
  - **Cross-device signalling: SOLVED [probed, T-N3/T-N4, devices 1↔3, 2026-08-20].** A custom collective is feasible; the earlier blocker was a memory *type*, not a hardware limit.
    - **System-scope atomic loads do NOT work** — the earlier hypothesis is disproved. Polling a peer-written flag with `__hip_atomic_load(..., __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM)` times out exactly like `volatile`, whether the writer uses a plain store or a system-scope atomic store. Peer atomics working (§8) does not extend to a peer's write becoming *observable* to a spinning reader.
    - **Host-resident coherent flags DO work: 1.58 µs one-way handshake** (`hipHostMalloc(..., hipHostMallocCoherent)`), 12× inside the 20 µs design target. A `volatile` negative control ran in the same rig and timed out, so this is a real cross-device observation, not an artefact of the harness.
    - **The deeper fact — a GPU cannot see peer-written data in its own memory when that memory is coarse-grained.** Default `hipMalloc` is coarse-grained, which the ROCm model only guarantees coherent at kernel-dispatch boundaries. A peer's stores land in the owner's DRAM correctly (verified by host readback) while the owner's own kernel reads its stale L2 forever. Neither a `__threadfence_system()` acquire nor per-dword system-scope atomic loads invalidate it.
    - **Fix: allocate the exchange staging buffer with `hipExtMallocWithFlags(..., hipDeviceMallocUncached)`.** A pushed 10 KB payload then verifies correct at **11.45 µs/op — 7.2× better than RCCL's 82 µs**, and inside the WS2 stretch target. Host-resident staging also works (14.20 µs) but is slower, since the receiver reads back across PCIe instead of from local DRAM. Only the staging buffer needs this; memory elsewhere stays cached and unaffected.
    - **`hipDeviceMallocFinegrained` is a silent no-op on this part** — the allocation succeeds, returns no error, and delivers stale data. Fine-grained is effectively an APU/XGMI feature. Do not trust it here.
    - **This pins down T18.** vLLM's custom all-reduce is one-shot peer-writes-plus-spin-barrier over ordinary coarse-grained memory; on this hardware that barrier can never observe its peer, so the workers hang — and a hung worker during CUDA-graph capture is exactly the crash we saw. The defect is the memory type, not gfx1030 kernel compilation.
    - Still open: whether the ordering survives CUDA-graph capture and long soaks (WS2.1).

## 10. Software stack ground truth

- **Triton `num_stages` on gfx1030: use 1.** The default (3) and even 2 multiply the K/V tile
  LDS footprint and halve occupancy at large head sizes; measured ~2× on prefill attention
  (T32) and reconfirmed on two hand-written kernels (T33/T34). This is the single biggest
  launch-parameter knob observed on this chip.
- **Fused-dequant Triton GEMMs lose to dequantize-then-rocBLAS at prefill shapes.** The W4
  fused kernel that is mandatory at decode (M=1) is ~3× off the vendor GEMM at M=8192; a
  single-pass dequant to dense fp16 (~2.5 bytes/element, ~1% of chunk time) followed by
  `torch.nn.functional.linear` restored GPTQ-band prefill for the AWQ build. This is the same
  structure Exllama uses internally for large M — when a quant path is slow at prefill, check
  whether it is fusing dequant into its own GEMM before tuning its tiles. [T36]
- **Custom Triton prefill-attention kernels lose to vLLM's stock kernel here.** Two structures
  measured (byte-sliced fused-dequant 0.70×, interleave-reconstructed deep-dot 0.58×, both
  correct): at prefill shapes KV is Infinity-Cache-served and the stock kernel's scheduling
  wins. The decode regime (no reuse, bandwidth-bound) is where custom kernels pay [T-A1/T23];
  the prefill regime is not. [T33/T34]

- **vLLM's RDNA-gated HIP kernels port to gfx1030 by widening their guards.** The
  `skinny_gemms_int4.cu` device code (`wvSplitK_int4_g`, used by RDNAHybridW4A16 for W4 decode
  GEMV) gates on `__GFX11__||__GFX12__`, but everything it needs — wave32, 64 KiB LDS,
  `v_dot2_f32_f16`, DPP `row_shr` — is gfx10.3 hardware. Adding `__GFX10__` to the guard (plus
  a scalar fallback for the bf16 dot, which gfx10 lacks) ran first try and beat Exllama q_gemm
  at 42k context (43.6 vs 41.4 t/s decode, T35). Check the guard before assuming a "gfx11+"
  kernel is off-limits; the actual requirements are often in gfx10's envelope.
- **Asymmetric W4 (uint4 + zero-points) narrows the kernel field sharply.** Exllama implements
  only uint4b8/uint8b128 — symmetric. On ROCm the uint4+zp takers at group_size 32 are
  RDNAHybridW4A16 (after the port above) and TritonW4A16 (correct everywhere, ~7 t/s decode at
  M=1 on this chip). Quant format choice is therefore also a kernel-path choice. [T35]

- Container: ROCm 7.2.3, torch 2.11.0+gitd0c8b1f, Triton 3.6.0, py3.12 (`vllm-gfx1030:0.27.1-patched`).
- **No vendor stack ships gfx1030 kernels.** Everything fast was earned by source build + patches.
- Upstream gates that exclude gfx1030 (all verified in source): vLLM `use_custom_allreduce()` → gfx94/95 only; QuickReduce → gfx94/95; `csrc/rocm/attention.cu` → gfx90a/942/950; RDNA GEMM files are `_rdna3` (WMMA); skinny-GEMM gate excludes gfx10x; AITER, CK flash-attn, FlashInfer: no RDNA2.
- Exllama GPTQ kernels (`q_gemm.cu`): compile clean for gfx1030 and are the current best W4 path [probed].
- llama.cpp: mainline flash-attn supports KV q8_0/q4_0 via templated vec kernels (`fattn-vec.cuh`) with explicit RDNA branches — the reference design for register-accumulator decode attention [published/source-read].
- Known-good env: `HSA_OVERRIDE_GFX_VERSION=10.3.0`, `TORCH_BLAS_PREFER_HIPBLASLT=0`, AITER off, `HIP_FORCE_DEV_KERNARG=1`, `PYTORCH_TUNABLEOP_ENABLED=1`.
- **Tooling constraints found during readiness review (2026-08-19):**
  - `rocm-smi --setperfdeterminism` is a **silent no-op** from the container (returns success, perf level stays `high`); host-side would need sudo we do not have. **Clock pinning is unavailable → all microbenchmarks must sample SCLK and normalize.**
  - `rocm-bandwidth-test` is **not in the image** — host↔device and P2P bandwidth must be hand-written `hipMemcpy`/kernel benches.
  - Triton ISA is dumpable: `TRITON_CACHE_DIR` + `TRITON_KERNEL_DUMP=1` leaves readable `.amdgcn` — the mechanism T-I1/T-I2/T-I3 rely on is confirmed working.
- **Working tooling recipes [probed, T-I5/T-I6]:**
  - Per-kernel resources: `hipcc --offload-arch=gfx1030 -O3 -Rpass-analysis=kernel-resource-usage -c <src>` → stderr gives TotalSGPRs, VGPRs, ScratchSize, **Occupancy [waves/SIMD]**, SGPR/VGPR spills, LDS bytes/block.
  - ISA: `cd /tmp && hipcc --offload-arch=gfx1030 -O3 --save-temps=obj -c <src>` → `/tmp/*gfx1030*.s`.
  - Profiling: `rocprofv3 --kernel-trace --stats -f csv -d <dir> -o run [--pmc ...] -- <short-lived binary>`. **Must wrap a binary that exits** — the report is only written on a clean exit, and a server under the profiler did not exit across three attempts (2026-08-19). *Note: plain vLLM does exit cleanly on SIGTERM in ~1.2 s; it was the profiler that prevented shutdown, so the rule is about the instrumented process, not about vLLM.*
  - **Counter reality on gfx1030: 78 counters total, but ZERO cache counters** — no `TCC_*` (L2), no `TCP_*` (L1), no MALL/Infinity-Cache counters. Only `SQ_WAVES`/`SQ_LEVEL_WAVES` and coarse `GRBM_*_BUSY` signals. **Consequence: T-M2's Infinity Cache conclusions rest entirely on bandwidth inference — no hardware counter can corroborate them.** Plan accordingly and state it in the results.

## 11. Measured performance anchors (application level, this machine)

**All percentages below are against the measured 506 GB/s achievable read bandwidth (T-M1), not the 512 theoretical and not the ~244 GB/s figure used earlier in this project — that number was an achieved decode rate mistaken for a ceiling.**

| Metric | Value | % of achievable |
|---|---|---:|
| Decode weight-streaming — ⚠️ **superseded 2026-08-20 [T25]** | ~~TP=1: 16.67 GB / 46.82 ms = 356 GB/s~~ That figure divided weight bytes by the **whole token time**, charging attention, comms, GDN, elementwise work and idle to weight streaming — a lower bound, not a measurement. Profiled in-server at TP=2: weight kernels take **20.03 ms for 8.588 GB/card = 460 GB/s**. | ~~70%~~ **91%** |
| Same, TP=2 (per card) | 8.34 GB / 33.93 ms = 246 GB/s | **49%** |
| **Attention decode kernel — the rewrite's baseline** [T-A1, harness at production shapes, 64 segments] | fp16 KV @43k **173.6 GB/s (34.3%)** · int8 KV @43k **85.1 GB/s (16.8%)** · @4k ctx only 11.6% / 6.0% | |
| ↳ **the int8 anomaly** | int8 moves half the bytes but achieves half the bandwidth, so it is **time-neutral (0.51 vs 0.52 ms) instead of the ~2× it should be**. There is no hardware reason: the int8 dot units are reachable (T-I2) and the dequant chain has 1.88× headroom (T-C4). **Fixing the kernel's int8 path alone is worth ~2× on attention at long context.** | |
| Same, before the segments fix (32 workgroups) | 32.2 GB/s | **6%** |
| int8 KV inside that kernel | 2.5× slower per byte than fp16 [dequant path defect — T-C4/T-I2] | |
| Prefill (with the TILE 16 clamp from the LDS patch) | ~550 tok/s | [T-L1 may lift this] |
| End-to-end decode TP=2 | 29.4 t/s short-ctx · 18.4 t/s @41k · int8 KV neutral at short ctx | |
| Server-level graph capture | ⚠️ **Corrected 2026-08-20 [T26].** The 2× FULL_AND_PIECEWISE regression (T7) **does not reproduce** — it dated from the throttled, pre-Exllama-dispatch era. Re-measured with the current stack, FULL_DECODE_ONLY and FULL_AND_PIECEWISE are **within 0.1%** of each other (28.27 vs 28.26 ms/token), and enabling compilation (`mode:3`, VLLM_COMPILE) is worth **+20%**. `mode:0` was adopted on the strength of the obsolete result and cost ~20% for months. | |
| **rocBLAS fp16 GEMV at real layer shapes** [probed, T-A2] | 5120×5120 **443.9 GB/s (87.7%)** · 5120×17408 260.7 (51.5%) · lm_head 5120×248320 **277.7 (54.9%)**. **M=1 through M=8 are identical to within 1%** — already memory-bound, so batching to M=8 is free. The two wide shapes leave ~1.8× on the table; the fp16 lm_head is 15.3% of per-token bytes, so its 54.9% efficiency is worth ~4 ms/token at TP=1. | |
| **Cooperative grid barrier** [probed, T-O6] | **0.43–0.53 µs** at 72–144 workgroups (2.89 µs at 36). Persistent single-device kernels are viable. *Cross-device persistent designs are not — see T-N2.* | |

**Consequence:** the attention kernel is at **17% of what this silicon does**, not the ~35% previously believed — the headroom is roughly 6×, not 3×. Weight streaming at 70% is respectable; TP=2's 49% per card reflects comms overhead and less work per card, and is a separate lever.

## 12. Gap index

*(✅ = measured; result folded into the sections above. **All 37 tests complete as of 2026-08-19.** The profile has no open TBDs.)*

| Gap | Test |
|---|---|
| Real achievable DRAM bandwidth (read/write/copy) | T-M1 — ✅ 506 GB/s |
| Infinity Cache behavior under mixed streaming+resident workloads | T-M2 — ✅ IC 1890 GB/s; KV cannot stay resident |
| Latency ladder (L0/GL1/L2/IC/DRAM) | T-M3 — ✅ ladder 20/40/117/150/180 ns |
| Nontemporal load semantics | T-M4 — ✅ nontemporal is a no-op |
| Optimal load width / vectorization / gather cost | T-M5 — ✅ stride ∝; gather free |
| Line size & coalescing granularity | T-M6 — ✅ 128 B granularity |
| Scalar-path loads for uniform data | T-M8 — ✅ automatic, not a lever |
| Dot/FMA/packed instruction throughputs | T-C1, T-C5 |
| wave32 vs wave64 | T-C2 |
| `v_exp_f32` rate | T-C3 |
| int8→fp16 convert cost in isolation + **GPTQ int4 extract/dequant chain** | T-C4 |
| Cross-lane reduction primitive set & throughput | T-C8 |
| bf16 support | T-C6 |
| Global atomic throughput | T-C7 |
| >64 KiB LDS per workgroup | T-L1 — ✅ 64 KiB hard cap |
| LDS bank-conflict penalty | T-L2 |
| LDS atomics | T-L3 |
| CU vs WGP mode control | T-L4 |
| lds-misaligned-bug impact | T-L5 |
| Occupancy vs VGPR curve | T-O1 |
| WGs/WGP to saturate memory | T-O2 — ✅ 2 WG/WGP saturates |
| Workgroup size sweep | T-O3 — ✅ block size irrelevant |
| Launch overhead ± HIP graphs | T-O4 |
| HW queue count effect | T-O5 |
| Triton `tl.dot` lowering | T-I1 |
| Triton int8 dot feasibility | T-I2 |
| Triton num_stages on RDNA2 | T-I3 |
| Compiler resource-report workflow | T-I5 — ✅ recipe established |
| rocprof counter availability | T-I6 — ✅ 78 counters, NO cache counters |
| rocBLAS small-M GEMV map — ✅ **memory-bound already; M=1..8 identical** | T-A2 |
| Host↔device bandwidth | T-M7 — ✅ 7.1 GB/s (x8 root) |
| Kernel-initiated peer transport (BW, visibility latency, fence contract) | T-N2 |
| Cooperative grid-sync barrier cost | T-O6 (conditional) |
