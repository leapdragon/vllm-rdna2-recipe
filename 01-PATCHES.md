# 01 — What's in the patches, how to apply them, and what will bite you

Copyright © 2026 Aron Hsiao · GPL-3.0-or-later (see LICENSE)

Nine patches against **pristine vLLM 0.27.1**. Plus two net-new plugin
packages under `builds/shared/plugins/`, which patch nothing — they install and register themselves.

## Applying

```bash
pip download vllm==0.27.1 --no-deps --no-binary :all: -d /tmp   # or clone the v0.27.1 tag
tar xzf /tmp/vllm-0.27.1.tar.gz -C /tmp && cd /tmp/vllm-0.27.1
for p in /path/to/recipe/patches/*.patch; do patch -p1 < "$p"; done
```

All nine apply cleanly to unmodified 0.27.1, and the fully-patched tree has been verified
byte-for-byte against the tree every published number was measured on (6,361 files compared;
last verified 2026-08-23, after patch 0009). If one fails
because you are on a different vLLM version, **do not force it** — read what it accomplishes below
and re-implement, then run its verification line. A fuzzy-applied hunk that lands in the wrong
place is worse than no patch, because it fails silently.

---

## 0001 — gfx10x platform support · *mandatory*

Adapted from upstream vLLM PR #52391 (unmerged at time of writing).

**What:** teaches vLLM that gfx10xx exists — adds `on_gfx10x()` to `vllm/platforms/rocm.py`, gates
the W4A16 Triton kernel on it, fixes a dtype cast in the FLA (linear-attention) path that the
hybrid Qwen3.x models need, and adds `gfx1030` to the base-image arch list.

**Why:** without it, vLLM does not recognise the platform and the hybrid model's GatedDeltaNet
layers fail on a dtype mismatch.

**Verify:** `python -c "from vllm.platforms.rocm import on_gfx10x; print(on_gfx10x())"` → `True`.

---

## 0002 — LDS tile fix + prefill tuning for `head_dim ≥ 256` · *mandatory*

**What:** two things in the same gfx10x branch. (a) Reduces the attention kernel's LDS tile so
it fits Navi 21's 64 KiB workgroup limit — without this the kernel will not launch at
`head_dim=256`. (b) Sets `num_warps=4, num_stages=1` for the same branch: **stages=1 is ~2× on
prefill-attention throughput** (software pipelining doubles the K/V LDS footprint and halves
occupancy at this head size). Worth **+35% end-to-end prefill at 37k context** (521 vs 386
tok/s), greedy outputs byte-identical.

**A measured dead end inside this patch's territory:** widening the query tile (BLOCK_M
32–128; upstream ships exactly this fix for Blackwell as `tuned_large_head`) is uniformly
*worse* on gfx1030 — the fp32 accumulator is `BLOCK_M × 256` per program and register pressure
beats FlashAttention amortisation. Measured, not theorised; do not re-try it.

**Verify:** the server boots (without the LDS part you get a launch failure at the first
attention call), and cold-prompt prefill at ~15k context reaches ~750 tok/s on the reference
hardware (~600 indicates the stages fix is missing).

---

## 0003 — softmax segments for decode · *worth 1.6×*

**What:** scales `num_par_softmax_segments` so that at decode the launch grid reaches
`MIN_LAUNCH_GRID_SIZE_2D`.

**Why:** with 4 KV heads the stock decode grid is far too small to fill 36 WGPs — the GPU sits
mostly idle. Measured **11.5 → 18.4 t/s at 41k context**.

**Verify:** compare decode t/s at long context before and after; the gain grows with context.

---

## 0004 — W4 GEMM blocking 128 → 256 · *worth +3.3%*

**What:** one constant, `BLOCK_KN_SIZE`, in the Exllama GPTQ kernel.

**Why:** we swept it against this model's actual shapes. 256 gives 8 wave32 waves per block and a
shallower K-split (less atomic accumulation traffic): mean **400.0 GB/s vs 364.8** at 128.
64 is much worse (305.7); 512 collapses small shapes (o_proj falls to 18 blocks).
**Our starting hypothesis — that small shapes were block-starved and wanted *smaller* blocks —
was exactly backwards.** Measure before you assume.

**Verify:** rebuild the extension and re-run the benchmark; expect ~+3% at every context.

---

## 0005 — base image gfx1030 arch · *mandatory if you build the base yourself*

**What:** adds `gfx1030` to `PYTORCH_ROCM_ARCH` in vLLM's own `docker/Dockerfile.rocm_base`.

**Where:** vendored pre-applied as [`containers/Dockerfile.rocm_base`](containers/Dockerfile.rocm_base);
`containers/build.sh --base` builds it with `PYTORCH_ROCM_ARCH=gfx1030`.

**Why:** that Dockerfile builds PyTorch from source for a fixed arch list that excludes gfx1030.
Without this you get `HIP error: invalid device function` on the first matmul. **This one line is
the difference between "vLLM cannot run on this card" and "vLLM runs on this card."**

**Verify:** with GPU devices attached to the container —
```bash
docker run --rm --device /dev/kfd --device /dev/dri --group-add render --group-add video \
  -e HSA_OVERRIDE_GFX_VERSION=10.3.0 <image> python3 -c \
  'import torch; print(torch.cuda.get_arch_list()); print(torch.cuda.get_device_properties(0).gcnArchName)'
```
Expect `['gfx1030']` and `gfx1030`.

⚠️ **Run this with the devices attached.** Without `--device /dev/kfd --device /dev/dri`, torch
cannot initialise HIP and `get_arch_list()` returns an empty list **with no error** — which looks
exactly like a failed arch patch on a perfectly good build. This is the pitfall pattern in this
repo in miniature: a check that fails open teaches you the wrong thing.

---

## 0006 — RDNA-hybrid W4A16 kernel on gfx1030 · *mandatory for asymmetric W4 (AWQ-style) checkpoints*

**What:** three files. Extends `csrc/rocm/skinny_gemms_int4.cu`'s device-code guard from
`__GFX11__||__GFX12__` to include `__GFX10__` (with a scalar fallback for the bf16 dot, which
gfx10 lacks — unused under fp16 serving), widens the matching Python gate in
`rdna_hybrid_w4a16.py`, and teaches that kernel two gfx1030 lessons: gfx10 prefill tile tables
with `num_stages=1`, and a **dequant-to-dense + rocBLAS route for M ≥ 256** (a single-pass
Triton kernel expands the packed weights to fp16, then `torch.nn.functional.linear`). Also sets
`num_stages=1` in `triton_w4a16.py`'s gfx10x launches.

**Why:** symmetric GPTQ checkpoints run through Exllama (see the pitfall below), but
**asymmetric uint4 — compressed-tensors AWQ exports with real zero-points — is outside
Exllama's supported types** (`uint4b8`/`uint8b128` only; its zero-point path *synthesizes*
zeros for symmetric checkpoints, it does not consume real ones). Without this patch the only
willing kernel is the pure-Triton fallback at ~7 t/s decode on a 27B. With it,
`RDNAHybridW4A16LinearKernel` becomes eligible: HIP skinny GEMV for decode (49.8/43.9 t/s at
3.8k/41k — at parity with the Exllama route, ahead at long context) and the dense-matmul route
for prefill (753/736/524 — the fused Triton GEMM it replaces is ~3× slower; rocBLAS wins at
prefill shapes, same reason the custom prefill attention kernels lost).

The guard extension itself is the transferable part: the kernel needs wave32, 64 KiB LDS,
`v_dot2_f32_f16`, and DPP `row_shr` — all gfx10.3 hardware. **Check the arch guard before
assuming a "gfx11+" kernel is off-limits**; the ISA was inspected to confirm all three
instruction classes survive compilation for gfx1030.

**Cost:** touches a `.cu` file, so the C++ extension must be rebuilt after applying
(`VLLM_TARGET_DEVICE=rocm PYTORCH_ROCM_ARCH=gfx1030 python3 setup.py build_ext --inplace`,
~4 min incremental). Harmless to symmetric/GPTQ deployments — the launcher's default disable
list keeps the hybrid kernel off there.

**Verify:** boot an asymmetric-W4 model with
`DISABLED_KERNELS=RDNA3W4A16LinearKernel,ConchLinearKernel` (the shared launcher exposes this
override) and look for `Using RDNAHybridW4A16LinearKernel for CompressedTensorsWNA16` in the
log. See `builds/cyankiwi-Qwen3.8-27B-AWQ-INT4/` for a complete working configuration.

---

## 0007 — CUDA moe_wna16 MoE kernel on ROCm/gfx1030 · *for W4 MoE models (e.g. Qwen3.5-122B-A10B)*

**What:** seven files, ~70 lines. vLLM ships a dedicated small-batch W4A16 MoE GEMM
(`csrc/libtorch_stable/moe/moe_wna16.cu`) that was CUDA-only at four layers: the CMake
source list, the op declaration/binding, and two Python gates. The device code needed
exactly three portable substitutions — the `lop3` PTX helper's only LUT is `(a & b) | c`,
`prmt` maps to HIP's native `__byte_perm`, and `nv_bfloat16` gets type aliases (hipify
converts the header include but not the type names) — plus a CAS-loop fp16 `atomicAdd`
(gfx10 has no fp16 global atomics) and one `res2 = {};` → `scalar_t2()` (ambiguous for
HIP's `__half2`). The bf16 code paths self-discard when `__CUDA_ARCH__` is undefined, so
they cost nothing. Touches `.cu`/CMake → rebuild the extension after applying.

**Why, honestly:** at real decode batch sizes it measures ~1.1× vLLM's Triton fused-MoE
path — worthwhile but not transformative, and the port's bigger value was diagnostic: on
the official Qwen 122B quant, decode is split roughly evenly between the MoE experts,
the checkpoint's *unquantized bf16 backbone* (see the build's BUILD.md), and
pipeline-parallel overhead. Don't expect this patch alone to change end-to-end numbers;
apply it because it is correct, slightly faster, and required groundwork for any future
W4 MoE work on this platform.

**Verify:** `python3 -c "import torch, vllm._custom_ops; print(hasattr(torch.ops._moe_C, 'moe_wna16_gemm'))"`
inside the container → `True`, and the serving log stops routing small batches through the
Triton path. Correctness: greedy outputs vs a pre-patch baseline (ours: 7/8 identical,
factual PASS).

---

## 0008 — MoE decode skinny GEMV · *the kernel that makes W4 MoE decode fast on gfx1030*

**What:** a wave-per-output-row W4A16 expert GEMV pair (six files: the kernel appended to
`csrc/rocm/skinny_gemms_int4.cu`, its binding/declaration, and dispatch hooks). At decode
batch sizes, one wave computes one output row with lanes striding along K — coalesced
weight streaming, the thing the tile-based MoE kernels structurally lack (they read across
K; measured 10×+ off bandwidth). Activations stage in LDS, silu·mul fuses into the
gate_up epilogue, and the down kernel does the topk-weighted combine internally — one
write, no atomics, and the whole moe_align machinery is bypassed. Plain fp32 FMA: this
regime is bandwidth-bound and dequant is free on this chip. **432 GB/s effective (85% of
the memory ceiling), linear in M from 1 to 8.** Gated to M ≤ 8, symmetric int4, SILU,
fp16; `VLLM_ROCM_MOE_SKINNY=0` disables. Touches a `.cu` → rebuild the extension.

**Measured on Qwen3.5-122B-A10B (Intel AutoRound quant, PP=3)**: decode 15.6 → **26.9 t/s**
at 3.5k (25.4 @14k, 21.8 @42k), prefill unaffected — short-context parity with llama.cpp
on the same cards.

**The integration pitfall this patch survived, so you don't have to**: vLLM's modular MoE
framework defines `TritonExperts.apply` *and* an override `TritonWNA16Experts.apply` with
near-identical bodies in the same file. A dispatch hook in the former is silently dead for
WNA16 models — no error, no log, no change. If your hook "doesn't fire", overlay a
one-shot debug print (the launcher's `EXTRA_MOUNT` knob exists for exactly this) before
assuming your kernel is at fault.

---

## 0009 — MTP speculative decoding under pipeline parallelism · *for MTP models served with PP > 1; requires the V2 model runner*

**What:** five Python files that make MTP drafting work when the model is layer-split
across GPUs. Two are a backport of upstream PR #46994 (author eastwood-c, snapshot at head
`bee2011` while still open): `qwen3_5_mtp.py`/`deepseek_mtp.py` gain `SupportsPP` (without
it the engine refuses to even build the draft under PP) and a last-rank branch in
`forward` (the drafter lives entirely on the last PP stage, where the target's hidden
states are local — without the branch it projects uninitialized intermediate tensors).
Two more carry the PR's V2-runner PP fixes (`gpu/pp_utils.py` verbatim, one relay hunk
hand-ported into `gpu/model_runner.py`): sampled-token broadcasts padded to a fixed width,
and proposed draft tokens relayed to non-last ranks. The fifth (`gpu_model_runner.py`, the
V1 runner) only guards `self.drafter` accesses so V1 PP boots stop crashing in
profile/KV/cudagraph init. Python-only — no extension rebuild.

**The hard-won constraint: MTP under PP requires `VLLM_USE_V2_MODEL_RUNNER=1`.** The V1
runner's drafter path page-faults under PP (reproduced on two models, dense and MoE, with
cudagraphs off and on, int8 and fp16 KV) — upstream has zero V1 PP+spec test coverage;
every PP row in the PR's validation ran V2. This patch makes V1 fail *cleanly* instead of
crashing, and makes V2 work. First V2 boots on gfx1030 measured **identical** to V1 at
MTP=0 (decode and prefill), so the runner switch itself is free.

**Measured on Qwen3.5-122B-A10B (Intel AutoRound, PP=3, packed MTP head)**: correct greedy
output, 81% draft acceptance at K=2 — and with the two companion fixes, **39.7–41 / 33.4 /
21.9 t/s at 3.5k/13k/40k vs 27.0/25.7/22.1 at MTP=0** (past llama.cpp on the same cards).
The companions matter as much as the patch: (1) the fd_rdna2 plugin generalized to GQA ≤ 16
so verification (q ≤ 4) and drafter attention ride the flash-decode kernel — without it,
verification runs the context-proportional prefill-class path and MTP *regresses*; (2)
TunableOp rows for the fp16 lm_head shapes (`vllm::rocm_unquantized_gemm`'s skinny path is
CDNA-gated, and the default Tensile pick runs the M ≤ 4 logits GEMM ~4× slow; in-server
tuning never reaches the shape — tune it offline, one-liner in the Intel BUILD.md). MoE MTP
heads shipped dense also need the checkpoint-side treatment in `builds/Intel-.../convert.py`
(+`quantize_mtp.py`) — the draft otherwise builds quantized against dense tensors, or
starves the last stage's KV.

---

## The plugins (net-new, patch nothing)

```bash
pip install ./builds/shared/plugins/fd_rdna2 ./builds/shared/plugins/ar_rdna2
```

Both register through vLLM's `vllm.general_plugins` entry point and are **off unless enabled**:

| | env | what it does | worth |
|---|---|---|---|
| `fd_rdna2` | `FD_RDNA2=1` | replaces decode attention with a flash-decode kernel using byte-sliced `tl.dot` over the int8 KV layout; **also handles speculative-verification batches** (`max_seqlen_q` ≤ `FD_MAXQ`=4). Geometry-general since 2026-08-23: shapes derive from the cache, any GQA ≤ 16 (the 27B's 24q/4kv and the 122B's 32q/2kv both covered); one-KV-pass batching for `nq × PAD ≤ 32` columns, per-position passes of the same kernel beyond. Correctness harness: `verify/fd_gqa_test.py` | context slope **6.06× flatter**; makes MTP a win instead of a large loss |
| `ar_rdna2` | `AR_RDNA2=1` | replaces the TP=2 all-reduce with a push-based one-shot collective | +1.9% |

⚠️ **If you enable MTP, you need this exact `fd_rdna2`.** Speculative verification carries
multiple query tokens per step; an attention override that only gates on `q == 1` silently
delegates every verification pass to the stock kernel — whose context slope is ~6× worse — and
MTP measures as a large regression while looking perfectly healthy.

**Honest note on mechanism:** delivery is idiomatic (entry points), but once loaded each one
*monkey-patches* a vLLM function — `triton_attn.unified_attention` and
`GroupCoordinator._all_reduce_out_place` respectively. The latter is a private method. This binds
to names and signatures vLLM never promised to keep stable, so **expect these to need rework on
any vLLM upgrade**, and see the pitfall below about silent inertness.

---

# Pitfalls — the things that look like something else

Each of these cost us hours. They are the real content of this repo.

**TunableOp's tuning mode stalls prefill for minutes, unpredictably.** `PYTORCH_TUNABLEOP_ENABLED=1`
with tuning active autotunes every never-seen GEMM shape by timing many rocBLAS algorithms
mid-request — and prefill M is prompt-length-dependent, so nearly **every fresh prompt is a new
shape**. We measured identical ~3.4k prompts at 771 tok/s and then 37 tok/s back to back; one
logged search chose `Default` after minutes of work. Run production with
`PYTORCH_TUNABLEOP_TUNING=0` (lookup-only; tuned shapes use the csv, unseen shapes take the
default instantly) and re-enable tuning only for deliberate offline sessions. Two more reasons:
tuning mode perturbs greedy outputs (breaks byte-identical validation), and on this machine its
allocation churn correlated with an `svm_range_restore_work [amdgpu]` kernel storm that preceded
two hard system crashes. If you do run an offline tuning session, raise the engine watchdog
first (`EXEC_TIMEOUT=3600`, i.e. `VLLM_EXECUTE_MODEL_TIMEOUT_SECONDS`): a tuning forward pass
can stall for minutes, and the default 300 s watchdog declares the engine dead mid-tune — a
clean shutdown that looks like a crash. The csv flushes incrementally but the LAST
entries are written at process exit — stop a tuning session gracefully (`docker stop`, not
`docker rm -f`/SIGKILL) or the tail of what it tuned is lost. Expect diminishing returns fast:
chunk-sized shapes tune in one long prompt; everything after that is one-off
prompt-length-dependent tail shapes worth nothing in lookup-only production. One class of
shape never gets tuned in-server at all in our experience: the fp16 logits GEMM (hidden ×
vocab at M ≤ 4). On CDNA, vLLM routes it to a hand-written skinny kernel; that path is
arch-gated off gfx1030, the default Tensile pick runs it ~4× slow (10.6 ms per speculative
verification step on the 122B), and in-server tuning sessions never touched the shape. Tune
it OFFLINE with a bare `torch.nn.functional.linear` loop under `PYTORCH_TUNABLEOP_TUNING=1`
and merge the resulting `tn_<vocab>_*` rows into your per-rank csvs — the one-liner is in
the Intel build's BUILD.md.

**Triton's default `num_stages` costs ~2× on this chip.** Triton defaults to 3-stage software
pipelining, which multiplies each kernel's K/V tile footprint in LDS; at head_size 256 that
halves occupancy. Pass `num_stages=1` explicitly in every Triton kernel you write for gfx1030 —
this single knob was worth 2× on prefill attention and was reconfirmed on two further kernels.

**A plugin can load, log nothing, and be completely inert.** `init_logger("name")` creates a
logger outside vLLM's configured `vllm.*` namespace, so every INFO record is discarded. The server
boots, serves correct text, and gives no sign either way. Use `init_logger("vllm.yourname")` and
log the first occurrence of every delegation reason, so "is my kernel actually running" is
answerable from the log rather than assumed.

**TunableOp perturbs greedy output on the first run after boot.** `PYTORCH_TUNABLEOP_ENABLED=1`
autotunes unseen GEMM shapes by trying several rocBLAS algorithms, which shifts numerics enough to
flip a token. A correctness divergence that appears *only* on the first run after boot, and never
again, is this — not your kernel. Check whether `tunableop/*.csv` grew before blaming your code.

**Coarse-grained memory is invisible to a peer mid-kernel.** Plain `hipMalloc` is coarse-grained:
a peer's writes land in DRAM correctly but the owner's running kernel reads its stale L2 forever.
No fence and no system-scope atomic load fixes it. Allocate exchange buffers with
`hipExtMallocWithFlags(..., hipDeviceMallocUncached)`. **`hipDeviceMallocFinegrained` is a silent
no-op on this part** — it succeeds, returns no error, and delivers stale data.

**Device-memory flags cannot be polled across PCIe.** A spin-wait on a peer-written flag never
observes the write, under any fence or atomic scope. Host-resident coherent memory
(`hipHostMalloc(..., hipHostMallocCoherent)`) works and gives a 1.58 µs handshake.

**Never mount host ROCm into the container.** Mixed host ROCm versions cause page faults. Use the
in-container ROCm exclusively.

**Pick the weights kernel by quant format; never accept the default.** On gfx1030 vLLM's
dispatch falls to `TritonW4A16LinearKernel`, which is ~4–7 t/s class on a 27B. For **symmetric**
W4 (GPTQ, uint4b8): force Exllama with
`VLLM_DISABLED_KERNELS=RDNA3W4A16LinearKernel,RDNAHybridW4A16LinearKernel,TritonW4A16LinearKernel,ConchLinearKernel`
(the shared launcher's default). For **asymmetric** W4 (AWQ-style, real zero-points): Exllama
*cannot* implement it — apply patch 0006 and disable only
`RDNA3W4A16LinearKernel,ConchLinearKernel` so `RDNAHybridW4A16LinearKernel` is chosen. The
forcing list is also your tripwire: with the alternatives disabled, a format the kernel can't
take fails loudly at boot instead of silently serving at 7 t/s.

**Benchmarks that fit in Infinity Cache lie.** One attention layer's KV at 43k is ~45 MB and sits
in the 128 MB cache, reporting >1000 GB/s. The server reads 16 layers' worth per token and nothing
is reused. Rotate through 16 buffers to make a harness DRAM-resident.

**Measure inside the serving process.** An out-of-server microbenchmark told us TP=2 comms cost
10.5 ms/token; in-server it is ~2.35 ms. We built a working collective for a deficit that was 4×
smaller than believed. Likewise, timing a Python loop charges ~15–20 µs of dispatch to every call —
time under CUDA-graph capture instead.

**Thermals silently cap everything.** Confirm your fan control is running before trusting any
measurement. A missing fan controller once made this machine look 3× slower than it was.

## Measured dead ends — do not spend time here

| Route | Why it is dead |
|---|---|
| Wide `dwordx4` KV loads | Loading already runs at **83% of peak**; widening changes nothing. Also unreachable: at a 520-byte KV entry stride, V rows are 4-byte aligned. |
| Re-laying-out KV to 544 B for alignment | The alignment it buys is worth nothing (above). |
| Plain FMA instead of `tl.dot` | **4.9× slower** at equal width. The dot units dominate; FLOP-budget arguments mislead. |
| A custom W4 GEMV kernel | Weight streaming is already at **91% of achievable**. Max win ~1.8 ms/token even if perfect. |
| More all-reduce work | In-server comms is ~2.35 ms/token total; our collective already took 0.7 of it. |
| `GPU_MAX_HW_QUEUES=8` | **32% regression** vs 4. Four is the optimum. |
| Hoisting the attention range-mask behind a full-tile branch | **71% slower** — duplicating the loop body costs more than the selects it removes. |
| Custom Triton PREFILL attention kernels (two structures) | Byte-sliced fused-dequant: **0.70×** the stock kernel; interleave-reconstructed deep-dot: **0.58×**. Both correct. At prefill shapes KV is served from Infinity Cache, so the decode kernel's load-path wins don't apply and the stock kernel's scheduling wins. Custom kernels pay at *decode* (no reuse, bandwidth-bound), not prefill. |
| Prefill chunk size ≠ 8192 (`--max-num-batched-tokens`) | 4096 and 16384 both measurably worse at mid/long context. 8192 is the optimum. |
| Tile-tuning a fused-dequant Triton GEMM for prefill weights | Two tile configs measured on the hybrid kernel: no change from 246 tok/s. The structure is the problem, not the tiles: dequantize to dense fp16 (~1% of chunk time) and call rocBLAS — 3× faster, and exactly what the Exllama path does internally for large M. |
| Wider prefill query tiles (BLOCK_M 32–128) | Uniformly worse — the fp32 accumulator is BLOCK_M×256 per program; register pressure beats amortisation. Upstream ships this as a *fix* for Blackwell; on gfx1030 it is a regression. |
