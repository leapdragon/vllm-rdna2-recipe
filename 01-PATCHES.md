# 01 — What's in the patches, how to apply them, and what will bite you

Copyright © 2026 Aron Hsiao · GPL-3.0-or-later (see LICENSE)

Five patches, 182 lines total, against **pristine vLLM 0.27.1**. Plus two net-new plugin
packages under `plugins/`, which patch nothing — they install and register themselves.

## Applying

```bash
pip download vllm==0.27.1 --no-deps --no-binary :all: -d /tmp   # or clone the v0.27.1 tag
tar xzf /tmp/vllm-0.27.1.tar.gz -C /tmp && cd /tmp/vllm-0.27.1
for p in /path/to/recipe/patches/*.patch; do patch -p1 < "$p"; done
```

All five apply cleanly to unmodified 0.27.1 (verified with `patch --dry-run`). If one fails
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

## 0002 — LDS tile fix for `head_dim ≥ 256` · *mandatory*

**What:** reduces the attention kernel's LDS tile so it fits Navi 21's 64 KiB workgroup limit.

**Why:** Qwen3.8-27B has `head_dim=256`. The upstream tile overflows 64 KiB and the kernel will
not launch. **This is a hard architectural cap, not a tuning knob.**

**Verify:** the server boots and serves a request. Without it you get a launch failure at the
first attention call.

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

## The plugins (net-new, patch nothing)

```bash
pip install ./plugins/fd_rdna2 ./plugins/ar_rdna2
```

Both register through vLLM's `vllm.general_plugins` entry point and are **off unless enabled**:

| | env | what it does | worth |
|---|---|---|---|
| `fd_rdna2` | `FD_RDNA2=1` | replaces decode attention with a flash-decode kernel using byte-sliced `tl.dot` over the int8 KV layout; **also handles speculative-verification batches** (`max_seqlen_q` ≤ `FD_MAXQ`=4) by packing all query positions into one KV pass | context slope **6.06× flatter**; makes MTP a win instead of a 3.6× loss |
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
two hard system crashes.

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

**Force the Exllama kernel.** On gfx1030 vLLM's dispatch picks `TritonW4A16LinearKernel`, which is
~4 t/s class on a 27B. Set
`VLLM_DISABLED_KERNELS=RDNA3W4A16LinearKernel,RDNAHybridW4A16LinearKernel,TritonW4A16LinearKernel,ConchLinearKernel`.

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
