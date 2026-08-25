# TROUBLESHOOTING.md — failure modes that lie to you

**Copyright © 2026 Aron Hsiao. Licensed under the GNU General Public License v3.0 or later.**

This file documents a class of failures we hit on this platform whose error
messages point *away* from their actual cause. Each entry is symptom-first.
The common thread: on a mixed or unsupported-hardware system, the ROCm stack
does not fence off what it can't handle — it lets one bad component silently
poison the healthy ones, at three different layers, with three different
misleading symptoms.

The concrete trigger for all of section 1–3 on our rig was adding an ancient
**Polaris (gfx803, pre-Vega) display card** alongside four supported RDNA2
cards. One day of debugging later: **do not put pre-Vega silicon in a ROCm
compute machine, even display-only.** Both ROCm userspace and the kernel KFD
assume it doesn't exist, and neither isolates it. But the diagnostic lessons
generalize beyond that one card, so they're written up here.

---

## 1. "No ROCm-capable devices" — but every card is healthy

**Symptom.** `rocminfo`, llama.cpp, vLLM, PyTorch: all report zero GPUs (or
`HSA_STATUS_ERROR: A generic error has occurred`). rocm-smi and the amdgpu
kernel driver see every card fine.

**Cause.** ROCR-Runtime's device discovery is **all-or-nothing**: agent
creation walks every KFD node, and an unhandled property on *any* node —
in our case a pre-Vega doorbell type — throws `HSA_STATUS_ERROR`, which
aborts enumeration of *every* GPU. One obsolete display card ⇒ four
supported cards vanish.

**`ROCR_VISIBLE_DEVICES` cannot save you.** The filter is applied *after*
the crash point. Masking the sysfs topology node fails too (the runtime
cross-checks KFD ioctls).

**Fixes, best first:**
1. **Remove the offending card.** (See §3 for why software workarounds are
   not enough anyway.)
2. If you must coexist temporarily: the throw site is one token. In
   `runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp`, the deprecated-
   doorbell check throws `HSA_STATUS_ERROR`; change it to
   `HSA_STATUS_ERROR_INVALID_ISA` and the caller's existing catch skips the
   node like any other unrecognized GPU. Rather than rebuilding ROCR (a
   from-source rebuild of this library was itself implicated in instability
   for us — build fidelity matters in the runtime that owns the GPU trap
   handler), **binary-patch the vendor's shipped library**: in AMD's
   `libhsa-runtime64.so` the throw compiles to `mov esi, 0x1000` right
   after the `lea` that loads the "deprecated doorbell type" string; flip
   the immediate's low byte `0x00 → 0x0F` (`0x1000 HSA_STATUS_ERROR` →
   `0x100F HSA_STATUS_ERROR_INVALID_ISA`). One byte, vendor binary
   otherwise untouched. (In ROCm 7.2.3's `.so.1.18.70203` that byte sits at
   file offset `0x1ffb0`; locate it by string-xref on other versions.)

---

## 2. Wrong device identity: your GPU is suddenly a different card

**Symptom.** vLLM logs name the wrong GPU (ours claimed to be the display
card), tuned fused-MoE config filenames stop matching
(`device_name=<wrong card>` in the expected filename), and anything that
asks the platform layer for device name / total memory / topology can get
another card's answer.

**Cause.** `amdsmi` enumerates **physical** devices and **ignores
`ROCR_VISIBLE_DEVICES`/`HIP_VISIBLE_DEVICES` entirely.** vLLM's ROCm
platform layer indexes raw amdsmi handles with *logical* device ids in
several places (`get_device_name`, `get_device_total_memory`,
`is_fully_connected`, NUMA queries). Any GPU that HIP hides but amdsmi
lists — a display card first on the bus — shifts every lookup onto the
wrong device.

**Fix.** Route every amdsmi lookup through a handle list filtered to
compute-capable devices (gfx9+), so amdsmi's index space matches what HIP
actually exposes. A few lines in `vllm/platforms/rocm.py`; audit any *new*
amdsmi call site for the same bug. (Homogeneous rigs never notice this —
which is why it survives upstream.)

---

## 3. Phantom OOM: "CUDA error: out of memory" with gigabytes free

The expensive one. Read this before you touch a single memory knob.

**Symptom.** Server dies during **CUDA-graph capture** (`_warmup_and_capture`
/ `capture_model`) with `hipErrorOutOfMemory` — while `rocm-smi` and the
arithmetic say many GiB are free. In the worst version, the failing
operation is a **64-byte `torch.arange`**. Normal (eager) serving of the
same model is perfectly healthy. `expandable_segments` may also throw
`ExpandableSegment` exceptions from `HIPCachingAllocator`.

**First rule: do the arithmetic.** Weights + KV + activations vs. VRAM. If
the "OOM" is arithmetically impossible — a tiny allocation failing with
GiBs free — **it is not a memory problem, and no amount of
`gpu_memory_utilization` / batch-size / allocator tuning will fix it.** We
burned a dozen boots proving that so you don't have to.

**What it actually was.** HIP reports many failures as `hipErrorOutOfMemory`,
and errors can surface at the *next* API call rather than the faulting one
(the runtime even hints: "CUDA kernel errors might be asynchronously
reported at some other API call"). Our real cause: the unsupported display
card's **kernel-side KFD node** — which userspace patching (§1) cannot
reach — poisoned system-scoped **memory-mapping (VMM) paths**: exactly the
substrate under `expandable_segments` and CUDA-graph **memory pools**.
Hence the signature: eager compute fine (plain allocations), graph capture
dead (pool allocations), `expandable_segments:False` sometimes dodging it
(plain `hipMalloc` path). Kernels ≥6.14 are known-hostile to Polaris KFD
(lockups/doorbell resets; AMD closes these as unsupported-kernel), and no
kernel new enough for RDNA2 work is old enough for gfx803.

**Diagnostic ladder** (each step is cheap and discriminates):
1. Arithmetic check (above). Impossible ⇒ stop tuning memory.
2. `CGMODE=NONE` (no capture): serves fine ⇒ the fault lives in the
   capture/VMM path, not the model.
3. Trivial capture in a bare container (`torch.cuda.graph` around `x+1`):
   isolates capture machinery from vLLM.
4. Allocation *inside* the capture region (exercises graph memory pools /
   VMM): the layer that was broken for us.
5. `HIP_LAUNCH_BLOCKING=1` for one boot: if the failing frame doesn't move,
   it's a real allocation-API failure, not an async kernel fault.
6. If any of 2–5 implicate capture/VMM on a mixed-GPU system: **pull the
   unsupported card.** Ours went from ten consecutive capture failures to
   3-second capture success, and full recorded performance, with *zero*
   config changes — the moment the card left.

---

## 4. Cards drop off the PCIe bus under multi-GPU vLLM (but llama.cpp is fine)

**Symptom.** Under flat TP across 4 cards, GPUs die mid-boot or at the first
heavy request: kernel log shows `qcm fence timeout` then `device lost from
bus`; with `amdgpu.gpu_recovery=0` the card is gone until reboot. The same
cards run multi-GPU llama.cpp tensor-split for hours.

**Mechanism (our best-evidenced account).** RCCL collective kernels
**spin-wait on-GPU** while peer ranks stall (cold Triton JIT compiles,
warmup). Prolonged spin-wait wedges the command processor; recovery-off
turns that into a bus drop. llama.cpp is immune because its multi-GPU path
is CPU-coordinated copies — no GPU-side spin-waits. Victims rotate between
runs; narrower (x8) links add marginality.

**Mitigations.**
- Prefer **PP over flat TP** across many cards: PP's sparse send/recv
  doesn't hold spin-wait collectives open. Our 122B config of record is
  PP=3 for this reason; it has run for days.
- Keep a **persistent Triton cache** volume so mid-request JIT stalls (the
  spin-wait trigger) don't recur on every boot.
- If you do run TP, warm everything before real load.

---

## 5. Meta-lessons (the generalizable part)

- **Keep the KFD homogeneous.** Only put GPUs in the machine that your
  ROCm + kernel combination actually supports. "Display-only" is not
  isolation: enumeration (§1), management (§2), and kernel memory paths
  (§3) all see the card.
- **Distrust error strings; trust arithmetic.** `hipErrorOutOfMemory` is a
  catch-all. An impossible OOM is a *different bug wearing an OOM's
  clothes*.
- **Bisect with controls, not theories.** What broke us loose each time was
  a differential pair: TP=2 captures / PP=3 doesn't; trio A / trio B; same
  config cold / warm; patched lib / vendor binary. One variable per boot.
- **sysfs lies about link width** on some boards (claims x16 on x8 slots —
  bridge hop, not end-to-end). Throughput-test your links before making
  topology decisions.
- **Stop containers gracefully** (`docker stop`, never `rm -f`) — TunableOp
  writes its CSV at exit, and SIGKILL eats it.
