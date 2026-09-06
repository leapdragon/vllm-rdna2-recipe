# TROUBLESHOOTING.md — vLLM on Radeon PRO V620 (gfx1030): symptom → cause → fix

**Copyright © 2026 Aron Hsiao. Licensed under the GNU General Public License v3.0 or later.**

Symptom-first. Every entry has the same shape — **Symptom / Cause / Fix / Verify** — so you (or
your assistant) can match what you see, apply the fix, and confirm it took. Most reports end in
§0 or §1. Sections are numbered by problem area; the old section numbers are kept in the headings.

| You see… | Go to |
|---|---|
| Large prompts fail, the server misbehaves or hangs, short-prompt decode is fine | §1.1 |
| Decode is a flat ~30 ms/token at every context length; prefill fine; outputs correct | §1.2 |
| Boot dies ~2 min in with `MEMORY_APERTURE_VIOLATION` after changing which GPUs you serve on | §1.3 |
| Your change "did nothing"; the kernel is correct; the profile shows no new launches | §1.4 |
| Collectives or prefill slow after copying someone else's `NCCL_P2P_LEVEL` | §1.5 |
| `qcm fence timeout` → `device lost from bus` under tensor parallelism | §2.1 |
| GPU lost after a stream wait; `unsuccessful queues preemption` in the kernel log | §2.2 |
| `rocminfo` / vLLM see zero GPUs, `rocm-smi` sees them all | §3.1 |
| vLLM names the wrong GPU; tuned config filenames name the wrong card | §3.2 |
| `hipErrorOutOfMemory` during CUDA-graph capture with gigabytes free | §3.3 |
| Build/install errors (CUDA torch from PyPI, Triton cmake, `libcuda.so.1`, peer access) | §4 |
| Numbers that look too good or too bad to be true | §5 |

---

## 0. First things to try

1. **`VLLM_USE_V2_MODEL_RUNNER=0`** if large prompts fail or the server misbehaves (§1.1). The
   most likely fix, costs nothing, reported as curing "all my problems".
2. **Check that the TunableOp rows loaded** if decode is slow but correct (§1.2): ~40 % of decode
   rides on them and they fail silently.
3. **Scope or wipe the compile cache** when you change `DEVICES` (§1.3).
4. **Read four lines of the serve log** before assuming anything else: which model runner
   (`Using V1 Model Runner` / `Using V2 Model Runner`), the TunableOp CSV read,
   `fd_rdna2: flash-decode override active`, and the patch markers. The full arrival checklist is
   [02-VERSIONS.md → Verifying you arrived](02-VERSIONS.md).
5. **Stop containers with `docker stop`, never `rm -f`.** TunableOp writes its CSV at exit; SIGKILL
   eats it.
6. **Never mount a host ROCm into the container.** The image carries its own; a host ROCm on top
   is the most common way to break it.

---

## 1. Runtime configuration (env, caches, tuned rows)

### 1.1 Large prompts fail or the server misbehaves — use the V1 model runner (formerly §5d)

**Symptom.** Short-prompt decode works; large prompts fail or hang, or the server misbehaves.

**Cause.** vLLM's V2 model runner. This recipe's vLLM (0.27.1) selects V1 by default for the hybrid
Qwen3-Next architecture; V2 is *required* only for MTP under pipeline parallelism (patch 0009, the
122B build), and the 122B scripts and the 02-VERSIONS env table set it for that reason. Anyone who
copied `VLLM_USE_V2_MODEL_RUNNER=1` onto another configuration is on V2 without needing it.

**Fix.** `VLLM_USE_V2_MODEL_RUNNER=0` (or leave it unset) on every configuration that is not the
122B under PP. Reported by **CorbinD** (gfx1030 club Discord, 2026-09-06), running the recipe with
the int8-KV flash-decode plugin and the custom all-reduce on 4× V620: it "fixed all my problems",
large prompts included, at a stable 68 tok/s decode.

**Verify.** The serve log prints `Using V1 Model Runner`. Not reproduced by the maintainers: if you
hit the failure on V2, the exact error text and the prompt length are the missing data — please
report them.

### 1.2 Decode at 55–65 % of the recorded numbers, context-flat; prefill and correctness fine (formerly §5c)

**Symptom.** Decode is a flat ~30 ms/token at every context length instead of ~20–24. Prefill is
normal, greedy outputs are byte-identical, MTP acceptance is healthy, nothing in the logs complains.

**Cause.** The TunableOp results CSV is missing, or its `Validator` lines no longer match the stack
(torch / ROCm / rocBLAS / arch). `PYTORCH_TUNABLEOP_ENABLED=1` with `TUNING=0` then falls back
silently to rocBLAS's heuristic, which picks a large-M tile for the fp16 lm_head's skinny decode
GEMMs (~115 GB/s instead of the tuned 360–430). MTP pays the head three times per step, so the
loss is ~40 % of decode.

**Fix.** Copy the shipped per-build CSVs from `builds/<model>/tunableop/` into the live
`tunableop/` directory (the container entrypoint seeds `/tuning` from them when it is empty), or
retune offline per the build's BUILD.md. The live directory is gitignored — treat the CSVs like
weights, not scratch.

**Verify.** `grep tn_124160 tunableop/tunableop_results0.csv` (27B; the 122B head rows are
`tn_151936_*`). No rows, no file, or mismatched `Validator` stamps → this is the problem.

### 1.3 `MEMORY_APERTURE_VIOLATION` on boot after changing which GPUs you serve on (formerly §5b)

**Symptom.** The engine dies ~2 min into boot with `HSA_STATUS_ERROR_MEMORY_APERTURE_VIOLATION`;
the kernel log shows a `[gfxhub] page fault` (client UTCL2) on the rank-0 card at user-space
addresses, right as `Directly load AOT compilation from path /compile-cache/...` appears. No bus
drop; the cards are healthy.

**Cause.** The torch.compile/AOT cache is device-set-specific in practice. A cache populated on one
pair of cards (`DEVICES=1,3`) crashes the worker when its artifacts load on another (`2,4`);
first seen on the MTP drafter's `eagle_head` artifacts.

**Fix.** Never share a compile cache across device sets. `config/serve-rdna2-tp2.sh` scopes it as
`$STATE_DIR/compile-cache-<devices>`; if you mount `VLLM_CACHE_ROOT` yourself, key the directory by
the device list or wipe it when `DEVICES` changes. The container writes as root:
`docker run --rm -v <dir>:/wipe --entrypoint sh <image> -c 'rm -rf /wipe/*'`.

**Verify.** A cold boot on the new device set (~8 min longer) completes; the AOT-load lines appear
without the page fault.

### 1.4 A change "did nothing" although the kernel is correct

**Symptom.** You edited a kernel or a dispatch condition; the profile shows the same launches as
before.

**Cause, two variants.** (a) vLLM traces the model once for a dynamic token range: a Python
`if 0 < n <= 8:` inside the traced region is decided on the tracing example and baked into the
graph. (b) The torch.compile cache key does not cover edits to vLLM's own Python, so a changed
hook is served from a stale compiled graph.

**Fix.** Put decode/prefill choices inside an opaque custom op with a fake impl, not in traced
Python. While iterating, boot with `VLLM_DISABLE_COMPILE_CACHE=1`.

**Verify.** Count kernel *launches* in a profile after every change, not wall time.

### 1.5 `NCCL_P2P_LEVEL` — do not copy it from another host

**Symptom.** Collectives or prefill slower than the recorded numbers after adopting a pinned
`NCCL_P2P_LEVEL`.

**Cause.** The right RCCL peer-to-peer level depends on the PCIe topology. On a host with one V620
per switch (every pair PHB-class), `PXB` matched no pair and silently disabled peer access:
`rccl-tests` all_reduce 8.0 GB/s vs 21.0 unset (Waldecir Santos,
[PR #2](https://github.com/leapdragon/vllm-rdna2-recipe/pull/2)). On the 2+2 host this recipe
was developed on, `SYS` gave +8 % prefill over `PXB`.

**Fix.** Leave it unset (RCCL autodetects) unless `rocm-smi --showtopo` tells you otherwise. The
recipe no longer pins it (2026-09-06).

**Verify.** `rccl-tests/all_reduce_perf` on 8–256 MB messages, unset vs your candidate.

---

## 2. Multi-GPU stability (cards dropping off the bus)

### 2.1 Cards drop off the PCIe bus under tensor parallelism; llama.cpp is fine (formerly §4)

**Symptom.** Under TP across the cards, GPUs die mid-boot or at the first heavy request: kernel
log `qcm fence timeout` then `device lost from bus`; with `amdgpu.gpu_recovery=0` the card is gone
until reboot. The same cards run multi-GPU llama.cpp tensor-split for hours.

**Cause (best-evidenced).** GPU-resident, tightly synchronized per-layer collective kernels wedge
the command processor; recovery-off turns that into a bus drop. It is the *cadence*, not the
transport: the cleanest incident killed exactly one TP pair with P2P disabled and every byte
host-mediated, while the other pair (pipeline stage 2, identical machinery) survived. llama.cpp is
immune because its multi-GPU path is CPU-coordinated copies with no lockstep communication
kernels.

**Fix.** Flat TP=4 ran ~2.5 h sustained with zero events, at +43–124 % over PP, with the whole
platform-stability stack in place ([02-VERSIONS.md → Platform-stability stack](02-VERSIONS.md)):
kernel line `amdgpu.pcie_gen_cap=0x00070007 amdgpu.aspm=0 amdgpu.runpm=0 amdgpu.gpu_recovery=1`,
`HSA_NO_SCRATCH_RECLAIM=1`, `--max-num-batched-tokens 2048`, moderate power caps. Which subset is
load-bearing is not isolated — apply all of it. Without it every TP attempt lost cards, P2P on or
off. **Pipeline parallelism is the zero-kludge fallback**: sparse host-paced send/recv never
holds collective kernels open; the PP=3 config ran for days with no special measures. Keep a
persistent Triton cache volume for boot time (~480 s of one-time compile per topology); a
cold-cache control run survived, so it is convenience, not stability. Warm everything before
real load.

**Verify.** `journalctl -k | grep amdgpu` shows nothing new after a sustained run;
`verify/soak.py` for the long form.

### 2.2 A pending GPU stream wait can take the card off the bus

**Symptom.** GPU reset or `device lost from bus`; the kernel log shows `unsuccessful queues
preemption`, often near `svm_range_restore` activity (big mmaps, pinned buffers).

**Cause.** KFD's queue eviction cannot preempt a queue parked in `WAIT_REG_MEM`. Anything that
leaves a `hipStreamWaitValue32` pending on a queue — a GPU-side wait on a CPU producer — is a
reset waiting to happen. Also: `hipStreamWaitValue32` returns success during stream capture, but
the HIP graph replays *without* it, so a captured wait silently reads stale data.

**Fix.** Synchronize GPU work against CPU producers on the host, never with a GPU-side stream
wait; test any such primitive standalone before trusting it.

### 2.3 Do not poll host memory from GPU kernels across PCIe

**Symptom.** Two cards on different root complexes drop together under load; other PCIe devices
(an HBA, a tape drive) re-initialize at the same moment.

**Cause.** GPU kernels spinning on system-memory reads through both root complexes saturate the
fabric. Posted writes are ordered per destination only, so a payload to a peer's VRAM and a flag in
host memory can also arrive out of order.

**Fix.** Put barrier flags in each rank's own uncached VRAM and poll locally.

### 2.4 `Runlist is getting oversubscribed` in the kernel log during graph capture

**Cause.** A side job on a serving card. ROCR device numbering is not `/dev/dri/cardN` numbering.

**Fix.** Put harnesses and tests on a card the server does not use; check with `rocm-smi --showpids`.

---

## 3. Mixed or unsupported GPUs in the machine

The trigger for all three entries was one **pre-Vega (gfx803, Polaris) display card** alongside
four supported RDNA2 cards. **Do not put pre-Vega silicon in a ROCm compute machine, even
display-only.** ROCm userspace and the kernel KFD both assume it isn't there, and neither isolates
it: enumeration (§3.1), management (§3.2) and kernel memory paths (§3.3) all see it. The
[gfx1030 wiki](https://blivioniag.github.io/gfx1030-wiki/) reached the same rule independently.

### 3.1 "No ROCm-capable devices" — but every card is healthy (formerly §1)

**Symptom.** `rocminfo`, llama.cpp, vLLM, PyTorch all report zero GPUs (or
`HSA_STATUS_ERROR: A generic error has occurred`); `rocm-smi` and the amdgpu driver see every card.

**Cause.** ROCR-Runtime device discovery is all-or-nothing: an unhandled property on *any* KFD node
(here a pre-Vega doorbell type) throws `HSA_STATUS_ERROR` and aborts enumeration of every GPU.
`ROCR_VISIBLE_DEVICES` is applied after the crash point and cannot help; masking the sysfs node
fails too (the runtime cross-checks KFD ioctls).

**Fix.** Remove the card. If you must coexist temporarily: in
`runtime/hsa-runtime/core/runtime/amd_gpu_agent.cpp` the deprecated-doorbell check throws
`HSA_STATUS_ERROR`; make it `HSA_STATUS_ERROR_INVALID_ISA` and the caller skips the node like any
unrecognized GPU. Rather than rebuilding ROCR (a from-source ROCR was itself implicated in
instability — build fidelity matters in the library that owns the GPU trap handler), binary-patch
the vendor library: the throw compiles to `mov esi, 0x1000` right after the `lea` of the
"deprecated doorbell type" string; flip the immediate's low byte `0x00 → 0x0F`
(`HSA_STATUS_ERROR` → `HSA_STATUS_ERROR_INVALID_ISA`). In ROCm 7.2.3's `libhsa-runtime64.so.1.18.70203`
that byte is at file offset `0x1ffb0`; locate it by string xref on other versions.

**Verify.** `rocminfo` lists the compute cards; the display card is skipped, not fatal.

### 3.2 Wrong device identity (formerly §2)

**Symptom.** vLLM logs name the wrong GPU; tuned fused-MoE config filenames stop matching
(`device_name=<wrong card>`); device name, total memory or topology queries return another card's
answer.

**Cause.** `amdsmi` enumerates *physical* devices and ignores `ROCR_VISIBLE_DEVICES` /
`HIP_VISIBLE_DEVICES`. vLLM's ROCm platform layer indexes amdsmi handles with *logical* ids in
several places (`get_device_name`, `get_device_total_memory`, `is_fully_connected`, NUMA). Any GPU
that HIP hides but amdsmi lists — a display card first on the bus — shifts every lookup.

**Fix.** Route every amdsmi lookup through a handle list filtered to compute-capable devices (gfx9+)
so amdsmi's index space matches what HIP exposes (a few lines in `vllm/platforms/rocm.py`); audit
new amdsmi call sites for the same bug. Homogeneous rigs never notice, which is why it survives
upstream.

**Verify.** The boot log's device name matches `rocm-smi --showproductname` for the serving cards.

### 3.3 Phantom OOM: `hipErrorOutOfMemory` during graph capture with gigabytes free (formerly §3)

**Symptom.** The server dies in CUDA-graph capture (`capture_model`) with `hipErrorOutOfMemory`
while `rocm-smi` and arithmetic say many GiB are free — in the worst case on a 64-byte
`torch.arange`. Eager serving of the same model is healthy. `expandable_segments` may throw
`ExpandableSegment` exceptions.

**Cause.** HIP reports many failures as `hipErrorOutOfMemory`, and errors surface at the *next* API
call. Here the unsupported card's kernel-side KFD node — unreachable by the userspace patch in
§3.1 — poisoned system-scoped memory-mapping (VMM) paths, the substrate under `expandable_segments`
and graph memory pools. Hence the signature: plain allocations fine, pool allocations dead. Kernels
≥ 6.14 are hostile to Polaris KFD, and no kernel new enough for RDNA2 is old enough for gfx803.

**Fix / diagnostic ladder** (each step cheap, each discriminates):
1. Do the arithmetic (weights + KV + activations vs VRAM). An impossible OOM is not a memory
   problem; no `gpu_memory_utilization`, batch-size or allocator knob will fix it.
2. `CGMODE=NONE` (no capture) serves fine ⇒ the fault is in the capture/VMM path.
3. A trivial `torch.cuda.graph` capture around `x+1` in a bare container isolates the capture
   machinery from vLLM; an allocation *inside* the capture region exercises the pools.
4. `HIP_LAUNCH_BLOCKING=1` for one boot: if the failing frame doesn't move, it is a real
   allocation-API failure, not an async kernel fault.
5. If 2–4 implicate capture/VMM on a mixed-GPU system: **pull the unsupported card.** Ten
   consecutive capture failures became a 3-second capture with zero config changes.

**Related, no mixed GPUs needed:** large pageable host-memory transfers on multi-GPU can fault with
`illegal memory access ... current device: -1`, surfacing at a later `hipHostFree` or teardown
([ROCm/rocm-systems#4817](https://github.com/ROCm/rocm-systems/issues/4817)). Workaround from
[edwinbrowwn/llama.cpp-rdna2](https://github.com/edwinbrowwn/llama.cpp-rdna2): `hipHostRegister`
the buffer around the transfer.

---

## 4. Building and installing on the host (the container avoids all of these)

| Symptom | Cause | Fix |
|---|---|---|
| `pip install -r requirements/…` pulls the CUDA torch from PyPI over the torch you built | compressed-tensors, xgrammar depend on torch | Install your own wheels first; pin them |
| Triton cmake: `imported targets are referenced, but are missing: LLVMNVPTX…` | `/opt/rocm` on `CMAKE_PREFIX_PATH` makes `find_package(LLVM)` pick ROCm's LLVM (no NVPTX) while MLIR comes from Triton's tarball | Keep ROCm out of Triton's configure |
| `Failed to dlopen libcuda.so.1` from code that "tries cuda-python, else HIP" | a full venv carries `cuda-bindings` transitively, so importability picks cuda-python | Select by platform (`torch.version.hip`), not by importability |
| `No module named 'torchvision'` for text-only serving | `transformers`' vision processors hard-import it when the checkpoint carries a vision config | Build torchvision (CPU ops) too |
| `hipErrorPeerAccessAlreadyEnabled`, then torch throws "peer access is already enabled" | the error is sticky after a second `hipDeviceEnablePeerAccess` | `hipGetLastError()` to clear |
| A port compiles, launches, and returns garbage (relerr 0.6–0.9 on every shape) | upstream's `__HIP__GFX1X__` means gfx11/12, not gfx10 | Validate numerically before trusting a "just widen the macro" port |
| One card loads weights, three sit at 1 % VRAM forever | a collective's init raised on one rank inside an ordered barrier loop; that rank moved on, the others wait | Never keep singleton state in a communicator; never let an init path skip a barrier the peers will hit |

---

## 5. Measuring correctly

- **Prefill numbers with prefix caching on are fiction.** A repeated 44.5k prompt "prefilled" at
  5,640 tok/s vs 537 fresh. Use unique prompts and `max_tokens=1` (`verify/prefill-rate.py`).
- **The first pass on an idle card reads 5–10× slow** (clock ramp), and weights under 16 MB sit in
  the Infinity Cache in a timing loop and read as 800 GB/s. Warm the card; believe the
  DRAM-streaming number.
- **Distrust error strings; trust arithmetic.** `hipErrorOutOfMemory` is a catch-all (§3.3).
- **One variable per boot.** What broke every case here loose was a differential pair: TP=2
  captures / PP=3 doesn't; pair A / pair B; cold / warm; patched library / vendor binary.
- **sysfs lies about link width** on some boards (x16 claimed on x8 slots — a bridge hop, not
  end-to-end). Throughput-test links before topology decisions.
- **Correctness after any kernel or communication change:** `verify/validate.py` (greedy
  answers), `verify/fd_gqa_test.py` (the flash-decode plugin), and two identical greedy runs must
  match byte for byte.

---

## 6. Where else to look

- [Wiki GFX1030](https://blivioniag.github.io/gfx1030-wiki/) — power tuning, PCIe P2P readiness
  checks, env-var cheat sheets, its own troubleshooting pages; its `v620_toolbox` unlocks a 120 W
  power floor on cards that otherwise refuse anything under 250 W (relevant to §2.1's power
  transients).
- [RDNA2-RESOURCES.md](RDNA2-RESOURCES.md) — forks, images and toolboxes worth knowing.
- [leapdragon/vllm-rdna2-qwen](https://github.com/leapdragon/vllm-rdna2-qwen) — the Flash-Next
  successor fork; its `docs/rdna2/TROUBLESHOOTING.md` covers that model's own machinery (n-gram
  sidecar, tuned MoE configs, TunableOp rows per rocBLAS build), several of which generalize.
