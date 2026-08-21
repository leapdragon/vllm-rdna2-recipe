// Copyright (C) 2026 Aron Hsiao
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
// Torch bindings for the WS2.1 push-based all-reduce.
//
// Resources that must cross the TP process boundary:
//   * staging buffer (device, uncached) — shared with hipIpcGetMemHandle/OpenMemHandle
//   * flags (host-coherent)             — a named POSIX shm mapping, hipHostRegister'd per rank
#include <torch/extension.h>
#include <ATen/hip/HIPContext.h>
#include <hip/hip_runtime.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <vector>
#include "allreduce.hip.h"

#define TCHK(x) do { hipError_t e = (x); TORCH_CHECK(e == hipSuccess, "HIP: ", hipGetErrorString(e)); } while (0)

namespace {

struct State {
    bool ready = false;
    int  rank = -1;
    int64_t max_elems = 0;      // per dtype-slot; sized in bytes / 2 (fp16 worst case)
    void* stage = nullptr;      // ours (uncached), 2 * max_bytes
    void* peer_stage = nullptr; // peer's, via IPC
    int*  dflags = nullptr;     // device-visible pointer to the shm flags
    void* shm = nullptr;
    unsigned int* arrive = nullptr;      // [2], one slot per sequence parity
    int* seqbuf = nullptr;               // local mirror of our flag
    unsigned* timeout = nullptr;
    int64_t max_bytes = 0;
} g;

} // namespace

// Allocates staging + flags, returns our IPC handle for the caller to broadcast.
static py::bytes ar_init(int64_t rank, int64_t peer_dev, int64_t max_bytes,
                        const std::string& shm_name) {
    TORCH_CHECK(!g.ready, "ar_rdna2 already initialised");
    g.rank = (int)rank;
    g.max_bytes = max_bytes;

    // Peer access must be explicit. hipIpcMemLazyEnablePeerAccess is not a substitute: without
    // this the peer stores go nowhere and both ranks spin on a flag that never advances.
    hipError_t pe = hipDeviceEnablePeerAccess((int)peer_dev, 0);
    TORCH_CHECK(pe == hipSuccess || pe == hipErrorPeerAccessAlreadyEnabled,
                "enable peer access to device ", peer_dev, ": ", hipGetErrorString(pe));

    TCHK(hipExtMallocWithFlags(&g.stage, 2ull * max_bytes, hipDeviceMallocUncached));
    TCHK(hipMemset(g.stage, 0, 2ull * max_bytes));
    TCHK(hipMalloc((void**)&g.arrive, 8)); TCHK(hipMemset(g.arrive, 0, 8));
    TCHK(hipMalloc((void**)&g.seqbuf, 4)); TCHK(hipMemset(g.seqbuf, 0, 4));
    TCHK(hipMalloc((void**)&g.timeout, 4)); TCHK(hipMemset(g.timeout, 0, 4));

    // flags: the same physical host pages in every rank
    // A stale mapping from a previous run would carry non-zero flags and desynchronise the
    // ranks on the very first call, so rank 0 always starts from a fresh object.
    if (rank == 0) shm_unlink(shm_name.c_str());
    int fd = shm_open(shm_name.c_str(), O_CREAT | O_RDWR, 0600);
    TORCH_CHECK(fd >= 0, "shm_open failed: ", strerror(errno));
    TORCH_CHECK(ftruncate(fd, 4096) == 0, "ftruncate failed");
    g.shm = mmap(nullptr, 4096, PROT_READ | PROT_WRITE, MAP_SHARED, fd, 0);
    close(fd);
    TORCH_CHECK(g.shm != MAP_FAILED, "mmap failed");
    if (rank == 0) std::memset(g.shm, 0, 4096);          // rank 0 owns initialisation
    TCHK(hipHostRegister(g.shm, 4096, hipHostRegisterMapped | hipHostRegisterPortable));
    TCHK(hipHostGetDevicePointer((void**)&g.dflags, g.shm, 0));

    hipIpcMemHandle_t h;
    TCHK(hipIpcGetMemHandle(&h, g.stage));
    return py::bytes(reinterpret_cast<const char*>(&h), sizeof(h));
}

static void ar_connect(const std::string& peer_handle) {
    TORCH_CHECK(peer_handle.size() == sizeof(hipIpcMemHandle_t), "bad handle size");
    hipIpcMemHandle_t h;
    std::memcpy(&h, peer_handle.data(), sizeof(h));
    TCHK(hipIpcOpenMemHandle(&g.peer_stage, h, hipIpcMemLazyEnablePeerAccess));
    g.ready = true;
}

static bool ar_can(const torch::Tensor& t) {
    return g.ready && t.is_contiguous() && t.is_cuda()
        && (t.scalar_type() == at::kHalf || t.scalar_type() == at::kFloat)
        && t.numel() * t.element_size() <= g.max_bytes;
}

static torch::Tensor ar_all_reduce(torch::Tensor in) {
    TORCH_CHECK(ar_can(in), "ar_rdna2: unsupported tensor for fast path");
    auto out = torch::empty_like(in);
    const int n = (int)in.numel();
    const int64_t bytes = n * in.element_size();
    // 1 block for decode-sized traffic, 8 above ~16 KB — the sweep is non-monotonic and this
    // is where it crosses over (10 KB: 12.6 us at 1 block vs 17.8 at 8; 40 KB: 38.8 vs 15.0).
    const int nblocks = bytes <= 16384 ? 1 : 8;
    const int threads = 256;
    auto stream = at::cuda::getCurrentCUDAStream();

    if (in.scalar_type() == at::kHalf) {
        hipLaunchKernelGGL(ar_oneshot<__half>, dim3(nblocks), dim3(threads), 0, stream,
                           (const __half*)in.data_ptr(), (__half*)g.peer_stage,
                           (const __half*)g.stage, (__half*)out.data_ptr(),
                           g.dflags, g.arrive, g.seqbuf, g.rank, n, nblocks, g.timeout);
    } else {
        hipLaunchKernelGGL(ar_oneshot<float>, dim3(nblocks), dim3(threads), 0, stream,
                           (const float*)in.data_ptr(), (float*)g.peer_stage,
                           (const float*)g.stage, (float*)out.data_ptr(),
                           g.dflags, g.arrive, g.seqbuf, g.rank, n, nblocks, g.timeout);
    }
    return out;
}

// Bisect the three cross-process resources independently: shm visibility on the CPU, flag
// visibility from the GPU, and whether peer stores into the IPC staging buffer land.
__global__ void probe_write_flag(int* flags, int rank, int val) {
    if (threadIdx.x == 0)
        __hip_atomic_store(&flags[rank], val, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void probe_read_flag(const int* flags, int slot, int* out) {
    if (threadIdx.x == 0)
        *out = __hip_atomic_load(&flags[slot], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM);
}
__global__ void probe_push(int* peer_stage, int val) {
    if (threadIdx.x == 0) peer_stage[0] = val;
}
__global__ void probe_read_stage(const int* stage, int* out) {
    if (threadIdx.x == 0) *out = stage[0];
}

// phase 4: the actual thing the kernel does -- signal, then SPIN for the peer. Single loads
// after a kernel dispatch can succeed where a spin inside one kernel never observes an update.
__global__ void probe_handshake(int* flags, int rank, int seq, unsigned* to) {
    if (threadIdx.x != 0) return;
    __threadfence_system();
    __hip_atomic_store(&flags[rank], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
    unsigned long long s = 0;
    while (__hip_atomic_load(&flags[1 - rank], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq)
        if (++s > AR_SPIN_CAP) { *to = 1u; return; }
}

static std::vector<int64_t> ar_probe(int64_t phase, int64_t val) {
    int* dout; TCHK(hipMalloc((void**)&dout, 4));
    int host = -1;
    auto stream = at::cuda::getCurrentCUDAStream();
    if (phase == 0) {                       // write our flag from the GPU
        hipLaunchKernelGGL(probe_write_flag, dim3(1), dim3(64), 0, stream, g.dflags, g.rank, (int)val);
        TCHK(hipStreamSynchronize(stream));
    } else if (phase == 1) {                // read the PEER's flag from our GPU
        hipLaunchKernelGGL(probe_read_flag, dim3(1), dim3(64), 0, stream, g.dflags, 1 - g.rank, dout);
        TCHK(hipStreamSynchronize(stream));
        TCHK(hipMemcpy(&host, dout, 4, hipMemcpyDeviceToHost));
    } else if (phase == 2) {                // push one dword into the peer's staging buffer
        hipLaunchKernelGGL(probe_push, dim3(1), dim3(64), 0, stream, (int*)g.peer_stage, (int)val);
        TCHK(hipStreamSynchronize(stream));
    } else if (phase == 4) {                // signal + spin, cross-process
        TCHK(hipMemset(g.timeout, 0, 4));
        hipLaunchKernelGGL(probe_handshake, dim3(1), dim3(64), 0, stream,
                           g.dflags, g.rank, (int)val, g.timeout);
        TCHK(hipStreamSynchronize(stream));
        unsigned t = 0; TCHK(hipMemcpy(&t, g.timeout, 4, hipMemcpyDeviceToHost));
        host = t ? -999 : 0;                // -999 = the spin never saw the peer
    } else {                                // read our own staging buffer
        hipLaunchKernelGGL(probe_read_stage, dim3(1), dim3(64), 0, stream, (const int*)g.stage, dout);
        TCHK(hipStreamSynchronize(stream));
        TCHK(hipMemcpy(&host, dout, 4, hipMemcpyDeviceToHost));
    }
    TCHK(hipFree(dout));
    // also report what the CPU sees in the shm flags, to separate "not written" from "not visible"
    int* cpu = (int*)g.shm;
    return {host, cpu[0], cpu[1]};
}

static bool ar_timed_out() {
    if (!g.ready) return false;
    unsigned t = 0;
    TCHK(hipMemcpy(&t, g.timeout, 4, hipMemcpyDeviceToHost));
    return t != 0;
}

PYBIND11_MODULE(TORCH_EXTENSION_NAME, m) {
    m.def("init", &ar_init, "allocate staging+flags, return our IPC handle");
    m.def("spin_cap", []{ return (double)AR_SPIN_CAP; }, "poll bound before abort");
    m.def("connect", &ar_connect, "open the peer's staging buffer");
    m.def("can", &ar_can, "is this tensor eligible for the fast path");
    m.def("all_reduce", &ar_all_reduce, "push-based one-shot all-reduce");
    m.def("timed_out", &ar_timed_out, "did any spin hit its bound");
    m.def("probe", &ar_probe, "bisect cross-process visibility");
    m.def("clear_timeout", []{ if (g.ready) TCHK(hipMemset(g.timeout, 0, 4)); },
          "clear the sticky abort flag so one failure does not mask later results");
}
