// Copyright (C) 2026 Aron Hsiao
// SPDX-License-Identifier: GPL-3.0-or-later
// Part of vllm-rdna2-recipe: vLLM on the Radeon PRO V620 (Navi 21 / gfx1030).
// WS2.1 — push-based one-shot all-reduce for TP=2 on RDNA2 (gfx1030).
//
// Built on the WS2.0 findings (PROFILE §9, T-N3/T-N4):
//   * push, never pull      — peer STORE 14.30 GB/s vs peer LOAD 5.70 GB/s
//   * staging must be UNCACHED (hipDeviceMallocUncached). Coarse-grained memory is invisible
//     to a peer mid-kernel: the write lands in DRAM but the owner reads stale L2 forever.
//   * flags live in host-coherent memory. Device-memory flags cannot be polled across PCIe
//     under any fence or atomic scope.
//
// Graph-capture safety, which is where vLLM's own custom all-reduce dies (T18):
//   * no host round-trip inside the kernel;
//   * the sequence number is derived on-device from a ticket counter, NOT passed as a kernel
//     argument — a captured argument is frozen at capture time, so every replay would re-use
//     the same flag value and the second replay would fall straight through the wait,
//     silently producing wrong results rather than failing;
//   * spins are bounded and set an abort flag rather than hanging the GPU.
//
// Double buffering by sequence parity lets one rank run a full call ahead of the other
// without writing into a buffer the peer is still reducing; the flag wait bounds skew to one.
#pragma once
#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>

// ~1 us per cross-PCIe poll, so this is a ~2 s bound. Sized to surface a failure as an
// abort rather than an apparent hang -- 500M polls would be ~8 minutes of spinning.
#define AR_SPIN_CAP 2000000ull

// host-side conversions + the shared host-coherent flag pair used by the tests
inline float ar_to_f_host(float x)  { return x; }
inline float ar_to_f_host(__half x) { return __half2float(x); }
inline void  ar_from_f_host(float& d, float v)  { d = v; }
inline void  ar_from_f_host(__half& d, float v) { d = __float2half(v); }
static int* g_flags = nullptr;

__device__ __forceinline__ float ar_to_f(float x)  { return x; }
__device__ __forceinline__ float ar_to_f(__half x) { return __half2float(x); }
__device__ __forceinline__ void  ar_from_f(float& d, float v)  { d = v; }
__device__ __forceinline__ void  ar_from_f(__half& d, float v) { d = __float2half(v); }

// One workgroup per block; `nblocks` blocks cooperate through a grid-wide barrier before the
// flag store, so a rank never signals until every one of its blocks has finished pushing.
template <typename T>
__global__ void ar_oneshot(const T* __restrict__ in,
                           T* __restrict__ peer_stage,
                           const T* __restrict__ my_stage,
                           T* __restrict__ out,
                           int* flags,
                           unsigned int* arrive,          // [2], one slot per sequence parity
                           int* seqbuf,                   // local mirror of our flag (cached)
                           int rank, int n, int nblocks, unsigned* timeout)
{
    __shared__ int s_seq;
    __shared__ int s_abort;
    const int t = threadIdx.x, nt = blockDim.x;
    const int b = blockIdx.x;

    // The sequence number is derived from our OWN flag, which advances exactly once per call.
    // Every block reads it before the grid barrier, so all blocks of one launch agree, and the
    // value is independent of nblocks -- a ticket counter divided by nblocks silently breaks
    // the moment the block count changes between calls on a shared counter.
    // It is also read from device memory rather than passed as an argument: a kernel argument
    // is frozen at CUDA-graph capture, so every replay would reuse one value and the second
    // replay would fall straight through the wait.
    if (t == 0) {
        // Read the LOCAL mirror, not the host-resident flag: the flag costs a PCIe round trip
        // on the critical path (~4 us/op measured). Block 0 refreshes the mirror below, after
        // the grid barrier -- and a block only reaches that barrier once it has read here, so
        // every read strictly precedes the write.
        s_seq = __hip_atomic_load(seqbuf, __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT) + 1;
        s_abort = 0;
    }
    __syncthreads();
    const int seq = s_seq;
    const int p = seq & 1;                       // double buffer: tolerates one call of skew
    const long long half = (long long)p * n;

    // 1. push our slice straight into the peer's staging buffer
    const int gid = b * nt + t, gstride = nblocks * nt;
    for (int i = gid; i < n; i += gstride) peer_stage[half + i] = in[i];
    __syncthreads();

    // 2. grid barrier, then payload-before-flag, then announce and wait
    if (t == 0) {
        __threadfence_system();
        atomicAdd(&arrive[p], 1u);
        if (b == 0) {
            unsigned long long s = 0;
            while (__hip_atomic_load(&arrive[p], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_AGENT)
                   < (unsigned)nblocks)
                if (++s > AR_SPIN_CAP) { *timeout = 1u; s_abort = 1; break; }
            if (!s_abort) {
                arrive[1 - p] = 0u;              // ready the slot the next call will use
                __hip_atomic_store(seqbuf, seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_AGENT);
                __hip_atomic_store(&flags[rank], seq, __ATOMIC_RELEASE, __HIP_MEMORY_SCOPE_SYSTEM);
            }
        }
        if (!s_abort) {
            unsigned long long s = 0;
            while (__hip_atomic_load(&flags[1 - rank], __ATOMIC_ACQUIRE, __HIP_MEMORY_SCOPE_SYSTEM) < seq)
                if (++s > AR_SPIN_CAP) { *timeout = 1u; s_abort = 1; break; }
        }
    }
    __syncthreads();
    if (s_abort) return;

    // 3. reduce: our contribution plus what the peer pushed to us, accumulated in fp32
    for (int i = gid; i < n; i += gstride) {
        float v = ar_to_f(in[i]) + ar_to_f(my_stage[half + i]);
        ar_from_f(out[i], v);
    }
}

// Reference path for an honest same-run comparison: stage across PCIe with the runtime's own
// copy engine, then add locally. This is the shape of a naive implementation.
template <typename T>
__global__ void ar_add(const T* __restrict__ a, const T* __restrict__ b,
                       T* __restrict__ out, int n)
{
    for (int i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += gridDim.x * blockDim.x)
        ar_from_f(out[i], ar_to_f(a[i]) + ar_to_f(b[i]));
}
