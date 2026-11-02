// SPDX-License-Identifier: AGPL-3.0-only
// Copyright 2026-present the Unsloth AI Inc. team. All rights reserved.

#include "moe-cache.cuh"

// device-side lookup of the MoE expert cache, see ggml_moe_cache_lookup and llama-moe-cache.cpp
//
// state: [0] CLOCK hand, [1] epoch (one per lookup), [2..3] hits, [4..5] misses, [6..7] unused, then key_of, ref and pin [n_slots]
// key_of: il*n_expert + expert (-1 if empty); ref: CLOCK reference bits; pin: epoch of the last lookup using the slot (that lookup does not evict it)
// dst:   slots[n_ids], then a row starting with the number of copied experts, then their experts[n_ids] and slots[n_ids]

#define MOE_CACHE_HDR   8
#define MOE_CACHE_LANES 32

// the lookup runs in a single block of 32 threads, so a block barrier is a barrier of the warp (also with 64-wide wavefronts)
static __device__ __forceinline__ unsigned moe_cache_ballot(const bool p) {
#if defined(GGML_USE_HIP)
    return (unsigned) __ballot(p);
#else
    return __ballot_sync(0xffffffffu, p);
#endif
}

// ids out of range (never produced by the router) are read as expert 0
static __device__ __forceinline__ int32_t moe_cache_id(const int32_t * ids, const int i, const int n_expert) {
    const int32_t e = ids[i];
    return (uint32_t) e < (uint32_t) n_expert ? e : 0;
}

static __global__ void moe_cache_assign(
        const int32_t * __restrict__ ids, int32_t * __restrict__ map, int32_t * __restrict__ state, int32_t * __restrict__ dst,
        const int n_ids, const int ne0, const int n_expert, const int il, const int n_slots, const int tail_rank, const int split) {
    extern __shared__ int32_t smem[];
    int32_t * miss_e = smem;         // [n_ids] missing experts, in the order of their first use
    int32_t * victim = smem + n_ids; // [n_ids] slots given to them

    const int      lane = threadIdx.x;
    const unsigned full = 0xffffffffu;

    int32_t * key_of = state + MOE_CACHE_HDR;
    int32_t * ref    = key_of + n_slots;
    int32_t * pin    = ref + n_slots;
    int32_t * row    = map + (int64_t) il*n_expert;

    int32_t * slots  = dst;             // split: the resident experts' slots, the others the zero expert
    int32_t * miss   = dst + n_ids;     // split: the missing experts' slots, the others the zero expert
    int32_t * n_fill = dst + 2*n_ids;
    int32_t * fill_e = dst + 2*n_ids + ne0;
    int32_t * fill_s = fill_e + n_ids;

    uint32_t epoch = (uint32_t) state[1] + 1;
    if (epoch == 0) {
        // wrapped around: no slot may look pinned by this lookup
        for (int s = lane; s < n_slots; s += MOE_CACHE_LANES) {
            pin[s] = 0;
        }
        epoch = 1;
    }
    __syncthreads();
    if (lane == 0) {
        state[1] = (int32_t) epoch;
    }

    // hits: set the reference bit and pin the slot for this lookup
    int n_hit = 0;
    for (int i = lane; i < n_ids; i += MOE_CACHE_LANES) {
        const int32_t s = row[moe_cache_id(ids, i, n_expert)];
        slots[i] = s;
        if (split) {
            miss[i] = n_slots + i % ne0; // the zero expert of this rank
        }
        if (s >= 0) {
            ref[s] = 1;
            if ((uint32_t) atomicExch(&pin[s], (int32_t) epoch) != epoch) {
                n_hit++;
            }
        }
    }
    __syncthreads();

    // tail skip: a missing expert that some token routes at a rank below tail_rank (ids are sorted by descending weight,
    // the rank is the position within the token) must be fetched, mark it -3; a missing expert left at -1 is skipped
    if (tail_rank > 0) {
        for (int i = lane; i < n_ids; i += MOE_CACHE_LANES) {
            if (slots[i] < 0 && i % ne0 < tail_rank) {
                row[moe_cache_id(ids, i, n_expert)] = -3;
            }
        }
        __syncthreads();
    }

    // misses: the first use of each missing expert claims it, in the order of the ids
    int n_miss = 0;
    int n_skip = 0;
    for (int i0 = 0; i0 < n_ids; i0 += MOE_CACHE_LANES) {
        const int i = i0 + lane;
        int32_t e = i < n_ids && slots[i] < 0 ? moe_cache_id(ids, i, n_expert) : -1;
        if (tail_rank > 0 && e >= 0 && row[e] == -1) {
            // skipped: the zero expert of this rank in the extra slots of the banks, nothing is copied (miss[i] is a zero expert already)
            slots[i] = n_slots + i % ne0;
            n_skip++;
            e = -1;
        }
        bool first = e >= 0;
        for (int k = 0; k < MOE_CACHE_LANES; ++k) {
            const int32_t ek = __shfl_sync(full, e, k, MOE_CACHE_LANES);
            if (k < lane && ek == e) {
                first = false;
            }
        }
        // a previous group of ids may have claimed it already
        first = first && (row[e] == -1 || row[e] == -3);
        const unsigned m = moe_cache_ballot(first);
        if (first) {
            row[e] = -2;
            miss_e[n_miss + __popc(m & ((1u << lane) - 1))] = e;
        }
        n_miss += __popc(m);
        __syncthreads();
    }

    // victims: CLOCK sweep, 32 slots at a time, the slots used by this lookup are skipped
    // there are enough unpinned slots because the number of distinct ids is at most the number of slots
    int h = state[0];
    const int width = min(MOE_CACHE_LANES, n_slots);
    int found = 0;
    while (found < n_miss) {
        const bool active = lane < width;
        const int  s      = active ? (h + lane < n_slots ? h + lane : h + lane - n_slots) : 0;
        const bool pinned = active && (uint32_t) pin[s] == epoch;
        const bool r      = active && ref[s] != 0;
        const bool elig   = active && !pinned && !r;

        const unsigned m    = moe_cache_ballot(elig);
        const int      need = n_miss - found;
        int      adv  = width;
        unsigned take = m;
        if (__popc(m) >= need) {
            unsigned mm = m;
            for (int k = 0; k < need - 1; ++k) {
                mm &= mm - 1;
            }
            const int last = __ffs(mm) - 1;
            adv  = last + 1;
            take = last == 31 ? m : m & ((2u << last) - 1);
        }
        if (lane < adv && r && !pinned) {
            ref[s] = 0;
        }
        if (take & (1u << lane)) {
            victim[found + __popc(take & ((1u << lane) - 1))] = s;
            pin[s] = (int32_t) epoch;
        }
        found += __popc(take);
        h += adv;
        if (h >= n_slots) {
            h -= n_slots;
        }
        __syncthreads();
    }

    // evict and insert
    for (int j = lane; j < n_miss; j += MOE_CACHE_LANES) {
        const int32_t s   = victim[j];
        const int32_t e   = miss_e[j];
        const int32_t old = key_of[s];
        if (old >= 0) {
            map[old] = -1;
        }
        key_of[s] = il*n_expert + e;
        ref[s]    = 1;
        row[e]    = s;
        fill_e[j] = e;
        fill_s[j] = s;
    }
    __syncthreads();

    for (int i = lane; i < n_ids; i += MOE_CACHE_LANES) {
        if (slots[i] < 0) {
            const int32_t s = row[moe_cache_id(ids, i, n_expert)];
            if (split) {
                miss[i]  = s;
                slots[i] = n_slots + i % ne0;
            } else {
                slots[i] = s;
            }
        }
    }

    for (int off = MOE_CACHE_LANES/2; off > 0; off >>= 1) {
        n_hit  += __shfl_xor_sync(full, n_hit,  off, MOE_CACHE_LANES);
        n_skip += __shfl_xor_sync(full, n_skip, off, MOE_CACHE_LANES);
    }
    if (lane == 0) {
        state[0] = h;
        *n_fill  = n_miss;
        *(unsigned long long *) (state + 2) += (unsigned long long) n_hit;
        *(unsigned long long *) (state + 4) += (unsigned long long) n_miss;
        *(unsigned long long *) (state + 6) += (unsigned long long) n_skip;
    }
}

struct moe_cache_fetch_args {
    const char * src[4];
    char       * dst[4];
    int64_t      nb[4];
};

// copy the missing experts from (mapped) host memory to their slots
// work items are (missing expert, bank, chunk); the grid is fixed because the launch is captured in a CUDA graph, so the
// blocks loop over the items: a lookup without misses costs one small launch instead of n_ids*n_banks*n_chunks idle blocks
static __global__ void moe_cache_fetch(const int32_t * __restrict__ info, const int n_ids, const int ne0, const moe_cache_fetch_args args,
        const int64_t chunk, const int n_banks, const int n_chunks) {
    const int32_t * n_fill = info + 2*n_ids;
    const int32_t * fill_e = info + 2*n_ids + ne0;
    const int32_t * fill_s = fill_e + n_ids;

    const int n_work = (*n_fill)*n_banks*n_chunks;
    for (int w = blockIdx.x; w < n_work; w += gridDim.x) {
    const int     j   = w / (n_banks*n_chunks);
    const int     r   = w - j*(n_banks*n_chunks);
    const int     b   = r / n_chunks;
    const int64_t nb  = args.nb[b];
    const int64_t beg = (int64_t) (r - b*n_chunks)*chunk;
    if (beg >= nb) {
        continue;
    }
    const int64_t end = min(beg + chunk, nb);

    const char * src = args.src[b] + fill_e[j]*nb;
    char       * dst = args.dst[b] + fill_s[j]*nb;

    if ((((uintptr_t) (src + beg)) | ((uintptr_t) (dst + beg)) | (uintptr_t) (end - beg)) % 16 == 0) {
        const int4 * s4 = (const int4 *) (src + beg);
        int4       * d4 = (int4 *) (dst + beg);
        const int64_t n = (end - beg)/16;
        constexpr int unroll = 4;
        int64_t i = threadIdx.x;
        for (; i + (unroll - 1)*blockDim.x < n; i += unroll*blockDim.x) {
            int4 v[unroll];
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                v[u] = s4[i + u*blockDim.x];
            }
#pragma unroll
            for (int u = 0; u < unroll; ++u) {
                d4[i + u*blockDim.x] = v[u];
            }
        }
        for (; i < n; i += blockDim.x) {
            d4[i] = s4[i];
        }
    } else {
        for (int64_t i = beg + threadIdx.x; i < end; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
    }
}

bool ggml_cuda_moe_cache_lookup_supported(const ggml_tensor * op) {
    if (ggml_get_op_params_i32(op, 0) < 0) {
        return op->src[0] != nullptr && op->src[0]->op == GGML_OP_MOE_CACHE_LOOKUP; // the join
    }
    const int n_banks = ggml_get_op_params_i32(op, 1);
    if (n_banks < 1 || n_banks > 4) {
        return false;
    }
    const int64_t n_ids = ggml_nelements(op->src[0]);
    return op->src[0]->type == GGML_TYPE_I32 && op->src[1]->type == GGML_TYPE_I32 && op->src[2]->type == GGML_TYPE_I32 &&
        n_ids <= 4096;
}

// the side stream of a split lookup and the events that fork it off the main stream and join it back
static constexpr int MOE_CACHE_SIDE_STREAM = GGML_CUDA_MAX_STREAMS - 1;

static cudaEvent_t moe_cache_event(int device, int which) {
    static cudaEvent_t events[GGML_CUDA_MAX_DEVICES][2] = {};
    if (events[device][which] == nullptr) {
        CUDA_CHECK(cudaEventCreateWithFlags(&events[device][which], cudaEventDisableTiming));
    }
    return events[device][which];
}

void ggml_cuda_op_moe_cache_lookup(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    if (ggml_get_op_params_i32(dst, 0) < 0) {
        // join: the main stream waits for the copy of the missing experts
        if (ctx.moe_fork_open) {
            CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(), moe_cache_event(ctx.device, 1), 0));
            ctx.moe_fork_open = false;
        }
        return;
    }

    const ggml_tensor * ids   = dst->src[0];
    ggml_tensor       * map   = dst->src[1];
    ggml_tensor       * state = dst->src[2];

    const int il        = ggml_get_op_params_i32(dst, 0);
    const int n_banks   = ggml_get_op_params_i32(dst, 1);
    const int tail_rank = ggml_get_op_params_i32(dst, 2 + 2*4);
    const int split     = ggml_get_op_params_i32(dst, 2 + 2*4 + 1);

    const int n_ids    = (int) ggml_nelements(ids);
    const int ne0      = (int) ids->ne[0];
    const int n_expert = (int) map->ne[0];
    const int n_slots  = (int) ((ggml_nelements(state) - MOE_CACHE_HDR)/3);

    GGML_ASSERT(ggml_is_contiguous(ids) && ggml_is_contiguous(dst));

    if (n_ids == 0) {
        return; // for example the last layer of a ubatch without outputs
    }

    cudaStream_t stream = ctx.stream();

    moe_cache_assign<<<1, MOE_CACHE_LANES, 2*n_ids*sizeof(int32_t), stream>>>(
        (const int32_t *) ids->data, (int32_t *) map->data, (int32_t *) state->data, (int32_t *) dst->data,
        n_ids, ne0, n_expert, il, n_slots, tail_rank, split);
    CUDA_CHECK(cudaGetLastError());

    moe_cache_fetch_args args = {};
    int64_t max_nb = 0;
    for (int b = 0; b < n_banks; ++b) {
        const ggml_tensor * bank = dst->src[3 + b];
        memcpy(&args.src[b], (const int32_t *) dst->op_params + 2 + 2*b, sizeof(void *));
        args.dst[b] = (char *) bank->data;
        args.nb[b]  = bank->nb[2];
        max_nb = std::max(max_nb, (int64_t) bank->nb[2]);
    }

    const int64_t chunk    = 64*1024;
    const int     n_chunks = (int) ((max_nb + chunk - 1)/chunk);
    // enough blocks to keep PCIe busy for a handful of misses (each block has 16 KB of loads in flight), few enough that an
    // empty launch stays at a few microseconds (RTX 5090: 2250 idle blocks cost 17 us per layer, 48 layers per step)
    const int n_blocks = std::min(n_ids*n_banks*n_chunks, 512);
    if (split) {
        // fork: the copy runs on the side stream while the main stream computes the resident experts; ggml_moe_cache_join waits for it
        GGML_ASSERT(!ctx.moe_fork_open && "a split MoE cache lookup was not joined");
        cudaStream_t side = ctx.stream(ctx.device, MOE_CACHE_SIDE_STREAM);
        CUDA_CHECK(cudaEventRecord(moe_cache_event(ctx.device, 0), stream));
        CUDA_CHECK(cudaStreamWaitEvent(side, moe_cache_event(ctx.device, 0), 0));
        moe_cache_fetch<<<n_blocks, 256, 0, side>>>((const int32_t *) dst->data, n_ids, ne0, args, chunk, n_banks, n_chunks);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaEventRecord(moe_cache_event(ctx.device, 1), side));
        ctx.moe_fork_open = true;
        return;
    }
    moe_cache_fetch<<<n_blocks, 256, 0, stream>>>((const int32_t *) dst->data, n_ids, ne0, args, chunk, n_banks, n_chunks);
    CUDA_CHECK(cudaGetLastError());
}
