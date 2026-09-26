// Device-side MoE expert cache (M3 of docs/expert_cache_plan.md).
//
// The routed expert weights stay in host memory, page-locked and mapped into the device address
// space. Three device banks hold `slots` (layer, expert) triples, partitioned per layer. For a
// decode token the op maps the router's k expert ids of one layer to bank slots: a lookup kernel
// runs the per-layer LRU (hits refresh their stamp, misses evict the least recently used slot that
// this token does not use) and writes a miss list; a fill kernel then copies the missing experts
// from host memory into their slots with the whole GPU (PCIe DMA from mapped memory runs at the
// link speed). Both kernels run on the compute stream, so the following mul_mat_id is ordered
// after them without any host synchronization, and the whole token stays inside one CUDA graph.

#include "expert-cache.cuh"

#include <cstdint>
#include <cstring>
#include <vector>

#define ECACHE_MAX_K 32

struct ecache_desc {
    int32_t n_layer;
    int32_t n_expert;
    int32_t k_max;
    int32_t slots_per_part;     // P: slots of one layer partition

    int32_t * part_of_layer;    // [n_layer] partition index or -1
    int32_t * table;            // [n_layer * n_expert] local slot or -1
    int32_t * gid_of;           // [n_parts * P] expert id or -1
    int32_t * stamp;            // [n_parts * P]
    int32_t * tick;             // [1]
    int32_t * miss_list;        // [k_max * 2] (expert id, global slot)
    int32_t * miss_count;       // [1]
    unsigned long long * stats; // [2] hits, misses

    const char ** host_src;     // [n_layer * 3] device-visible pointers into the mapped host tensors
    char * bank[3];
    size_t nb[3];
};

struct ecache_host {
    ecache_desc          h;        // host copy of the descriptor
    ecache_desc        * d;        // device copy (what the op receives)
    std::vector<void *>  registered;
    std::vector<void *>  device_allocs;
};

// ---------------------------------------------------------------------------------------------
// lookup: one block per op, thread 0 does the bookkeeping, all threads do the LRU argmin

__global__ void ecache_lookup_kernel(const ecache_desc * __restrict__ d, int layer, const int32_t * __restrict__ ids, int k, int32_t * __restrict__ out) {
    __shared__ int32_t s_pending[ECACHE_MAX_K];
    __shared__ int32_t s_n_pending;
    __shared__ int32_t s_victim;
    __shared__ int32_t s_red_val[32];
    __shared__ int32_t s_red_idx[32];

    const int part = d->part_of_layer[layer];
    const int P    = d->slots_per_part;
    int32_t * table  = d->table + (size_t) layer * d->n_expert;
    int32_t * gid_of = d->gid_of + (size_t) part * P;
    int32_t * stamp  = d->stamp  + (size_t) part * P;
    const int base   = part * P;

    if (threadIdx.x == 0) {
        const int32_t tick = ++(*d->tick);
        int n_pending = 0, hits = 0;
        for (int j = 0; j < k; ++j) {
            const int32_t e = ids[j];
            const int32_t s = table[e];
            if (s >= 0) {
                stamp[s] = tick; out[j] = base + s; hits++;
            } else {
                s_pending[n_pending++] = j; out[j] = -1;
            }
        }
        s_n_pending = n_pending;
        *d->miss_count = 0;
        if (hits) atomicAdd(&d->stats[0], (unsigned long long) hits);
    }
    __syncthreads();

    const int n_pending = s_n_pending;
    const int32_t tick  = *d->tick;
    for (int p = 0; p < n_pending; ++p) {
        const int j = s_pending[p];
        const int32_t e = ids[j];
        // the same expert may appear twice in a row: resolved by the first occurrence
        if (table[e] >= 0) {
            if (threadIdx.x == 0) { out[j] = base + table[e]; }
            __syncthreads();
            continue;
        }
        // argmin of stamp over the partition, skipping slots this token already uses (stamp == tick)
        int32_t best_val = INT32_MAX, best_idx = -1;
        for (int s = threadIdx.x; s < P; s += blockDim.x) {
            const int32_t st = stamp[s];
            if (st != tick && st < best_val) { best_val = st; best_idx = s; }
        }
        // warp reduce
        for (int off = 16; off > 0; off >>= 1) {
            const int32_t ov = __shfl_down_sync(0xffffffff, best_val, off);
            const int32_t oi = __shfl_down_sync(0xffffffff, best_idx, off);
            if (ov < best_val || (ov == best_val && oi >= 0 && oi < best_idx)) { best_val = ov; best_idx = oi; }
        }
        const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
        if (lane == 0) { s_red_val[warp] = best_val; s_red_idx[warp] = best_idx; }
        __syncthreads();
        if (threadIdx.x == 0) {
            int32_t bv = INT32_MAX, bi = -1;
            for (int w = 0; w < (int) (blockDim.x / 32); ++w) {
                if (s_red_val[w] < bv || (s_red_val[w] == bv && s_red_idx[w] >= 0 && s_red_idx[w] < bi)) { bv = s_red_val[w]; bi = s_red_idx[w]; }
            }
            s_victim = bi;
            // evict + install
            const int32_t old = gid_of[bi];
            if (old >= 0) { table[old] = -1; }
            gid_of[bi] = e; table[e] = bi; stamp[bi] = tick;
            const int32_t m = (*d->miss_count)++;
            d->miss_list[2*m] = e; d->miss_list[2*m + 1] = base + bi;
            out[j] = base + bi;
            atomicAdd(&d->stats[1], 1ull);
        }
        __syncthreads();
    }
}

// ---------------------------------------------------------------------------------------------
// fill: grid (k_max, chunks); block (m, c) copies chunk c of the three tensors of miss m

__global__ void ecache_fill_kernel(const ecache_desc * __restrict__ d, int layer) {
    const int m = blockIdx.x;
    if (m >= *d->miss_count) {
        return;
    }
    const int32_t e = d->miss_list[2*m];
    const int32_t s = d->miss_list[2*m + 1];
    const int c = blockIdx.y, nc = gridDim.y;
    for (int t = 0; t < 3; ++t) {
        const size_t nb = d->nb[t];
        const uint4 * src = (const uint4 *) (d->host_src[(size_t) layer * 3 + t] + (size_t) e * nb);
        uint4 * dst = (uint4 *) (d->bank[t] + (size_t) s * nb);
        const size_t n16 = nb / 16;
        const size_t per = (n16 + nc - 1) / nc;
        const size_t lo = (size_t) c * per, hi = min(n16, lo + per);
        for (size_t i = lo + threadIdx.x; i < hi; i += blockDim.x) {
            dst[i] = src[i];
        }
    }
}

// ---------------------------------------------------------------------------------------------

void ggml_cuda_op_expert_cache(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * ids = dst->src[0];
    GGML_ASSERT(ids->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(ids) && ggml_is_contiguous(dst));
    GGML_ASSERT(ids->ne[1] == 1 && ids->ne[2] == 1 && ids->ne[3] == 1 && "expert cache op: one token per op");

    void * desc = nullptr;
    memcpy(&desc, dst->op_params, sizeof(desc));
    const int layer = ((const int32_t *) dst->op_params)[2];
    const int k = (int) ids->ne[0];
    GGML_ASSERT(desc != nullptr && k <= ECACHE_MAX_K);

    const auto * d = (const ecache_desc *) desc;
    cudaStream_t stream = ctx.stream();
    ecache_lookup_kernel<<<1, 256, 0, stream>>>(d, layer, (const int32_t *) ids->data, k, (int32_t *) dst->data);
    ecache_fill_kernel<<<dim3(k, 32), 256, 0, stream>>>(d, layer);
}

// ---------------------------------------------------------------------------------------------
// host API

template <typename T>
static T * ecache_dev_alloc(ecache_host * h, size_t n) {
    T * p = nullptr;
    CUDA_CHECK(cudaMalloc(&p, n * sizeof(T)));
    CUDA_CHECK(cudaMemset(p, 0xff, n * sizeof(T)));   // -1 for int tables
    h->device_allocs.push_back(p);
    return p;
}

void * ggml_backend_cuda_ecache_create(const struct ggml_cuda_ecache_params * p) {
    auto * h = new ecache_host();
    ecache_desc & d = h->h;
    d.n_layer = p->n_layer; d.n_expert = p->n_expert; d.k_max = ECACHE_MAX_K;
    GGML_ASSERT(p->n_expert_used <= ECACHE_MAX_K);

    // partitions: covered layers get equal shares
    std::vector<int32_t> part_of_layer(p->n_layer, -1);
    int n_parts = 0;
    for (int il = 0; il < p->n_layer; ++il) {
        if (p->host_src[(size_t) il * 3] != nullptr) part_of_layer[il] = n_parts++;
    }
    GGML_ASSERT(n_parts > 0);
    d.slots_per_part = p->slots / n_parts;
    GGML_ASSERT(d.slots_per_part >= p->n_expert_used);

    // map the host expert tensors into the device address space (page-locked)
    std::vector<const char *> host_src((size_t) p->n_layer * 3, nullptr);
    const size_t page = 4096;
    size_t pinned = 0;
    for (int il = 0; il < p->n_layer; ++il) {
        for (int t = 0; t < 3; ++t) {
            const void * src = p->host_src[(size_t) il * 3 + t];
            if (!src) continue;
            const uintptr_t b = (uintptr_t) src & ~(uintptr_t) (page - 1);
            const uintptr_t e = ((uintptr_t) src + p->host_bytes[(size_t) il * 3 + t] + page - 1) & ~(uintptr_t) (page - 1);
            cudaError_t err = cudaHostRegister((void *) b, e - b, cudaHostRegisterMapped | cudaHostRegisterPortable | cudaHostRegisterReadOnly);
            if (err != cudaSuccess) {
                (void) cudaGetLastError();
                GGML_LOG_ERROR("%s: cudaHostRegister(mapped) failed for layer %d tensor %d: %s\n", __func__, il, t, cudaGetErrorString(err));
                ggml_backend_cuda_ecache_free(h);
                return nullptr;
            }
            h->registered.push_back((void *) b);
            pinned += e - b;
            void * dptr = nullptr;
            CUDA_CHECK(cudaHostGetDevicePointer(&dptr, (void *) b, 0));
            host_src[(size_t) il * 3 + t] = (const char *) dptr + ((uintptr_t) src - b);
        }
    }
    GGML_LOG_INFO("%s: %d partitions x %d slots, %.1f GiB of host expert memory mapped for device access\n",
            __func__, n_parts, d.slots_per_part, pinned / (1024.0 * 1024.0 * 1024.0));

    d.part_of_layer = ecache_dev_alloc<int32_t>(h, p->n_layer);
    CUDA_CHECK(cudaMemcpy(d.part_of_layer, part_of_layer.data(), part_of_layer.size() * sizeof(int32_t), cudaMemcpyHostToDevice));
    d.table   = ecache_dev_alloc<int32_t>(h, (size_t) p->n_layer * p->n_expert);
    d.gid_of  = ecache_dev_alloc<int32_t>(h, (size_t) n_parts * d.slots_per_part);
    d.stamp   = ecache_dev_alloc<int32_t>(h, (size_t) n_parts * d.slots_per_part);
    CUDA_CHECK(cudaMemset(d.stamp, 0, (size_t) n_parts * d.slots_per_part * sizeof(int32_t)));   // stamp 0 = never used
    d.tick    = ecache_dev_alloc<int32_t>(h, 1);
    CUDA_CHECK(cudaMemset(d.tick, 0, sizeof(int32_t)));
    d.miss_list  = ecache_dev_alloc<int32_t>(h, (size_t) ECACHE_MAX_K * 2);
    d.miss_count = ecache_dev_alloc<int32_t>(h, 1);
    CUDA_CHECK(cudaMemset(d.miss_count, 0, sizeof(int32_t)));
    d.stats = ecache_dev_alloc<unsigned long long>(h, 2);
    CUDA_CHECK(cudaMemset(d.stats, 0, 2 * sizeof(unsigned long long)));
    d.host_src = (const char **) ecache_dev_alloc<const char *>(h, host_src.size());
    CUDA_CHECK(cudaMemcpy((void *) d.host_src, host_src.data(), host_src.size() * sizeof(const char *), cudaMemcpyHostToDevice));
    for (int t = 0; t < 3; ++t) { d.bank[t] = (char *) p->bank[t]; d.nb[t] = p->nb[t]; GGML_ASSERT(d.nb[t] % 16 == 0); }

    CUDA_CHECK(cudaMalloc(&h->d, sizeof(ecache_desc)));
    CUDA_CHECK(cudaMemcpy(h->d, &d, sizeof(ecache_desc), cudaMemcpyHostToDevice));
    return h;   // opaque handle; the op receives ecache_desc_device()
}

void ggml_backend_cuda_ecache_free(void * handle) {
    auto * h = (ecache_host *) handle;
    if (!h) return;
    cudaDeviceSynchronize();
    for (void * p : h->device_allocs) cudaFree(p);
    if (h->d) cudaFree(h->d);
    for (void * p : h->registered) cudaHostUnregister(p);
    delete h;
}

void ggml_backend_cuda_ecache_stats(void * handle, int64_t * hits, int64_t * misses) {
    auto * h = (ecache_host *) handle;
    unsigned long long st[2] = {0, 0};
    cudaDeviceSynchronize();
    cudaMemcpy(st, h->h.stats, sizeof(st), cudaMemcpyDeviceToHost);
    *hits = (int64_t) st[0]; *misses = (int64_t) st[1];
}

// debug: compare the bytes of resident slots with their host experts (samples up to `max_check` entries)
int ggml_backend_cuda_ecache_verify(void * handle, int max_check) {
    auto * h = (ecache_host *) handle;
    const ecache_desc & d = h->h;
    cudaDeviceSynchronize();
    const size_t n_gid = (size_t) d.n_layer * d.n_expert;
    std::vector<int32_t> table(n_gid), part_of_layer(d.n_layer);
    cudaMemcpy(table.data(), d.table, n_gid * sizeof(int32_t), cudaMemcpyDeviceToHost);
    cudaMemcpy(part_of_layer.data(), d.part_of_layer, d.n_layer * sizeof(int32_t), cudaMemcpyDeviceToHost);
    std::vector<const char *> host_src((size_t) d.n_layer * 3);
    cudaMemcpy(host_src.data(), d.host_src, host_src.size() * sizeof(const char *), cudaMemcpyDeviceToHost);
    int checked = 0, bad = 0;
    std::vector<char> buf;
    for (size_t gid = 0; gid < n_gid && checked < max_check; gid += 97) {   // stride: sample across layers
        const int32_t s = table[gid];
        if (s < 0) continue;
        const int il = (int) (gid / d.n_expert), e = (int) (gid % d.n_expert);
        const int gslot = part_of_layer[il] * d.slots_per_part + s;
        for (int t = 0; t < 3; ++t) {
            buf.resize(d.nb[t]);
            cudaMemcpy(buf.data(), d.bank[t] + (size_t) gslot * d.nb[t], d.nb[t], cudaMemcpyDeviceToHost);
            // host_src holds device-visible pointers of mapped memory: read them back through the device too
            std::vector<char> ref(d.nb[t]);
            cudaMemcpy(ref.data(), host_src[(size_t) il * 3 + t] + (size_t) e * d.nb[t], d.nb[t], cudaMemcpyDefault);
            if (memcmp(buf.data(), ref.data(), d.nb[t]) != 0) bad++;
        }
        checked++;
    }
    GGML_LOG_INFO("%s: %d resident experts checked, %d tensor mismatches\n", __func__, checked, bad);
    return bad;
}

void * ggml_backend_cuda_ecache_desc(void * handle) {
    return ((ecache_host *) handle)->d;
}
