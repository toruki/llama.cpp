#include "argsort.cuh"
#include "top-k.cuh"

#ifdef GGML_CUDA_USE_CUB
#    include <cub/cub.cuh>
#    if (CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2)
#        define CUB_TOP_K_AVAILABLE
#        include <cuda/iterator>
using namespace cub;
#    endif  // CCCL_MAJOR_VERSION >= 3 && CCCL_MINOR_VERSION >= 2
#endif      // GGML_CUDA_USE_CUB

#ifdef CUB_TOP_K_AVAILABLE

// Ties are broken by the lower index, so the selected set is a deterministic function of the input.
// (DeviceTopK on the raw floats picks arbitrarily among equal values, which changes run to run: the
// qwen4exp sparse attention expands one score to every cell of a block, so its cut-off is a tie 3 times in 4.)
// Each value is packed with its index into one 64-bit key: the order-preserving image of the float in
// the high word and the bitwise-complemented index in the low word, so a lower index makes a larger key.
static __device__ __forceinline__ uint32_t top_k_ordered_bits(float f) {
    const uint32_t u = __float_as_uint(f);
    return (u & 0x80000000u) ? ~u : (u | 0x80000000u);
}

static __global__ void top_k_pack_keys(const float * __restrict__ src, unsigned long long * __restrict__ keys, const int ncols) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < ncols) {
        keys[i] = ((unsigned long long) top_k_ordered_bits(src[i]) << 32) | (unsigned long long) (~(uint32_t) i);
    }
}

static __global__ void top_k_unpack_keys(const unsigned long long * __restrict__ keys, int * __restrict__ dst, const int k) {
    const int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i < k) {
        dst[i] = (int) ~(uint32_t) (keys[i] & 0xFFFFFFFFull);
    }
}

// Multi-row top-k in one launch: one thread block per row selects the k largest packed keys by radix
// select (8 passes of 8 bits over the 64-bit key, histogram in shared memory), then collects the keys at or
// above the k-th largest. The keys are unique, so exactly k are collected; the set is the same as the
// per-row DeviceTopK path above (same key, same tie rule), the order within a row is arbitrary as there.
// The keys are packed on the fly from the floats, so a row is read 9 times (4 bytes per element) and nothing
// is materialized. Used for batched (prefill) rows: the per-row DeviceTopK costs ~8 launches per row.
#define TOPK_SELECT_BLOCK 256

static __global__ void k_top_k_select(const float * __restrict__ src, int * __restrict__ dst, const int ncols, const int k, const int64_t ncols_stride) {
    const int row = blockIdx.x;
    const float * x = src + (int64_t) row*ncols_stride;
    int * out = dst + (int64_t) row*k;

    __shared__ int hist[256];
    __shared__ unsigned long long s_prefix;
    __shared__ int s_remaining;
    __shared__ int s_count;

    if (threadIdx.x == 0) {
        s_prefix = 0;
        s_remaining = k;
        s_count = 0;
    }
    __syncthreads();

    unsigned long long prefix = 0;   // the digits chosen so far, in place
    unsigned long long mask   = 0;   // which digits are chosen

    for (int pass = 0; pass < 8; ++pass) {
        const int shift = 56 - 8*pass;

        for (int b = threadIdx.x; b < 256; b += blockDim.x) {
            hist[b] = 0;
        }
        __syncthreads();

        for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
            const unsigned long long key = ((unsigned long long) top_k_ordered_bits(x[i]) << 32) | (unsigned long long) (~(uint32_t) i);
            if ((key & mask) == prefix) {
                atomicAdd(&hist[(int) ((key >> shift) & 255)], 1);
            }
        }
        __syncthreads();

        if (threadIdx.x == 0) {
            // walk the digits from the largest: the k-th largest key lives in the first bin that
            // brings the cumulative count to s_remaining; the keys in the higher bins are all selected
            int cum = 0;
            int chosen = 0;
            for (int d = 255; d >= 0; --d) {
                if (cum + hist[d] >= s_remaining) {
                    chosen = d;
                    break;
                }
                cum += hist[d];
            }
            s_remaining -= cum;
            s_prefix = prefix | ((unsigned long long) chosen << shift);
        }
        __syncthreads();
        prefix = s_prefix;
        mask  |= (unsigned long long) 255 << shift;
        __syncthreads();
    }

    // prefix is now the k-th largest key; collect every key at or above it
    const unsigned long long threshold = prefix;
    for (int i = threadIdx.x; i < ncols; i += blockDim.x) {
        const unsigned long long key = ((unsigned long long) top_k_ordered_bits(x[i]) << 32) | (unsigned long long) (~(uint32_t) i);
        if (key >= threshold) {
            const int pos = atomicAdd(&s_count, 1);
            if (pos < k) {
                out[pos] = i;
            }
        }
    }
}

static void top_k_cub(ggml_cuda_pool & pool,
                      const float *    src,
                      int *            dst,
                      const int        ncols,
                      const int        k,
                      cudaStream_t     stream) {
    auto requirements = cuda::execution::require(cuda::execution::determinism::not_guaranteed,
                                                 cuda::execution::output_ordering::unsorted);
    auto stream_env   = cuda::stream_ref{ stream };
    auto env          = cuda::std::execution::env{ stream_env, requirements };

    // GGML_CUDA_TOPK_DET=0 restores the value-only selection (arbitrary among ties), for A/B tests
    static const bool det = [] { const char * e = getenv("GGML_CUDA_TOPK_DET"); return e == nullptr || atoi(e) != 0; }();
    if (!det) {
        auto indexes_in = cuda::make_counting_iterator(0);
        size_t temp_storage_bytes = 0;
        CUDA_CHECK(DeviceTopK::MaxPairs(nullptr, temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k, env));
        ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
        CUDA_CHECK(DeviceTopK::MaxPairs(temp_storage_alloc.get(), temp_storage_bytes, src, cuda::discard_iterator(), indexes_in, dst, ncols, k, env));
        return;
    }

    ggml_cuda_pool_alloc<unsigned long long> keys_alloc(pool, ncols + k);
    unsigned long long * keys_in  = keys_alloc.get();
    unsigned long long * keys_out = keys_in + ncols;

    top_k_pack_keys<<<(ncols + 255)/256, 256, 0, stream>>>(src, keys_in, ncols);

    size_t temp_storage_bytes = 0;
    CUDA_CHECK(DeviceTopK::MaxKeys(nullptr, temp_storage_bytes, keys_in, keys_out, ncols, k, env));

    ggml_cuda_pool_alloc<uint8_t> temp_storage_alloc(pool, temp_storage_bytes);
    void *                        d_temp_storage = temp_storage_alloc.get();

    CUDA_CHECK(DeviceTopK::MaxKeys(d_temp_storage, temp_storage_bytes, keys_in, keys_out, ncols, k, env));

    top_k_unpack_keys<<<(k + 255)/256, 256, 0, stream>>>(keys_out, dst, k);
}

#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE

static int next_power_of_2(int x) {
    int n = 1;
    while (n < x) {
        n *= 2;
    }
    return n;
}

#endif                            // CUB_TOP_K_AVAILABLE

void ggml_cuda_op_top_k(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0   = dst->src[0];
    const float *       src0_d = (const float *) src0->data;
    int *               dst_d  = (int *) dst->data;
    cudaStream_t        stream = ctx.stream();

    // are these asserts truly necessary?
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_I32);
    GGML_ASSERT(ggml_is_contiguous(src0));

    const int64_t    ncols = src0->ne[0];
    const int64_t    nrows = ggml_nrows(src0);
    const int64_t    k     = dst->ne[0];
    ggml_cuda_pool & pool  = ctx.pool();
#ifdef CUB_TOP_K_AVAILABLE
    // batched rows: one radix-select launch for all rows (GGML_CUDA_TOPK_BATCH=0 falls back to the per-row loop)
    static const bool batch_select = [] { const char * e = getenv("GGML_CUDA_TOPK_BATCH"); return e == nullptr || atoi(e) != 0; }();
    if (batch_select && nrows > 1) {
        GGML_ASSERT(nrows <= INT32_MAX && ncols <= INT32_MAX && k <= ncols);
        k_top_k_select<<<(int) nrows, TOPK_SELECT_BLOCK, 0, stream>>>(src0_d, dst_d, (int) ncols, (int) k, ncols);
        return;
    }
    // TODO: Switch to `DeviceSegmentedTopK` for multi-row TopK once implemented
    // https://github.com/NVIDIA/cccl/issues/6391
    for (int i = 0; i < nrows; i++) {
        top_k_cub(pool, src0_d + i * ncols, dst_d + i * k, ncols, k, stream);
    }
#elif defined(GGML_CUDA_USE_CUB)  // CUB_TOP_K_AVAILABLE
    // Fall back to argsort + copy
    const int    ncols_pad      = next_power_of_2(ncols);
    const size_t shared_mem     = ncols_pad * sizeof(int);
    const size_t max_shared_mem = ggml_cuda_info().devices[ggml_cuda_get_device()].smpb;
    const bool   use_bitonic    = shared_mem <= max_shared_mem && ncols <= 1024;
    const int    chunk_nrows    = argsort_f32_i32_cuda_cub_chunk_nrows(src0->nb[1], nrows);

    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * chunk_nrows);
    int *                     tmp_dst = temp_dst_alloc.get();

    for (int64_t i = 0; i < nrows; i += chunk_nrows) {
        int iter_nrows = std::min((int64_t) chunk_nrows, nrows - i);

        if (use_bitonic) {
            argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        } else {
            argsort_f32_i32_cuda_cub(pool, src0_d, tmp_dst, ncols, iter_nrows, GGML_SORT_ORDER_DESC, stream);
        }
        CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), iter_nrows,
                                     cudaMemcpyDeviceToDevice, stream));

        src0_d += ncols * iter_nrows;
        dst_d  += k     * iter_nrows;
    }
#else                             // GGML_CUDA_USE_CUB
    ggml_cuda_pool_alloc<int> temp_dst_alloc(pool, ncols * nrows);
    int *                     tmp_dst = temp_dst_alloc.get();
    argsort_f32_i32_cuda_bitonic(src0_d, tmp_dst, ncols, nrows, GGML_SORT_ORDER_DESC, stream);
    CUDA_CHECK(cudaMemcpy2DAsync(dst_d, k * sizeof(int), tmp_dst, ncols * sizeof(int), k * sizeof(int), nrows,
                                 cudaMemcpyDeviceToDevice, stream));
#endif
}
