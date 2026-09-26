#pragma once

#include "ggml.h"
#include "ggml-backend.h"

#ifdef  __cplusplus
extern "C" {
#endif

#ifdef GGML_USE_HIP
#define GGML_CUDA_NAME "ROCm"
#define GGML_CUBLAS_NAME "hipBLAS"
#elif defined(GGML_USE_MUSA)
#define GGML_CUDA_NAME "MUSA"
#define GGML_CUBLAS_NAME "muBLAS"
#else
#define GGML_CUDA_NAME "CUDA"
#define GGML_CUBLAS_NAME "cuBLAS"
#endif
#define GGML_CUDA_MAX_DEVICES       16

// backend API
GGML_BACKEND_API ggml_backend_t ggml_backend_cuda_init(int device);

GGML_BACKEND_API bool ggml_backend_is_cuda(ggml_backend_t backend);

// device buffer
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device);

// conduct allreduce operation between devices
GGML_BACKEND_API bool ggml_backend_cuda_allreduce_tensor(ggml_backend_t * backends, struct ggml_tensor ** tensors, size_t n_backends);

// pinned host buffer for use with the CPU backend for faster copies between CPU and GPU
GGML_BACKEND_API ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type(void);

GGML_BACKEND_API int  ggml_backend_cuda_get_device_count(void);
GGML_BACKEND_API void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size);
GGML_BACKEND_API void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total);

GGML_BACKEND_API bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size);

// device-side MoE expert cache (ggml-cuda/expert-cache.cu). Also reachable through
// ggml_backend_reg_get_proc_address("ggml_backend_cuda_ecache_*") so that llama does not link CUDA.
struct ggml_cuda_ecache_params {
    int32_t n_layer;
    int32_t n_expert;
    int32_t n_expert_used;          // k: experts per token
    int32_t slots;                  // total slots in the banks
    const void * const * host_src;  // [n_layer*3] host pointers to the expert tensors (gate, up, down), NULL = layer not cached
    const size_t * host_bytes;      // [n_layer*3] bytes of each host tensor (for page-locking)
    void * bank[3];                 // device bank base pointers (gate, up, down)
    size_t nb[3];                   // bytes per expert in each bank
};
GGML_BACKEND_API void * ggml_backend_cuda_ecache_create(const struct ggml_cuda_ecache_params * params);
GGML_BACKEND_API void   ggml_backend_cuda_ecache_free(void * desc);
// hits, misses since creation (synchronizes the device)
GGML_BACKEND_API void   ggml_backend_cuda_ecache_stats(void * desc, int64_t * hits, int64_t * misses);
// the device descriptor to pass to ggml_expert_cache()
GGML_BACKEND_API void * ggml_backend_cuda_ecache_desc(void * handle);
GGML_BACKEND_API int    ggml_backend_cuda_ecache_verify(void * handle, int max_check);
GGML_BACKEND_API void ggml_backend_cuda_unregister_host_buffer(void * buffer);

GGML_BACKEND_API ggml_backend_reg_t ggml_backend_cuda_reg(void);

#ifdef  __cplusplus
}
#endif
