#pragma once

// Host-driven expert cache for MoE decode (M1 of docs/expert_cache_plan.md).
//
// The routed expert weights stay in host memory (mmap). Three device "bank" tensors hold S slots,
// one (layer, expert) triple per slot, shared by all layers. For a single-token decode the graph
// runs a CPU custom op on the router's top-k ids: it looks the ids up in a global LRU, copies the
// missing experts host -> device, and returns the slot ids that mul_mat_id then uses on the banks.

#include "ggml.h"
#include "ggml-backend.h"
#include "ggml-cpp.h"

#include <cstdint>
#include <vector>

struct llama_model;

struct llama_expert_cache {
    // slots: number of (layer, expert) triples resident on the device
    llama_expert_cache(const llama_model & model, ggml_backend_dev_t dev, uint32_t slots);
    ~llama_expert_cache();

    // true when this layer's routed experts are served from the cache (host tensors match the banks)
    bool covers(int il, const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down) const;

    ggml_tensor * bank_gate() const { return t_gate; }
    ggml_tensor * bank_up()   const { return t_up;   }
    ggml_tensor * bank_down() const { return t_down; }

    // userdata for the prepare op of layer il (valid for the cache's lifetime)
    void * userdata(int il) { return &per_layer[il]; }

    // ggml_custom1_op_t: dst/a are I32 [k, n_tokens]; a = original expert ids, dst = slot ids
    static void prepare_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata);

    uint32_t n_slots() const { return slots; }
    int64_t  n_hits()  const { return hits; }
    int64_t  n_miss()  const { return misses; }
    size_t   bytes_transferred() const { return bytes_h2d; }

private:
    struct layer_src {
        llama_expert_cache * cache = nullptr;
        int                  il    = -1;
        const ggml_tensor *  gate  = nullptr;
        const ggml_tensor *  up    = nullptr;
        const ggml_tensor *  down  = nullptr;
    };

    void prepare(int il, const int32_t * ids, int64_t k, int64_t n_tok, size_t nb_ids, int32_t * out, size_t nb_out);
    int32_t take_slot();          // free slot, or evict the least recently used one
    void    touch(int32_t slot);  // move to the most-recently-used end

    uint32_t slots   = 0;
    int      n_layer = 0;
    int      n_expert = 0;

    std::vector<layer_src> per_layer;

    ggml_context_ptr        bank_ctx;
    ggml_backend_buffer_ptr bank_buf;
    ggml_tensor * t_gate = nullptr;
    ggml_tensor * t_up   = nullptr;
    ggml_tensor * t_down = nullptr;

    // LRU over slots: doubly linked list, head = most recently used
    std::vector<int32_t> slot_of;   // gid -> slot or -1
    std::vector<int32_t> gid_of;    // slot -> gid or -1
    std::vector<int32_t> lru_prev, lru_next;
    int32_t lru_head = -1, lru_tail = -1;
    std::vector<int32_t> free_slots;

    int64_t hits = 0, misses = 0;
    size_t  bytes_h2d = 0;

    ggml_backend_buffer_ptr staging_buf;   // optional page-locked staging for the misses
    uint8_t * staging = nullptr;

    std::vector<void *> pinned_ranges;
    void (*unreg_fn)(void *) = nullptr;
};
