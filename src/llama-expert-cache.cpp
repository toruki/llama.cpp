#include "llama-expert-cache.h"

#include "llama-impl.h"
#include "llama-model.h"

#include "ggml-alloc.h"

#include <cstdio>
#include <cstdlib>
#include <algorithm>
#include <cstring>
#include <stdexcept>

llama_expert_cache::llama_expert_cache(const llama_model & model, ggml_backend_dev_t dev, ggml_backend_t backend_, uint32_t slots_)
    : slots(slots_), n_layer((int) model.hparams.n_layer()), n_expert((int) model.hparams.n_expert), backend(backend_), dev_(dev) {
    if (slots < 2 * model.hparams.n_expert_used) {
        throw std::runtime_error("expert cache: too few slots");
    }
    per_layer.resize(n_layer);

    // the banks are shaped like layer 0's routed experts; every cached layer must match
    const ggml_tensor * g0 = nullptr;
    const ggml_tensor * u0 = nullptr;
    const ggml_tensor * d0 = nullptr;
    int n_covered = 0;
    // LLAMA_EXPERT_CACHE_LAYERS="a-b" restricts the cache to layers a..b (testing aid)
    int lay_lo = 0, lay_hi = n_layer - 1;
    if (const char * env = getenv("LLAMA_EXPERT_CACHE_LAYERS")) {
        if (sscanf(env, "%d-%d", &lay_lo, &lay_hi) != 2) {
            throw std::runtime_error("expert cache: bad LLAMA_EXPERT_CACHE_LAYERS");
        }
    }
    for (int il = 0; il < n_layer; ++il) {
        const auto & L = model.layers[il];
        auto & src = per_layer[il];
        src.cache = this; src.il = il;
        if (!L.ffn_gate_exps || !L.ffn_up_exps || !L.ffn_down_exps || il < lay_lo || il > lay_hi) {
            continue;
        }
        bool host = true;
        for (const ggml_tensor * t : {L.ffn_gate_exps, L.ffn_up_exps, L.ffn_down_exps}) {
            if (!t->buffer || !ggml_backend_buffer_is_host(t->buffer) || t->data == nullptr) {
                host = false;
            }
        }
        if (!host) {
            LLAMA_LOG_WARN("%s: layer %d routed experts are not in host memory, not cached\n", __func__, il);
            continue;
        }
        if (!g0) { g0 = L.ffn_gate_exps; u0 = L.ffn_up_exps; d0 = L.ffn_down_exps; }
        bool same = true;
        for (auto pr : {std::make_pair(g0, (const ggml_tensor *) L.ffn_gate_exps),
                        std::make_pair(u0, (const ggml_tensor *) L.ffn_up_exps),
                        std::make_pair(d0, (const ggml_tensor *) L.ffn_down_exps)}) {
            const ggml_tensor * a = pr.first; const ggml_tensor * b = pr.second;
            same &= a->type == b->type && a->ne[0] == b->ne[0] && a->ne[1] == b->ne[1] && a->ne[2] == b->ne[2] &&
                    a->nb[1] == b->nb[1] && a->nb[2] == b->nb[2] && ggml_is_contiguous(b);
        }
        if (!same) {
            LLAMA_LOG_WARN("%s: layer %d routed experts differ from layer 0 (gate %s/%s, up %s/%s, down %s/%s), not cached\n",
                    __func__, il, ggml_type_name(L.ffn_gate_exps->type), ggml_type_name(g0->type),
                    ggml_type_name(L.ffn_up_exps->type), ggml_type_name(u0->type),
                    ggml_type_name(L.ffn_down_exps->type), ggml_type_name(d0->type));
            continue;
        }
        src.gate = L.ffn_gate_exps; src.up = L.ffn_up_exps; src.down = L.ffn_down_exps;
        n_covered++;
    }
    if (!g0) {
        throw std::runtime_error("expert cache: no routed experts in host memory (use --cpu-moe)");
    }
    this->n_covered = n_covered;
    for (int il = 0; il < n_layer; ++il) { if (per_layer[il].gate) { first_covered = il; break; } }
    use_count.assign((size_t) n_layer * n_expert, 0.0f);
    if (const char * env = getenv("LLAMA_EXPERT_CACHE_WARM")) warm_enabled = atoi(env) != 0;
    if (g0->ne[2] != n_expert) {
        throw std::runtime_error("expert cache: unexpected expert count");
    }

    // device banks
    ggml_init_params ip = { /*.mem_size =*/ ggml_tensor_overhead() * 8, /*.mem_buffer =*/ nullptr, /*.no_alloc =*/ true };
    bank_ctx.reset(ggml_init(ip));
    ggml_context * ctx = bank_ctx.get();
    t_gate = ggml_new_tensor_3d(ctx, g0->type, g0->ne[0], g0->ne[1], slots); ggml_set_name(t_gate, "expert_cache.gate");
    t_up   = ggml_new_tensor_3d(ctx, u0->type, u0->ne[0], u0->ne[1], slots); ggml_set_name(t_up,   "expert_cache.up");
    t_down = ggml_new_tensor_3d(ctx, d0->type, d0->ne[0], d0->ne[1], slots); ggml_set_name(t_down, "expert_cache.down");
    ggml_backend_buffer_type_t buft = ggml_backend_dev_buffer_type(dev);
    bank_buf.reset(ggml_backend_alloc_ctx_tensors_from_buft(ctx, buft));
    if (!bank_buf) {
        throw std::runtime_error("expert cache: failed to allocate the device banks");
    }
    ggml_backend_buffer_set_usage(bank_buf.get(), GGML_BACKEND_BUFFER_USAGE_WEIGHTS);
    GGML_ASSERT(t_gate->nb[2] == g0->nb[2] && t_up->nb[2] == u0->nb[2] && t_down->nb[2] == d0->nb[2]);

    slot_of.assign((size_t) n_layer * n_expert, -1);
    gid_of.assign(slots, -1);
    lru_prev.assign(slots, -1);
    lru_next.assign(slots, -1);
    free_slots.reserve(slots);
    for (int32_t s = (int32_t) slots - 1; s >= 0; --s) {
        free_slots.push_back(s);
    }

    // LLAMA_EXPERT_CACHE_PIN=1: page-lock the host expert tensors so the misses are copied by DMA at
    // full PCIe speed instead of the driver's staged pageable copies (pins ~45 GiB of page cache)
    if (const char * env = getenv("LLAMA_EXPERT_CACHE_PIN"); env && atoi(env) > 0) {
        auto * reg_fn = (bool (*)(void *, size_t)) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_register_host_buffer");
        unreg_fn = (void (*)(void *)) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev), "ggml_backend_unregister_host_buffer");
        if (!reg_fn || !unreg_fn) {
            LLAMA_LOG_WARN("%s: device cannot register host memory, not pinning\n", __func__);
        } else {
            size_t pinned = 0; int failed = 0;
            const size_t page = 4096;
            for (int il = 0; il < n_layer; ++il) {
                const auto & src = per_layer[il];
                if (!src.gate) continue;
                for (const ggml_tensor * t : {src.gate, src.up, src.down}) {
                    const uintptr_t b = (uintptr_t) t->data & ~(uintptr_t) (page - 1);
                    const uintptr_t e = ((uintptr_t) t->data + ggml_nbytes(t) + page - 1) & ~(uintptr_t) (page - 1);
                    if (reg_fn((void *) b, e - b)) { pinned_ranges.push_back((void *) b); pinned += e - b; } else { failed++; }
                }
            }
            LLAMA_LOG_INFO("%s: pinned %.1f GiB of host expert memory (%d ranges failed)\n", __func__, pinned / (1024.0 * 1024.0 * 1024.0), failed);
        }
    }

    // LLAMA_EXPERT_CACHE_STAGING=1: copy each missing expert through a page-locked staging buffer so the
    // device transfer is a single DMA instead of the driver's chunked pageable copy
    if (const char * env = getenv("LLAMA_EXPERT_CACHE_STAGING"); env && atoi(env) > 0) {
        ggml_backend_buffer_type_t hbuft = ggml_backend_dev_host_buffer_type(dev);
        staging_n = (int) model.hparams.n_expert_used;                 // misses of one token, copied asynchronously
        const size_t need = (size_t) staging_n * (g0->nb[2] + u0->nb[2] + d0->nb[2]);
        if (hbuft) {
            staging_buf.reset(ggml_backend_buft_alloc_buffer(hbuft, need));
        }
        if (staging_buf) {
            staging = (uint8_t *) ggml_backend_buffer_get_base(staging_buf.get());
            LLAMA_LOG_INFO("%s: staging misses through %.1f MiB of page-locked host memory (%s copies)\n", __func__,
                    need / (1024.0 * 1024.0), backend ? "async" : "sync");
        } else {
            LLAMA_LOG_WARN("%s: no page-locked host buffer type, staging disabled\n", __func__);
        }
    }

    // device-side cache (default when the backend provides it; LLAMA_EXPERT_CACHE_DEVICE=0 keeps the host LRU)
    {
        const char * env = getenv("LLAMA_EXPERT_CACHE_DEVICE");
        const bool want = env == nullptr || atoi(env) != 0;
        auto * reg = ggml_backend_dev_backend_reg(dev);
        auto * create_fn = (void * (*)(const void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_ecache_create");
        auto * desc_fn   = (void * (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_ecache_desc");
        dev_free_fn  = (void (*)(void *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_ecache_free");
        dev_stats_fn = (void (*)(void *, int64_t *, int64_t *)) ggml_backend_reg_get_proc_address(reg, "ggml_backend_cuda_ecache_stats");
        if (want && create_fn && desc_fn && dev_free_fn && dev_stats_fn) {
            struct params_t {   // mirrors ggml_cuda_ecache_params (kept here to avoid including the CUDA header)
                int32_t n_layer, n_expert, n_expert_used, slots;
                const void * const * host_src; const size_t * host_bytes;
                void * bank[3]; size_t nb[3];
            } prm;
            std::vector<const void *> hs((size_t) n_layer * 3, nullptr);
            std::vector<size_t> hb((size_t) n_layer * 3, 0);
            for (int il = 0; il < n_layer; ++il) {
                const auto & src = per_layer[il];
                if (!src.gate) continue;
                const ggml_tensor * ts[3] = {src.gate, src.up, src.down};
                for (int t = 0; t < 3; ++t) { hs[(size_t) il * 3 + t] = ts[t]->data; hb[(size_t) il * 3 + t] = ggml_nbytes(ts[t]); }
            }
            prm.n_layer = n_layer; prm.n_expert = n_expert; prm.n_expert_used = (int32_t) model.hparams.n_expert_used; prm.slots = (int32_t) slots;
            prm.host_src = hs.data(); prm.host_bytes = hb.data();
            prm.bank[0] = t_gate->data; prm.bank[1] = t_up->data; prm.bank[2] = t_down->data;
            prm.nb[0] = t_gate->nb[2]; prm.nb[1] = t_up->nb[2]; prm.nb[2] = t_down->nb[2];
            dev_handle = create_fn(&prm);
            if (dev_handle) {
                dev_desc = desc_fn(dev_handle);
                LLAMA_LOG_INFO("%s: device-side lookup and fill enabled (GGML_OP_EXPERT_CACHE)\n", __func__);
            } else {
                LLAMA_LOG_WARN("%s: device-side cache unavailable, using the host LRU\n", __func__);
            }
        }
    }

    const double slot_mib = (double) (g0->nb[2] + u0->nb[2] + d0->nb[2]) / (1024.0 * 1024.0);
    LLAMA_LOG_INFO("%s: %u slots x %.3f MiB = %.2f GiB on %s, %d/%d layers cached (%.1f%% of their experts resident)\n",
            __func__, slots, slot_mib, ggml_backend_buffer_get_size(bank_buf.get()) / (1024.0 * 1024.0 * 1024.0),
            ggml_backend_dev_name(dev), n_covered, n_layer, 100.0 * slots / ((double) n_covered * n_expert));
}

llama_expert_cache::~llama_expert_cache() {
    if (dev_handle) {
        if (const char * env = getenv("LLAMA_EXPERT_CACHE_VERIFY"); env && atoi(env) > 0) {
            auto * verify_fn = (int (*)(void *, int)) ggml_backend_reg_get_proc_address(ggml_backend_dev_backend_reg(dev_), "ggml_backend_cuda_ecache_verify");
            if (verify_fn) verify_fn(dev_handle, atoi(env));
        }
        int64_t h = 0, m = 0;
        dev_stats_fn(dev_handle, &h, &m);
        if (h + m > 0) {
            LLAMA_LOG_INFO("%s: device expert cache: %lld lookups, hit %.2f%%\n", __func__, (long long) (h + m), 100.0 * h / (h + m));
        }
        dev_free_fn(dev_handle);
    }
    if (unreg_fn) {
        for (void * p : pinned_ranges) unreg_fn(p);
    }
    const int64_t total = hits + misses;
    if (total > 0) {
        LLAMA_LOG_INFO("%s: expert cache: %lld lookups, hit %.2f%%, %.2f MiB transferred per miss avg, %.1f GiB total\n",
                __func__, (long long) total, 100.0 * hits / total,
                misses ? (double) bytes_h2d / misses / (1024.0 * 1024.0) : 0.0, bytes_h2d / (1024.0 * 1024.0 * 1024.0));
    }
}

bool llama_expert_cache::covers(int il, const ggml_tensor * up, const ggml_tensor * gate, const ggml_tensor * down) const {
    if (il < 0 || il >= n_layer) {
        return false;
    }
    const auto & s = per_layer[il];
    return s.gate == gate && s.up == up && s.down == down && s.gate != nullptr;
}

void llama_expert_cache::touch(int32_t s) {
    if (s == lru_head) {
        return;
    }
    // unlink
    if (lru_prev[s] >= 0) lru_next[lru_prev[s]] = lru_next[s];
    if (lru_next[s] >= 0) lru_prev[lru_next[s]] = lru_prev[s];
    if (lru_tail == s) lru_tail = lru_prev[s];
    // push front
    lru_prev[s] = -1;
    lru_next[s] = lru_head;
    if (lru_head >= 0) lru_prev[lru_head] = s;
    lru_head = s;
    if (lru_tail < 0) lru_tail = s;
}

int32_t llama_expert_cache::take_slot() {
    if (!free_slots.empty()) {
        const int32_t s = free_slots.back();
        free_slots.pop_back();
        // link as head
        lru_prev[s] = -1; lru_next[s] = lru_head;
        if (lru_head >= 0) lru_prev[lru_head] = s;
        lru_head = s;
        if (lru_tail < 0) lru_tail = s;
        return s;
    }
    const int32_t s = lru_tail;
    GGML_ASSERT(s >= 0);
    slot_of[gid_of[s]] = -1;
    gid_of[s] = -1;
    touch(s);
    return s;
}

void llama_expert_cache::load_expert(int il, int32_t e, int32_t s, int & n_inflight) {
    const auto & src = per_layer[il];
    const size_t nb2 = src.gate->nb[2], nbu = src.up->nb[2], nbd = src.down->nb[2];
    const char * pg = (const char *) src.gate->data + (size_t) e * nb2;
    const char * pu = (const char *) src.up->data   + (size_t) e * nbu;
    const char * pd = (const char *) src.down->data + (size_t) e * nbd;
    if (staging) {
        if (n_inflight == staging_n) {          // staging full: drain the queued copies
            if (backend) ggml_backend_synchronize(backend);
            n_inflight = 0;
        }
        uint8_t * st = staging + (size_t) n_inflight * (nb2 + nbu + nbd);
        memcpy(st, pg, nb2); memcpy(st + nb2, pu, nbu); memcpy(st + nb2 + nbu, pd, nbd);
        pg = (const char *) st; pu = pg + nb2; pd = pu + nbu;
        n_inflight++;
    }
    if (staging && backend) {
        // page-locked source: truly async on the device stream, synchronized by the caller
        ggml_backend_tensor_set_async(backend, t_gate, pg, (size_t) s * nb2, nb2);
        ggml_backend_tensor_set_async(backend, t_up,   pu, (size_t) s * nbu, nbu);
        ggml_backend_tensor_set_async(backend, t_down, pd, (size_t) s * nbd, nbd);
    } else {
        ggml_backend_tensor_set(t_gate, pg, (size_t) s * nb2, nb2);
        ggml_backend_tensor_set(t_up,   pu, (size_t) s * nbu, nbu);
        ggml_backend_tensor_set(t_down, pd, (size_t) s * nbd, nbd);
    }
    bytes_h2d += nb2 + nbu + nbd;
    const size_t gid = (size_t) il * n_expert + e;
    slot_of[gid] = s; gid_of[s] = (int32_t) gid;
}

void llama_expert_cache::record(int il, const int32_t * ids, int64_t k, int64_t n_tok, size_t nb_ids) {
    if (!warm_enabled) return;
    if (il == first_covered) {
        for (float & c : use_count) c *= 0.5f;                  // new ubatch: older prompt parts count less
    }
    float * cnt = use_count.data() + (size_t) il * n_expert;
    for (int64_t t = 0; t < n_tok; ++t) {
        const int32_t * row = (const int32_t *) ((const char *) ids + t * nb_ids);
        for (int64_t j = 0; j < k; ++j) {
            const int32_t e = row[j];
            if (e >= 0 && e < n_expert) cnt[e] += 1.0f;
        }
    }
    warm_pending = true;
}

void llama_expert_cache::warm_from_counts() {
    warm_pending = false;
    if (n_covered == 0) return;
    const int per_layer_budget = (int) (slots / (uint32_t) n_covered);
    std::vector<int32_t> order(n_expert);
    int n_loaded = 0, n_inflight = 0;
    for (int il = 0; il < n_layer; ++il) {
        if (!per_layer[il].gate) continue;
        const float * cnt = use_count.data() + (size_t) il * n_expert;
        for (int e = 0; e < n_expert; ++e) order[e] = e;
        const int m = std::min(per_layer_budget, n_expert);
        std::partial_sort(order.begin(), order.begin() + m, order.end(), [cnt](int32_t a, int32_t b) { return cnt[a] > cnt[b]; });
        for (int i = 0; i < m; ++i) {
            const int32_t e = order[i];
            if (cnt[e] <= 0.0f) break;
            const size_t gid = (size_t) il * n_expert + e;
            int32_t s = slot_of[gid];
            if (s >= 0) { touch(s); continue; }
            s = take_slot();
            load_expert(il, e, s, n_inflight);
            n_loaded++;
        }
    }
    if (staging && backend && n_inflight > 0) ggml_backend_synchronize(backend);
    LLAMA_LOG_INFO("%s: warmed the expert cache with %d experts from the prompt's routing\n", __func__, n_loaded);
    std::fill(use_count.begin(), use_count.end(), 0.0f);
}

void llama_expert_cache::record_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) return;
    auto * src = (layer_src *) userdata;
    GGML_ASSERT(a->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32);
    src->cache->record(src->il, (const int32_t *) a->data, a->ne[0], a->ne[1], a->nb[1]);
    for (int64_t t = 0; t < a->ne[1]; ++t) {     // identity output: the graph consumes it so the op is scheduled
        memcpy((char *) dst->data + t * dst->nb[1], (const char *) a->data + t * a->nb[1], a->ne[0] * sizeof(int32_t));
    }
}

void llama_expert_cache::prepare(int il, const int32_t * ids, int64_t k, int64_t n_tok, size_t nb_ids, int32_t * out, size_t nb_out) {
    if (warm_pending) {
        warm_from_counts();
    }
    const auto & src = per_layer[il];
    GGML_ASSERT(src.gate && "layer not covered by the expert cache");
    std::vector<int32_t> miss_e;
    for (int64_t t = 0; t < n_tok; ++t) {
        const int32_t * row = (const int32_t *) ((const char *) ids + t * nb_ids);
        int32_t * orow = (int32_t *) ((char *) out + t * nb_out);
        miss_e.clear();
        // first pass: hits are protected (moved to the MRU end) before any eviction
        for (int64_t j = 0; j < k; ++j) {
            const int32_t e = row[j];
            GGML_ASSERT(e >= 0 && e < n_expert);
            const int32_t s = slot_of[(size_t) il * n_expert + e];
            if (s >= 0) {
                touch(s); hits++; orow[j] = s;
            } else {
                miss_e.push_back((int32_t) j); orow[j] = -1;
            }
        }
        int n_inflight = 0;
        for (int32_t j : miss_e) {
            const int32_t e = row[j];
            const size_t gid = (size_t) il * n_expert + e;
            int32_t s = slot_of[gid];             // the same expert may appear twice in a row
            if (s < 0) {
                s = take_slot();
                load_expert(il, e, s, n_inflight);
                misses++;
            }
            orow[j] = s;
        }
        if (staging && backend && n_inflight > 0) {
            ggml_backend_synchronize(backend);      // the staging area is reused by the next token / layer
        }
    }
}

void llama_expert_cache::prepare_op(ggml_tensor * dst, const ggml_tensor * a, int ith, int nth, void * userdata) {
    GGML_UNUSED(nth);
    if (ith != 0) {
        return;
    }
    auto * src = (layer_src *) userdata;
    GGML_ASSERT(a->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32);
    GGML_ASSERT(a->ne[0] == dst->ne[0] && a->ne[1] == dst->ne[1]);
    src->cache->prepare(src->il, (const int32_t *) a->data, a->ne[0], a->ne[1], a->nb[1], (int32_t *) dst->data, dst->nb[1]);
}
