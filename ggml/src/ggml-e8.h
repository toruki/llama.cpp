#pragma once

// E8-lattice codec shared by the CPU reference implementation (ggml-quants.c) and the CUDA
// kernels (ggml-cuda/e8.cuh). Ported from ninfer's src/ops/kernel/e8_lattice.cuh and
// e8_root_codec.cuh (design credit: UDPSendToFailed/ninfer-4090, Don-Chad/ninfer-3090 lineage).
//
// Both backends use the *scalar* algorithms below, so the CPU and CUDA quantizers agree
// bit-for-bit (ninfer's warp-cooperative variants can disagree on exact ties).
//
// This header must be included AFTER ggml-common.h's IMPL section: it reads the
// e8_root_i8x8 / e8_axis_i8x8 / e8_radius_scale tables defined there.

#include <math.h>
#include <stdint.h>

#if defined(__CUDACC__) || defined(__HIPCC__)
#define GGML_E8_FN static __device__ __forceinline__
#else
#define GGML_E8_FN static inline
#endif

// Nearest point of E8 = D8 U (D8 + 1/2), by the Conway-Sloane algorithm: round to the
// nearest point of each coset (fixing the parity of the coordinate sum by flipping the
// worst-rounded coordinate), then keep whichever is closer.
GGML_E8_FN void ggml_e8_project_8d(const float x[8], float out[8]) {
    float f_x[8];
    int   sum_f     = 0;
    float max_err   = -1.0f;
    int   worst_dim = 0;

    for (int i = 0; i < 8; ++i) {
        f_x[i] = rintf(x[i]);
        sum_f += (int) f_x[i];
        const float err = fabsf(x[i] - f_x[i]);
        if (err > max_err) { max_err = err; worst_dim = i; }
    }

    float d8[8];
    for (int i = 0; i < 8; ++i) { d8[i] = f_x[i]; }
    if ((sum_f & 1) != 0) {
        d8[worst_dim] += (x[worst_dim] >= f_x[worst_dim]) ? 1.0f : -1.0f;
    }

    float f_shift[8];
    int   sum_shift       = 0;
    float max_err_shift   = -1.0f;
    int   worst_shift_dim = 0;

    for (int i = 0; i < 8; ++i) {
        const float xs = x[i] - 0.5f;
        f_shift[i] = rintf(xs);
        sum_shift += (int) f_shift[i];
        const float err = fabsf(xs - f_shift[i]);
        if (err > max_err_shift) { max_err_shift = err; worst_shift_dim = i; }
    }

    float coset1[8];
    for (int i = 0; i < 8; ++i) { coset1[i] = f_shift[i] + 0.5f; }
    if ((sum_shift & 1) != 0) {
        coset1[worst_shift_dim] +=
            ((x[worst_shift_dim] - 0.5f) >= f_shift[worst_shift_dim]) ? 1.0f : -1.0f;
    }

    float dist_d8 = 0.0f, dist_coset1 = 0.0f;
    for (int i = 0; i < 8; ++i) {
        const float e0 = x[i] - d8[i];
        const float e1 = x[i] - coset1[i];
        dist_d8     += e0*e0;
        dist_coset1 += e1*e1;
    }

    for (int i = 0; i < 8; ++i) {
        out[i] = (dist_d8 <= dist_coset1) ? d8[i] : coset1[i];
    }
}

// Nearest of the 240 E8 minimal vectors ("roots") to the unit vector u; also returns the
// root itself. Type A (codes 0..111) are the permutations of (+-1,+-1,0,...,0); type B
// (codes 112..239) are 1/2 (+-1)^8 with an even number of minus signs.
GGML_E8_FN uint8_t ggml_e8_quantize_root_8d(const float u[8], float out_root[8]) {
    float abs_u[8];
    int   signs[8];
    for (int i = 0; i < 8; ++i) {
        abs_u[i] = fabsf(u[i]);
        signs[i] = (u[i] >= 0.0f) ? 1 : -1;
    }

    int   best_i = 0, best_j = 1;
    int   best_pair_idx = 0, pair_idx = 0;
    float max_pair_sum  = -1.0f;
    for (int i = 0; i < 7; ++i) {
        for (int j = i + 1; j < 8; ++j) {
            const float sum = abs_u[i] + abs_u[j];
            if (sum > max_pair_sum) {
                max_pair_sum  = sum;
                best_i        = i;
                best_j        = j;
                best_pair_idx = pair_idx;
            }
            ++pair_idx;
        }
    }

    const float   best_type_a_score = max_pair_sum;
    const int     s_i_bit           = (signs[best_i] > 0) ? 1 : 0;
    const int     s_j_bit           = (signs[best_j] > 0) ? 1 : 0;
    const uint8_t type_a_code       = (uint8_t) (best_pair_idx*4 + (s_i_bit << 1) + s_j_bit);

    float sum_abs = 0.0f, min_abs = 1e9f;
    int   minus_count = 0, min_idx = 0;
    int   b_signs[8];
    for (int i = 0; i < 8; ++i) {
        sum_abs += abs_u[i];
        if (u[i] >= 0.0f) {
            b_signs[i] = 1;
        } else {
            b_signs[i] = -1;
            ++minus_count;
        }
        if (abs_u[i] < min_abs) { min_abs = abs_u[i]; min_idx = i; }
    }

    float best_type_b_score = 0.5f*sum_abs;
    if ((minus_count & 1) != 0) {
        best_type_b_score -= min_abs;
        b_signs[min_idx]   = -b_signs[min_idx];
    }

    if (best_type_a_score >= best_type_b_score) {
        for (int i = 0; i < 8; ++i) { out_root[i] = 0.0f; }
        out_root[best_i] = (signs[best_i] > 0) ?  1.0f : -1.0f;
        out_root[best_j] = (signs[best_j] > 0) ?  1.0f : -1.0f;
        return type_a_code;
    }

    uint8_t type_b_code = 0;
    for (int i = 0; i < 7; ++i) {
        if (b_signs[i] > 0) { type_b_code |= (uint8_t) (1 << i); }
    }
    for (int i = 0; i < 8; ++i) { out_root[i] = (b_signs[i] > 0) ? 0.5f : -0.5f; }
    return (uint8_t) (112 + type_b_code);
}

// Cylinder factorization of one rotated 8-vector against the group scale `ks`:
//   root     = nearest E8 root of the unit vector
//   rad_idx  = 4-bit logarithmic radius, 0 meaning "this subvector is zero"
//   axis_idx = signed dominant dimension of the residual after removing the root component
GGML_E8_FN void ggml_e8_encode_cylinder_8d(const float rot[8], float ks,
                                           uint8_t * out_root, uint8_t * out_rad_axis) {
    float norm_sq = 0.0f;
    for (int i = 0; i < 8; ++i) { norm_sq += rot[i]*rot[i]; }
    const float out_norm = sqrtf(norm_sq);
    const float r_rel    = out_norm / (ks*2.82842712474619f + 1e-8f); // ks * sqrt(8)

    uint32_t rad_idx = 0;
    if (r_rel >= 0.08f) {
        const float log_val = 3.0f*(logf(r_rel)*1.4426950408889634f) + 8.0f; // 3*log2(r_rel) + 8
        const int   q_rad   = (int) rintf(log_val);
        rad_idx = (uint32_t) (q_rad < 1 ? 1 : (q_rad > 15 ? 15 : q_rad));
    }

    if (rad_idx == 0) {
        *out_root     = 0;
        *out_rad_axis = 0;
        return;
    }

    const float inv_norm = 1.0f/(out_norm + 1e-8f);
    float u[8];
    for (int i = 0; i < 8; ++i) { u[i] = rot[i]*inv_norm; }

    float v1[8];
    *out_root = ggml_e8_quantize_root_8d(u, v1);

    const float kInvSqrt2 = 0.7071067811865475f;
    float dot_u_v1 = 0.0f;
    for (int i = 0; i < 8; ++i) { dot_u_v1 += u[i]*v1[i]*kInvSqrt2; }

    float res[8];
    for (int i = 0; i < 8; ++i) { res[i] = u[i] - dot_u_v1*(v1[i]*kInvSqrt2); }

    int   best_dim    = 0;
    float max_abs_res = fabsf(res[0]);
    for (int i = 1; i < 8; ++i) {
        const float a = fabsf(res[i]);
        if (a > max_abs_res) { max_abs_res = a; best_dim = i; }
    }

    const uint32_t sign_bit = (res[best_dim] >= 0.0f) ? 0u : 1u;
    const uint32_t axis_idx = ((uint32_t) best_dim << 1) | sign_bit;

    *out_rad_axis = (uint8_t) ((rad_idx << 4) | (axis_idx & 0x0Fu));
}

// Table decode of one (root, rad_axis) pair into eight int8 codes. The reconstructed value
// is code[i] * d, where d is the block's fp16 group scale.
GGML_E8_FN void ggml_e8_root_decode_8d(uint8_t root, uint8_t rad_axis, int8_t out[8]) {
    const uint32_t rad_idx  = (uint32_t) rad_axis >> 4;
    const uint32_t axis_idx = (uint32_t) rad_axis & 0x0Fu;

    if (rad_idx == 0) {
        for (int i = 0; i < 8; ++i) { out[i] = 0; }
        return;
    }

    const uint64_t w_root = e8_root_i8x8[root];
    const uint64_t w_axis = e8_axis_i8x8[axis_idx];
    const float    scale  = e8_radius_scale[rad_idx];

    for (int i = 0; i < 8; ++i) {
        const int r = (int) (int8_t) ((w_root >> (8*i)) & 0xffu);
        const int a = (int) (int8_t) ((w_axis >> (8*i)) & 0xffu);
        out[i] = (int8_t) (int) rintf((float) (r + a) * scale);
    }
}
