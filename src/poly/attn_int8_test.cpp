// src/poly/attn_int8_test.cpp - Mathematical parity test comparing FP16 vs INT8 KV attention on AMD gfx1100.
#include "strata/poly/kernels.hpp"

#include <cuda_runtime.h>
#include <cublas_v2.h>

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <vector>

namespace {

void ck(cudaError_t e, const char* msg) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "CUDA error in %s: %s\n", msg, cudaGetErrorString(e));
        std::exit(1);
    }
}

template <typename T>
T* dalloc(size_t n) {
    T* p = nullptr;
    ck(cudaMalloc(&p, n * sizeof(T) + 256), "cudaMalloc");
    ck(cudaMemset(p, 0, n * sizeof(T) + 256), "cudaMemset");
    return p;
}

}  // namespace

int main() {
    std::printf("=================================================================\n");
    std::printf(" PolyStrata AMD gfx1100: INT8 KV Attention Parity Verification   \n");
    std::printf("=================================================================\n\n");

    const int dim = 256;
    const int n_head = 16;
    const int kv_heads = 2;
    const float scale = 1.0f / std::sqrt((float) dim);

    cudaStream_t stream = nullptr;
    ck(cudaStreamCreate(&stream), "stream create");
    void* blas = strata::poly::blas_create(stream);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, 1.0f);

    const std::vector<int> test_lengths = {1, 16, 64, 256, 1024, 4096};
    bool all_passed = true;

    for (int L : test_lengths) {
        std::printf("--- Testing Single-Token Decode at Context Length L = %d ---\n", L);

        // Host buffers
        std::vector<float> h_q(n_head * dim);
        std::vector<float> h_k((size_t) L * kv_heads * dim);
        std::vector<float> h_v((size_t) L * kv_heads * dim);

        for (auto& x : h_q) x = dist(rng);
        for (auto& x : h_k) x = dist(rng);
        for (auto& x : h_v) x = dist(rng);

        // Device buffers
        float* d_q = dalloc<float>(n_head * dim);
        float* d_k_raw = dalloc<float>((size_t) L * kv_heads * dim);
        float* d_v_raw = dalloc<float>((size_t) L * kv_heads * dim);

        ck(cudaMemcpyAsync(d_q, h_q.data(), h_q.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy q");
        ck(cudaMemcpyAsync(d_k_raw, h_k.data(), h_k.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy k");
        ck(cudaMemcpyAsync(d_v_raw, h_v.data(), h_v.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy v");

        // FP16 cache
        uint16_t* d_k_f16 = dalloc<uint16_t>((size_t) L * kv_heads * dim);
        uint16_t* d_v_f16 = dalloc<uint16_t>((size_t) L * kv_heads * dim);

        // INT8 cache
        const int G = dim / 64;
        int8_t* d_k_q8 = dalloc<int8_t>((size_t) L * kv_heads * dim);
        int8_t* d_v_q8 = dalloc<int8_t>((size_t) L * kv_heads * dim);
        uint16_t* d_k_scale = dalloc<uint16_t>((size_t) L * kv_heads * G);
        uint16_t* d_v_scale = dalloc<uint16_t>((size_t) L * kv_heads * G);

        // Scratch and output
        float* d_score = dalloc<float>((size_t) L * n_head);
        float* d_out_f16 = dalloc<float>(n_head * dim);
        float* d_out_q8 = dalloc<float>(n_head * dim);

        // 1. Populate FP16 cache & run FP16 attention
        strata::poly::store_f16(d_k_raw, d_k_f16, kv_heads * dim, 0, L, stream);
        strata::poly::store_f16(d_v_raw, d_v_f16, kv_heads * dim, 0, L, stream);
        strata::poly::attend(d_q, d_k_f16, d_v_f16, L - 1, 1, d_score, 1, d_out_f16,
                             dim, n_head, kv_heads, scale, stream, 0, 0, false, nullptr, 0);

        // 2. Populate INT8 cache & run INT8 attention
        strata::poly::store_int8(d_k_raw, d_k_q8, d_k_scale, kv_heads, dim, 0, L, stream);
        strata::poly::store_int8(d_v_raw, d_v_q8, d_v_scale, kv_heads, dim, 0, L, stream);
        strata::poly::attend_int8(d_q, d_k_q8, d_k_scale, d_v_q8, d_v_scale, L - 1, 1, d_score, 1, d_out_q8,
                                  dim, n_head, kv_heads, scale, stream, 0, 0, false, nullptr, 0);

        ck(cudaStreamSynchronize(stream), "sync decode");

        // Copy outputs back to host
        std::vector<float> h_out_f16(n_head * dim), h_out_q8(n_head * dim);
        ck(cudaMemcpy(h_out_f16.data(), d_out_f16, h_out_f16.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out f16");
        ck(cudaMemcpy(h_out_q8.data(), d_out_q8, h_out_q8.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy out q8");

        double min_cos = 1.0;
        double max_rel_err = 0.0;
        double max_abs_diff = 0.0;

        for (int h = 0; h < n_head; ++h) {
            const float* u = h_out_f16.data() + h * dim;
            const float* v = h_out_q8.data() + h * dim;

            double dot = 0.0, norm_u = 0.0, norm_v = 0.0, l2_diff = 0.0;
            for (int d = 0; d < dim; ++d) {
                dot += (double) u[d] * v[d];
                norm_u += (double) u[d] * u[d];
                norm_v += (double) v[d] * v[d];
                double diff = std::fabs((double) u[d] - v[d]);
                max_abs_diff = std::max(max_abs_diff, diff);
                l2_diff += diff * diff;
            }
            norm_u = std::sqrt(norm_u);
            norm_v = std::sqrt(norm_v);
            l2_diff = std::sqrt(l2_diff);

            double cos_sim = (norm_u > 0 && norm_v > 0) ? (dot / (norm_u * norm_v)) : 1.0;
            double rel_err = (norm_u > 0) ? (l2_diff / norm_u) : 0.0;

            min_cos = std::min(min_cos, cos_sim);
            max_rel_err = std::max(max_rel_err, rel_err);
        }

        std::printf("  Decode: Min Cosine Similarity: %.6f (threshold >= 0.9990)\n", min_cos);
        std::printf("  Decode: Max Relative Error:    %.6f (threshold <= 0.0200)\n", max_rel_err);
        std::printf("  Decode: Max Absolute Diff:     %.6f\n", max_abs_diff);

        bool decode_ok = (min_cos >= 0.9990 && max_rel_err <= 0.0200);
        if (!decode_ok) {
            std::printf("  [FAIL] Decode parity failed at length %d!\n", L);
            all_passed = false;
        } else {
            std::printf("  [PASS] Decode attention parity verified.\n");
        }

        // 3. Test prompt batch prefill parity if L <= 256
        if (L >= 16 && L <= 256) {
            std::printf("--- Testing Prompt Batch Prefill Parity at Batch = %d tokens ---\n", L);
            std::vector<float> h_q_batch((size_t) L * n_head * dim);
            for (auto& x : h_q_batch) x = dist(rng);
            float* d_q_batch = dalloc<float>((size_t) L * n_head * dim);
            ck(cudaMemcpyAsync(d_q_batch, h_q_batch.data(), h_q_batch.size() * sizeof(float), cudaMemcpyHostToDevice, stream), "copy q batch");

            size_t score_floats = (size_t) 16 * n_head * L;
            if (score_floats < (size_t) 2 * n_head) score_floats = 2 * n_head;
            float* d_batch_score = dalloc<float>(score_floats);
            size_t work_floats = (size_t) L * n_head * (dim + dim / 2 + 3) + 1;
            float* d_work = dalloc<float>(work_floats);
            uint16_t* d_kdequant = dalloc<uint16_t>(2048 * kv_heads * dim);
            uint16_t* d_vdequant = dalloc<uint16_t>(2048 * kv_heads * dim);
            float* d_batch_out_f16 = dalloc<float>((size_t) L * n_head * dim);
            float* d_batch_out_q8 = dalloc<float>((size_t) L * n_head * dim);

            strata::poly::attend_batch(blas, d_q_batch, d_k_f16, d_v_f16, 0, L, d_batch_score,
                                       score_floats, d_work, d_batch_out_f16, dim, n_head, kv_heads, scale, stream);

            strata::poly::attend_batch_int8(blas, d_q_batch, d_k_q8, d_k_scale, d_v_q8, d_v_scale, 0, L,
                                            d_batch_score, score_floats, d_work, d_kdequant, d_vdequant,
                                            d_batch_out_q8, dim, n_head, kv_heads, scale, stream);

            ck(cudaStreamSynchronize(stream), "sync batch");

            std::vector<float> h_b_f16((size_t) L * n_head * dim), h_b_q8((size_t) L * n_head * dim);
            ck(cudaMemcpy(h_b_f16.data(), d_batch_out_f16, h_b_f16.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy b f16");
            ck(cudaMemcpy(h_b_q8.data(), d_batch_out_q8, h_b_q8.size() * sizeof(float), cudaMemcpyDeviceToHost), "copy b q8");

            double b_min_cos = 1.0;
            double b_max_rel = 0.0;
            for (size_t idx = 0; idx < (size_t) L * n_head; ++idx) {
                const float* u = h_b_f16.data() + idx * dim;
                const float* v = h_b_q8.data() + idx * dim;
                double dot = 0.0, nu = 0.0, nv = 0.0, diff = 0.0;
                for (int d = 0; d < dim; ++d) {
                    dot += (double) u[d] * v[d];
                    nu += (double) u[d] * u[d];
                    nv += (double) v[d] * v[d];
                    double dval = std::fabs((double) u[d] - v[d]);
                    diff += dval * dval;
                }
                nu = std::sqrt(nu); nv = std::sqrt(nv); diff = std::sqrt(diff);
                double c = (nu > 0 && nv > 0) ? (dot / (nu * nv)) : 1.0;
                double r = (nu > 0) ? (diff / nu) : 0.0;
                b_min_cos = std::min(b_min_cos, c);
                b_max_rel = std::max(b_max_rel, r);
            }
            std::printf("  Batch:  Min Cosine Similarity: %.6f (threshold >= 0.9990)\n", b_min_cos);
            std::printf("  Batch:  Max Relative Error:    %.6f (threshold <= 0.0200)\n", b_max_rel);
            bool batch_ok = (b_min_cos >= 0.9990 && b_max_rel <= 0.0200);
            if (!batch_ok) {
                std::printf("  [FAIL] Batch prefill parity failed at length %d!\n", L);
                all_passed = false;
            } else {
                std::printf("  [PASS] Batch prefill parity verified.\n");
            }

            cudaFree(d_q_batch);
            cudaFree(d_batch_score);
            cudaFree(d_work);
            cudaFree(d_kdequant);
            cudaFree(d_vdequant);
            cudaFree(d_batch_out_f16);
            cudaFree(d_batch_out_q8);
        }

        cudaFree(d_q);
        cudaFree(d_k_raw);
        cudaFree(d_v_raw);
        cudaFree(d_k_f16);
        cudaFree(d_v_f16);
        cudaFree(d_k_q8);
        cudaFree(d_v_q8);
        cudaFree(d_k_scale);
        cudaFree(d_v_scale);
        cudaFree(d_score);
        cudaFree(d_out_f16);
        cudaFree(d_out_q8);
        std::printf("\n");
    }

    strata::poly::blas_destroy(blas);
    cudaStreamDestroy(stream);

    std::printf("=================================================================\n");
    if (all_passed) {
        std::printf(" RESULT: ALL ATTENTION PARITY TESTS PASSED (Cosine Sim >= 0.999)\n");
    } else {
        std::printf(" RESULT: SOME ATTENTION PARITY TESTS FAILED\n");
    }
    std::printf("=================================================================\n");

    return all_passed ? 0 : 1;
}
