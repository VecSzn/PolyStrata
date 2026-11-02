// src/poly/gemm_small_test.cu - gemm_f32() for one group of up to 256 outputs (gemm_small_kernel, gemm_tile_kernel past
// 64 of them) against the same products on the host: each output's sum of fmaf in the inputs' order, so the bits
// must be the same.
//
// usage: poly_gemm_small_test   (exit 0: every output the same)
#include "strata/poly/kernels.hpp"

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace {

using namespace strata::poly;

int check(cublasHandle_t h, int n_in, int n_out, int rows, std::mt19937& rng) {
    std::normal_distribution<float> nd(0.f, 1.f);
    std::vector<float> w(size_t(n_in) * n_out), x(size_t(n_in) * rows), want(size_t(n_out) * rows),
        got(want.size());
    for (float& v : w) v = 0.05f * nd(rng);
    for (float& v : x) v = nd(rng);
    for (int t = 0; t < rows; ++t)
        for (int o = 0; o < n_out; ++o) {
            float acc = 0.f;
            for (int k = 0; k < n_in; ++k) acc = std::fma(x[size_t(t) * n_in + k], w[size_t(o) * n_in + k], acc);
            want[size_t(t) * n_out + o] = acc;
        }
    float *dw = nullptr, *dx = nullptr, *dy = nullptr;
    cudaMalloc(&dw, w.size() * sizeof(float));
    cudaMalloc(&dx, x.size() * sizeof(float));
    cudaMalloc(&dy, got.size() * sizeof(float));
    cudaMemcpy(dw, w.data(), w.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(dx, x.data(), x.size() * sizeof(float), cudaMemcpyHostToDevice);
    gemm_f32(h, dw, dx, dy, n_in, n_out, n_out, rows);
    cudaMemcpy(got.data(), dy, got.size() * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(dw), cudaFree(dx), cudaFree(dy);
    const bool ok = !std::memcmp(want.data(), got.data(), want.size() * sizeof(float));
    std::printf("%s n_in %4d n_out %3d rows %4d\n", ok ? "ok  " : "FAIL", n_in, n_out, rows);
    return ok ? 0 : 1;
}

}  // namespace

int main() {
    cublasHandle_t h = nullptr;
    if (cublasCreate(&h) != CUBLAS_STATUS_SUCCESS) {
        std::printf("no cuBLAS\n");
        return 1;
    }
    std::mt19937 rng(3);
    int bad = 0;
    for (int n_out : {32, 64, 100, 128, 256})
        for (int rows : {32, 37, 1024, 2048}) bad += check(h, 2048, n_out, rows, rng);
    bad += check(h, 2000, 64, 45, rng);    // inputs that end inside a step
    bad += check(h, 2000, 256, 45, rng);   // and of gemm_tile_kernel's
    cublasDestroy(h);
    if (const cudaError_t e = cudaGetLastError(); e != cudaSuccess) {
        std::printf("CUDA: %s\n", cudaGetErrorString(e));
        return 1;
    }
    std::printf(bad ? "%d failed\n" : "all passed\n", bad);
    return bad ? 1 : 0;
}
