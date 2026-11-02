// Adapted from ggml-org/llama.cpp 7182ef71e6d5dd0c8ea76486fac134fcd25025d4.
// Original GDN MMA kernel: PR #29353. MIT; see third_party/ggml/LICENSE.
#pragma once
#include <cuda_runtime.h>
#include <cstdint>
struct ggml_cuda_gdn_mma_args {
    const float * q;
    const float * k;
    const float * v;
    const float * g;
    const float * beta;
    const float * state;
    float *       dst;
    float *       state_out;

    int64_t H;
    int64_t H_k;
    int64_t n_tokens;
    int64_t n_seqs;
    int64_t rq3;
    int64_t sq1;
    int64_t sq2;
    int64_t sq3;
    int64_t sv1;
    int64_t sv2;
    int64_t sv3;
    int64_t sb1;
    int64_t sb2;
    int64_t sb3;
    float   scale;
    bool    eligible;
};


#include "strata/kernels/gdn_mma_policy.hpp"
namespace strata::kernels {
GdnMmaPath gdn_mma_device_path(int device, int heads);
bool gdn_mma_prepare(int device, GdnMmaPath path);
void gdn_mma_launch(const ggml_cuda_gdn_mma_args& args, GdnMmaPath path, cudaStream_t stream);
}
