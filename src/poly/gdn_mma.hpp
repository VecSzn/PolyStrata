// Opt-in CUDA GDN prefill; the stock gdn_step API and snapshot path stay unchanged.
#pragma once
#include "strata/kernels/gdn_mma.hpp"
namespace strata::poly {
using kernels::gdn_mma_available;
// False with inputs untouched when the shape, device or snapshot contract is unsupported.
bool gdn_mma_step(float* qkv, float* alpha, float* beta, const float* dt, const float* a, float* state,
                  const float* z, const float* norm, float* out, int dim, int heads, int key_heads, int rows,
                  float norm_eps, void* stream, bool snapshots = false);
}
