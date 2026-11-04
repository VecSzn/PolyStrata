// Shared CUDA scalar-gated delta recurrence. No model or engine dependencies.
#pragma once
#include <cstdint>

namespace strata::kernels {
// Prepared FP32 device tensors for one sequence. Heads are contiguous dim-wide rows;
// row strides below are in float elements. Q and K have the same strides, and may
// share an interleaved Q/K/V allocation or use separate allocations.
// Value head h consumes Q/K head h % key_heads. The current eligibility envelope
// is dim=128, key_heads=16, heads=16/32/48/64 and rows>=64, on supported devices.
// log_decay and beta are already activated scalars per (token, value head), not
// raw projection outputs or vector-valued KDA decay. query_scale multiplies Q.
// State is [value head][value column][key dimension]; output is [token][head][dim].
// state_out may equal state; other output/input overlaps are not supported.
// Caller owns correctly sized, aligned allocations and ordered stream lifetime.
// Only final state is produced: callers needing intermediate snapshots or graph
// recording must retain their existing path. No normalization/gating epilogue runs.
struct GdnMmaArgs {
    const float* q = nullptr;
    const float* k = nullptr;
    const float* v = nullptr;
    const float* log_decay = nullptr;
    const float* beta = nullptr;
    const float* state = nullptr;
    float* output = nullptr;
    float* state_out = nullptr;
    int dim = 0, heads = 0, key_heads = 0, rows = 0;
    int64_t qk_row_stride = 0, v_row_stride = 0, gate_row_stride = 0;
    float query_scale = 1.f;
};

// Pure host contract check, without touching device buffers or initializing CUDA.
bool gdn_mma_supported(const GdnMmaArgs& args);
// Checks/prepares the current device for this geometry; does not inspect tensors.
bool gdn_mma_available(int dim, int heads, int key_heads);
// False before any tensor mutation when the contract or hardware is unsupported.
// Accepted calls enqueue on stream (CUDA stream as void*); launch errors throw.
// Async errors must be checked by the caller at its normal synchronization boundary.
bool gdn_mma_try(const GdnMmaArgs& args, void* stream);
} // namespace strata::kernels
