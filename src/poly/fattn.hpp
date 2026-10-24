// src/poly/fattn.hpp - a prompt batch's attention with llama.cpp's flash-attention MMA kernel (fattn_mma.cuh, MIT)
// over a q8_0 or fp16 cache: heads of 256 values, 8 or 16 heads a group (Qwen3.6 / Ornith 122B).
// The keys go in blocks: under q8_0 each block's K and V are dequantized to fp16 in the scratch, the kernel attends
// to them with the causal mask computed from the positions, and the blocks' results are merged by their softmax max
// and sum.  An fp16 cache is read where it lies, but for the last block (the batch's own rows), which is copied.
// Opt-in: STRATA_FATTN=1 (Qwen35::attn).  fdec_q8() and fdec_f16() are the decode side (a few tokens), a kernel of
// this project's own: opt-in with STRATA_FDEC=1.
//
// Every call enqueues on the given CUDA stream.  Pointers are device pointers.
#pragma once

#include "strata/poly/kernels.hpp"

#include <cstddef>
#include <cstdint>

namespace strata::poly {

/// This build has the kernel for heads of `dim` values, n_head / kv_heads heads a group, on the current card, and a
/// scratch of scratch_floats floats holds a block for batches of up to `rows` tokens: fattn_q8() and fattn_f16()
/// then take every such batch.
bool fattn_fits(int dim, int n_head, int kv_heads, int rows, size_t scratch_floats);

/// Causal attention of the n tokens from position pos to the cache rows 0 to pos + n - 1, as attend_batch()
/// (no window, no ring): q and out n_head * dim floats a token.  scratch (scratch_floats floats) holds a block's
/// results and its K and V; false, with nothing enqueued, when it cannot hold a block of n + 32 rows.
bool fattn_q8(const float* q, Kvq8 kq8, Kvq8 vq8, int64_t pos, int n, float* out, int dim, int n_head, int kv_heads,
              float scale, float* scratch, size_t scratch_floats, void* stream);

/// fattn_q8() over an fp16 cache (keys and vals, kv_heads * dim halves a row).
bool fattn_f16(const float* q, const uint16_t* keys, const uint16_t* vals, int64_t pos, int n, float* out, int dim,
               int n_head, int kv_heads, float scale, float* scratch, size_t scratch_floats, void* stream);

/// This build has the decode kernel for this shape on the current card (Ampere or later), and a scratch of
/// scratch_floats floats holds its splits for `rows` cache rows: fdec_q8() and fdec_f16() then take every call up to
/// that many.
bool fdec_fits(int dim, int n_head, int kv_heads, int rows, size_t scratch_floats);

/// attend() for n tokens (fewer than kStageRows) with no window, ring or `whole`, over a q8_0 cache: token t
/// attends the rows 0 to pos + t, with the position from `at` on the card when it is given, and the launch sized
/// for `room` rows then (a recorded launch).  The keys go in splits of 1,024 rows on the tensor cores, read once
/// for up to 4 tokens (group 8) or 2 (group 16), and the splits are merged; a token gets the same bits alone or
/// with others. scratch holds the splits' results; false, with nothing enqueued, where fdec_fits() is false.
bool fdec_q8(const float* q, Kvq8 kq8, Kvq8 vq8, int64_t pos, int n, const long long* at, int room, float* out,
             int dim, int n_head, int kv_heads, float scale, float* scratch, size_t scratch_floats, void* stream);

/// fdec_q8() over an fp16 cache (keys and vals, kv_heads * dim halves a row), read where it lies.
bool fdec_f16(const float* q, const uint16_t* keys, const uint16_t* vals, int64_t pos, int n, const long long* at,
              int room, float* out, int dim, int n_head, int kv_heads, float scale, float* scratch,
              size_t scratch_floats, void* stream);

}  // namespace strata::poly
