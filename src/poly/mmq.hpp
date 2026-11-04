// src/poly/mmq.hpp - llama.cpp's quantized matrix products (ggml-cuda mmq.cuh, MIT) for the prompt path: the rows
// of many tokens against matrices that stay in their GGUF type.  Built for the types the model's GGUFs keep
// their matrices in: the K-quants, the I-quants and the older Q4, Q5 and Q8.
//
// Every call enqueues on the given CUDA stream.  Pointers are device pointers.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::poly {

class Mmq {
public:
    Mmq();
    ~Mmq();
    Mmq(const Mmq&) = delete;
    Mmq& operator=(const Mmq&) = delete;

    /// This build multiplies ggml_type, and the card has a tile for a matrix of [w_rows, w_cols] of it.
    static bool fits(int ggml_type, int64_t w_rows, int64_t w_cols);
    /// Bytes of `rows` rows of `cols` values as quantize() writes them.
    static size_t q8_bytes(int64_t rows, int64_t cols);
    /// quantize() keeps the rows' block sums in halves for a matrix of ggml_type (guard_half_sums()).
    static bool half_sums(int ggml_type);

    /// The rows of a product with a matrix of ggml_type: row i of xq is row ids[i] of x, or row i when ids is null.
    /// x has ld floats a row, 16-byte aligned.  The rows are interleaved in xq, so a product takes the xq of one
    /// call whole.
    static void quantize(const float* x, const int32_t* ids, void* xq, int ggml_type, int64_t cols, int64_t ld,
                         int64_t rows, void* stream);

    /// One launch over n matrices of [w_rows, w_cols], `stride` bytes apart from w: matrix e times the rows
    /// [bounds[e], bounds[e + 1]) of xq, which quantize() wrote with total_rows rows.  Row i's product goes to row
    /// ids[i] of dst, ld_dst floats a row.  bounds (n + 1 values) and ids are on the device.  max_rows is the most
    /// rows one of the matrices has.  With one matrix, bounds and ids may both be null: it takes every row, and
    /// row i goes to row i.  Throws std::runtime_error.
    void mul(const void* w, int ggml_type, int64_t w_rows, int64_t w_cols, size_t stride, int n, const void* xq,
             const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
             int64_t ld_dst, void* stream);

private:
    void* ctx_ = nullptr;   ///< llama.cpp's launch context: a small scratch pool
};

}  // namespace strata::poly
