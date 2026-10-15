// src/poly/mmq.cu - see the header.  The kernels are compiled from the llama.cpp checkout the build already takes
// ggml from; src/prefill/ggml_cuda_host.cu supplies the host symbols of ggml-cuda.cu they use.
#include "mmq.hpp"

#include "common.cuh"
#include "mmq.cuh"
#include "quantize.cuh"

#include <climits>
#include <stdexcept>
#include <string>

namespace strata::poly {
namespace {

// The types with kernels in this build: one template instance each in CMakeLists.txt.
#define STRATA_POLY_MMQ_TYPES(X)                                                                                \
    X(Q4_0) X(Q4_1) X(Q5_0) X(Q5_1) X(Q8_0) X(Q2_K) X(Q3_K) X(Q4_K) X(Q5_K) X(Q6_K)                             \
    X(IQ1_S) X(IQ2_XXS) X(IQ2_XS) X(IQ2_S) X(IQ3_XXS) X(IQ3_S) X(IQ4_NL) X(IQ4_XS)

int64_t pad512(int64_t n) { return (n + 511) / 512 * 512; }

void ck(const char* what) {
    const cudaError_t e = cudaGetLastError();
    if (e != cudaSuccess) throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

}  // namespace

Mmq::Mmq() { ctx_ = new ggml_backend_cuda_context(ggml_cuda_get_device()); }

Mmq::~Mmq() { delete (ggml_backend_cuda_context*) ctx_; }

// mul_mat_q_case's own choice: a tile size whose configuration exists for this card and fits its shared memory.
// It aborts the process when there is none.
//
// The kernels step through a matrix MMQ_ITER_K columns at a time and read a whole step.  Where a row ends inside a
// step they read on into what lies behind it, and multiply that by the zeros quantize() pads the rows of x with.
// So what lies behind a matrix must be readable and must not be read as an infinite block scale: the engine keeps
// zero bytes there (kTail in runtime.hpp), as ggml-cuda does behind its tensors.
bool Mmq::fits(int type, int64_t w_rows, int64_t w_cols) {
    const ggml_type t = (ggml_type) type;
    switch (t) {
#define X(T) case GGML_TYPE_##T:
        STRATA_POLY_MMQ_TYPES(X)
#undef X
        break;
        default: return false;
    }
    if (w_cols % ggml_blck_size(t) != 0) return false;
    const bool fallback = w_rows % 128 != 0;
    const auto& dev = ggml_cuda_info().devices[ggml_cuda_get_device()];
    for (int J = 8; J <= 128; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, fallback, dev.cc, GGML_PREC_Q8);
        if (c.type != GGML_TYPE_COUNT && mmq_get_nbytes_shared(c, dev.cc) <= dev.smpbo) return true;
    }
    return false;
}

// the kernels read whole tiles, so the last rows need room behind them
size_t Mmq::q8_bytes(int64_t rows, int64_t cols) {
    return (size_t) rows * (size_t) (pad512(cols) / QK8_1_MMQ) * sizeof(block_q8_1_mmq) + 128 * sizeof(block_q8_1_mmq);
}

void Mmq::quantize(const float* x, const int32_t* ids, void* xq, int type, int64_t cols, int64_t ld, int64_t rows,
                   void* stream) {
    if (rows <= 0) return;
    quantize_mmq_q8_1_cuda(x, ids, xq, (ggml_type) type, cols, ld, rows * ld, rows * ld, pad512(cols), rows, 1, 1,
                           (cudaStream_t) stream);
    ck("mmq quantize");
}

void Mmq::mul(const void* w, int type, int64_t w_rows, int64_t w_cols, size_t stride, int n, const void* xq,
              const int32_t* bounds, const int32_t* ids, int64_t total_rows, int64_t max_rows, float* dst,
              int64_t ld_dst, void* stream) {
    if (n <= 0 || max_rows <= 0) return;
    const ggml_type t = (ggml_type) type;
    // the matrices are the channels of x; the tokens' rows are the columns of y and of dst
    mmq_args a = {};
    a.x = (const char*) w;
    a.type_x = t;
    a.y = (const int*) xq;
    a.ids_dst = ids;
    a.expert_bounds = bounds;
    a.dst = dst;
    a.ncols_x = w_cols;
    a.nrows_x = w_rows;
    a.stride_row_x = w_cols / ggml_blck_size(t);
    a.ncols_y = total_rows;
    a.ncols_dst = total_rows;
    a.nrows_dst = ld_dst;
    a.nchannels_x = n;
    a.nchannels_y = n;
    a.stride_channel_x = (int64_t) (stride / ggml_type_size(t));
    a.nsamples_x = 1;
    a.nsamples_y = 1;
    a.ncols_max = max_rows;
    // The tile width the kernel launches with, chosen the way ggml-cuda does: the config with the fewest tiles
    // over the widest expert's rows whose shared memory fits this card.
    const auto& dev = ggml_cuda_info().devices[ggml_cuda_get_device()];
    const bool fallback = ggml_cuda_mmq_needs_fallback(w_rows);
    int ntiles_best = INT_MAX;
    for (int J = 8; J <= 128 && ntiles_best > 1; J += 8) {
        const ggml_cuda_mmq_config c = ggml_cuda_mmq_get_config(t, J, fallback, dev.cc, GGML_PREC_Q8);
        if (c.type == GGML_TYPE_COUNT || mmq_get_nbytes_shared(c, dev.cc) > dev.smpbo) continue;
        const int ntiles = (int) ((max_rows + c.J - 1) / c.J);
        if (ntiles < ntiles_best) {
            a.J_best = J;
            ntiles_best = ntiles;
        }
    }
    if (a.J_best == 0) throw std::runtime_error("mmq: no tile configuration fits this GPU");

    auto& ctx = *(ggml_backend_cuda_context*) ctx_;
    const cudaStream_t s = (cudaStream_t) stream;
    switch (t) {
#define X(T) case GGML_TYPE_##T: mul_mat_q_case<GGML_TYPE_##T, GGML_PREC_Q8>(ctx, a, s); break;
        STRATA_POLY_MMQ_TYPES(X)
#undef X
        default: throw std::runtime_error("mmq: tensor type " + std::to_string(type) + " is not built");
    }
    ck("mmq product");
}

}  // namespace strata::poly
