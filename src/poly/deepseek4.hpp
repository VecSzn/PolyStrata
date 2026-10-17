// src/poly/deepseek4.hpp - DeepSeek V4 Flash (llama.cpp architecture "deepseek4").
//
// 43 layers with four residual streams joined by hyper-connections, as GLM 5.3 Flash's.  A layer's attention has
// one key head of 512 values that is its value too, and reads the last 128 positions' rows as they are.  Most
// layers also read rows that a compressor made of several positions each: 128 to one, all of them, or 4 to one,
// as many of those as an indexer picks.  Every layer's FFN is routed experts beside a shared one; the first layers
// route by the token alone.
#pragma once

#include "engine.hpp"
#include "strata/poly/kernels.hpp"

#include <vector>

namespace strata::poly {

class DeepSeek4 final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override { return 0; }
    size_t extra_bytes() const override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override;
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void shift_rows(ptrdiff_t by) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    int head_key(int il, int n) const override;

private:
    /// A compressor: its weights, and the rows of the positions whose block is not done yet, in two rings.
    struct Squeeze {
        Dev wkv, wgate, ape, norm;
        float* kv = nullptr;      ///< ring_ rows of width values
        float* score = nullptr;
        int dim = 0, width = 0;   ///< a compressed row's values; a position's (2 * dim with the overlap)
    };
    /// A layer's weights beside the ones every architecture has (Layer), and its caches.
    struct Block {
        int ratio = 0;   ///< positions to a compressed row: 0 (none), 4 or 128
        Dev hc_attn_fn, hc_attn_scale, hc_attn_base, hc_ffn_fn, hc_ffn_scale, hc_ffn_base;
        Dev q_a, q_a_norm, q_b, wkv, kv_norm, wo_a, wo_b, sinks;
        Squeeze comp, idx;   ///< the attention's compressor; the indexer's (4 to one only)
        Dev idx_q_b, idx_proj;
        uint16_t* raw = nullptr;    ///< raw_ring_ rows of the last positions, fp16
        uint16_t* rows = nullptr;   ///< a compressed row a block of ratio positions, fp16
        uint16_t* keys = nullptr;   ///< the indexer's key of each 4-to-1 row, fp16
        int ring = 0;               ///< the compressors' ring rows
        Rope rope;                  ///< the queries' and keys' turn
        std::vector<float> probs_b;      ///< what the routing adds to each expert's score before it picks
        Dev probs_b_dev;                 ///< the same on the card, when the file has it
        std::vector<int32_t> by_token;   ///< a hash layer's experts, n_used_ a token; empty in the others
    };

    bool batch_heads() const override { return true; }
    void attention(Block& B, int64_t pos, int n);
    void squeeze(Block& B, Squeeze& S, int64_t pos, int n);
    void out_proj(Block& B, int n);
    void pair(const Dev& a, const Dev& b, float* ya, float* yb, int n);
    size_t layer_extra(const Block& B) const;
    int keep_rows(const Block& B) const;

    std::vector<Block> blocks_;   ///< one a layer of layers_
    std::vector<int> ratio_;      ///< each layer's compress ratio, from the metadata

    int n_head_ = 0, head_dim_ = 0, n_rot_ = 0, window_ = 0;
    int q_rank_ = 0, o_groups_ = 0, o_rank_ = 0;
    int idx_heads_ = 0, idx_dim_ = 0, idx_top_k_ = 0;
    int sinkhorn_ = 0, hash_layers_ = 0;
    int raw_ring_ = 0;   ///< rows of a layer's raw ring: the window and a call's tokens
    float hc_eps_ = 0, w_scale_ = 1;
    bool w_norm_ = false;
    Rope rope_raw_;      ///< the turn of the layers without compressed rows
    Rope rope_comp_;     ///< the compressed layers', the compressed rows' and the indexer's turn
    int score_stride_ = 0, score_rows_ = 0;   ///< one token's indexer scores, and the tokens score_ holds at once
    Dev hc_head_fn_, hc_head_scale_, hc_head_base_;

    // device scratch
    float *mix_ = nullptr, *hcw_ = nullptr;
    float *t1_ = nullptr, *qr_ = nullptr, *q_ = nullptr, *kv_ = nullptr, *att_ = nullptr, *ag_ = nullptr,
          *oa_ = nullptr, *ob_ = nullptr;
    float *ckv_ = nullptr, *cscore_ = nullptr;
    float *iq_ = nullptr, *iw_ = nullptr, *score_ = nullptr;
    int32_t* sel_ = nullptr;
};

}  // namespace strata::poly
