// src/poly/lfm2moe.hpp - Liquid AI's LFM2 with routed experts (llama.cpp architecture "lfm2moe"): LFM2.5-24B-A2B.
//
// One residual stream.  Most layers mix the tokens with a gated short convolution of three taps; one in four has
// attention over the keys and values of every token so far, with normed queries and keys and a rotary embedding.
// The first layers' FFN is dense, the others' routed experts with sigmoid gates and a selection bias, no shared
// expert.  The output matrix is the token table.
#pragma once

#include "engine.hpp"

namespace strata::poly {

class Lfm2Moe final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override;
    bool snaps_spread() const override { return true; }
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override;
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    int head_key(int il, int n) const override;
    bool batch_heads() const override { return true; }
    const Names* tensor_alias(const char* name) const override;

private:
    /// A layer's weights beside the ones every architecture has (Layer), and its places in the state and caches.
    struct Block {
        bool attn = false;   ///< attention over the cache; else the short convolution
        Dev wo;
        // short convolution
        Dev in_proj, conv;
        float* past = nullptr;   ///< the two B * x before the next token, 2 * n_embd
        // attention
        Dev wq, wk, wv, q_norm, k_norm;
        uint16_t* keys = nullptr;   ///< max_context * n_kv * head_dim, fp16
        uint16_t* vals = nullptr;
        // a dense layer's FFN
        Dev ffn_gate, ffn_up, ffn_down;
        // routed experts: the bias their pick adds to the gates
        std::vector<float> probs_b;
        Dev probs_b_dev;
    };

    void conv(Block& B, int n);
    void attn(Block& B, int64_t pos, int n);
    void attn_rows(Block& B, int n);
    void dense_ffn(Block& B, int il, int n);

    std::vector<Block> blocks_;   ///< one a layer of layers_
    std::vector<int> kv_heads_;   ///< a layer's key heads; 0 for a convolution layer
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0, n_rot_ = 0, n_dense_ = 0;
    float theta_scale_ = 0;   ///< a rotary pair turns by position * theta_scale^pair
    bool w_norm_ = true;      ///< the picked experts' weights are made to sum to one
    float w_scale_ = 0;       ///< and then scaled by this, when it is not 0
    int n_conv_ = 0, n_attn_ = 0;
    int chunk_ = 1;   ///< tokens the attention takes at a time

    // device scratch
    float *proj_ = nullptr, *cv_ = nullptr;
    float *qraw_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr,
          *score_ = nullptr;
    float* awork_ = nullptr;   ///< a prompt batch's attention (attend_batch)
    size_t score_n_ = 0;       ///< the floats of score_: room for attend()'s blocks of rows too
};

}  // namespace strata::poly
