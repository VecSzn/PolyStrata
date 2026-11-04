// src/poly/flint.hpp - RecursiveSI's Flint (llama.cpp architecture "flint"): 40B-A2.5B.
//
// One residual stream and attention in every layer, with normed queries and keys and a learned sink a head: a
// logit of its own in each softmax that weighs no value.  Three layers in four attend only the last tokens (a
// sliding window, keys and values in a ring) and turn their queries and keys by the position; the others attend
// every token so far with no position at all.  The first layers' FFN is dense, the others' routed experts with
// sigmoid gates beside an ungated shared expert.  The output matrix is a tensor of its own.
#pragma once

#include "engine.hpp"

namespace strata::poly {

class Flint final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    size_t recurrent_floats() const override { return 0; }
    void alloc_caches() override;
    void assign_scratch() override;
    int widest() const override { return n_head_ * head_dim_; }
    int streams() const override { return 1; }
    void layer_head(Layer& L, int il, int n) override;
    void layer_tail(Layer& L, int il, int n) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    size_t extra_bytes() const override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    int head_key(int il, int n) const override;
    bool batch_heads() const override { return true; }

private:
    /// A layer's weights beside the ones every architecture has (Layer), and its caches.
    struct Block {
        bool swa = false;    ///< attends a sliding window; its caches are rings
        bool rope = false;   ///< turns its queries and keys by the position
        Dev wq, wk, wv, wo, q_norm, k_norm, sinks;
        uint16_t* keys = nullptr;   ///< rows of n_kv * head_dim, fp16: max_context of them, or ring_ in a window layer
        uint16_t* vals = nullptr;
        Dev ffn_gate, ffn_up, ffn_down;   ///< a dense layer's FFN
        std::vector<float> probs_b;       ///< the bias the routed experts' pick adds to the gates (zeros without one)
        Dev probs_b_dev;
    };

    void attn(Block& B, int64_t pos, int n);
    void attn_rows(Block& B, int n);
    void store(const float* x, const Block& B, int64_t pos, int n, bool value);
    void dense_ffn(Block& B, int il, int n);
    template <class F>
    void window_runs(int64_t pos, F&& f) const;

    std::vector<Block> blocks_;   ///< one a layer of layers_
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0, n_dense_ = 0;
    float theta_scale_ = 0;   ///< a rotary pair turns by position * theta_scale^pair
    bool w_norm_ = true;      ///< the picked experts' weights are made to sum to one
    float w_scale_ = 0;       ///< and then scaled by this, when it is not 0
    int window_ = 0;          ///< the tokens a window layer attends, the token itself among them
    int ring_ = 0;            ///< rows of a window layer's caches: the window and a call's tokens
    int keep_ = 0;            ///< rows of a window layer a sequence's state keeps
    int chunk_ = 1;           ///< tokens the attention takes at a time

    // device scratch
    float *qraw_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr,
          *score_ = nullptr;
    size_t score_n_ = 0;       ///< the floats of score_
    float* awork_ = nullptr;   ///< a prompt batch's attention (attend_batch)
};

}  // namespace strata::poly
