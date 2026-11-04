// src/poly/kolibri1.hpp - Aleph Alpha's Kolibri 1 (llama.cpp architecture "kolibri1").
//
// One residual stream, and attention in every layer: 48 query heads over 4 key heads of 128 values each, every
// head's queries and keys normed.  Four layers in five turn the normed queries and keys by their position and keep
// their keys and values in a ring of the window's tokens; the fifth in every five attends every token so far and
// has no position at all (RNoPE).  A sandwich norm around each of a layer's two parts: the residual is normed
// before the attention and the attention's own output after it, and the same around the experts.
//
// Every layer's FFN is the same mixture: 384 routed experts and one ungated shared expert of its own beside them,
// and the six a token takes are picked on the router's scores and the selection bias, then weighted by the
// sigmoid of the bare scores, which are not made to sum to one.  The output matrix is its own, not the token
// table's, so the logits are read as they come.
#pragma once

#include "engine.hpp"
#include "strata/poly/kernels.hpp"

namespace strata::poly {

class Kolibri1 final : public Engine {
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
    int widest() const override;
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
        bool swa = false;   ///< attends a sliding window; its caches are rings and it turns its queries and keys
        int n_rot = 0;      ///< the values of a head the rotary embedding turns (0: none, as the full layers')
        Dev wq, wk, wv, wo, q_norm, k_norm, post_attn, post_ffn;
        uint16_t* keys = nullptr;   ///< rows of n_kv * head_dim, fp16: max_context of them, or ring_ in a window layer
        uint16_t* vals = nullptr;
        // --kv q8_0 keeps those rows as ggml Q8_0 instead: n_kv * head_dim int8 codes and an fp16 scale a 32-value
        // block, in two arrays a side so that a reader's four codes are one aligned load (store_q8).  A layer has
        // either the pair above or the pairs below, never both.
        int8_t* k_codes = nullptr;      ///< n_kv * head_dim codes a row, back to back
        uint16_t* k_scales = nullptr;   ///< n_kv * head_dim / 32 halves a row, one a block
        int8_t* v_codes = nullptr;
        uint16_t* v_scales = nullptr;
        std::vector<float> probs_b;   ///< what each expert's selection adds to its score
        Dev probs_b_dev;              ///< the same on the card, for a prompt batch's routing
    };

    bool q8() const { return opt_.kv_format == KvFmt::kQ8_0; }
    void attn(Block& B, int64_t pos, int n);
    void attn_rows(Block& B, int n);
    void store(const float* x, uint16_t* cache, const Block& B, int64_t pos, int n);
    void store_q8(const float* x, int8_t* codes, uint16_t* scales, const Block& B, int64_t pos, int n);
    template <class F>
    void window_runs(int64_t pos, F&& f) const;

    std::vector<Block> blocks_;   ///< one a layer of layers_
    std::vector<uint8_t> swa_;    ///< per layer: a window layer (blocks_ takes its own copy on each pass)
    int n_full_ = 0;              ///< the full layers this engine runs: their rows the card's rows hold
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0;
    float theta_scale_ = 0;   ///< a rotary pair turns by position * theta_scale^pair
    int window_ = 0;          ///< the tokens a window layer attends, the token itself among them
    int ring_ = 0;            ///< rows of a window layer's caches: the window and a call's tokens
    int keep_ = 0;            ///< rows of a window layer a sequence's state keeps
    int chunk_ = 1;           ///< tokens the attention takes at a time

    // device scratch
    float *qraw_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr,
          *score_ = nullptr;
    float* awork_ = nullptr;   ///< a prompt batch's attention (attend_batch)
    float* q8w_ = nullptr;     ///< --kv q8_0: a prompt batch's K and V blocks, dequantized to fp16 (kwork | vwork)
    size_t work_halves_ = 0;   ///< the halves of each of q8w_'s two halves (kQ8WorkRows rows at most)
    int8_t* q8c_ = nullptr;    ///< STRATA_KV_INT8: this call's queries as codes for attend_int8's dp4a
    uint16_t* q8d_ = nullptr;  ///< and their deltas
};

}  // namespace strata::poly
