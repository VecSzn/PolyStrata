// src/poly/glm5next.hpp - GLM 5.3 Flash (llama.cpp architecture "glm5-next").
//
// 45 layers with four residual streams joined by hyper-connections.  A layer's attention is a recurrent one (KDA)
// or a sparse one over cached latents with an indexer (DSA); its FFN is dense in the first layers and routed
// experts with a shared one after them.  One more block after the model's guesses a token ahead.
#pragma once

#include "engine.hpp"

namespace strata::poly {

class Glm5Next final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    void check_draft_block() override;
    size_t recurrent_floats() const override;
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
    void block_attend(int64_t pos, int n) override;
    void block_rows(int64_t pos, int n) override;
    const Names* tensor_alias(const char* name) const override;
    const Names* meta_alias(const char* key) const override;

private:
    /// A layer's weights beside the ones every architecture has (Layer), and its places in the state and caches.
    struct Block {
        bool dsa = false;
        Dev hc_attn_fn, hc_attn_scale, hc_attn_base, hc_ffn_fn, hc_ffn_scale, hc_ffn_base;
        Dev wo;
        // KDA
        Dev wq, wk, wv, conv_q, conv_k, conv_v, f_a, f_b, dt_b, beta, a, g_a, g_b, o_norm;
        float* conv = nullptr;  ///< 3 * d_inner * (d_conv - 1)
        float* s = nullptr;     ///< n_head * head_dim * head_dim
        // DSA
        Dev q_a, q_a_norm, q_b, kv_a, kv_a_norm, k_b, v_b;
        Dev idx_k, idx_k_norm, idx_k_norm_b, idx_q_b, idx_proj, idx_gate, idx_ape;
        uint16_t* lat = nullptr;  ///< max_context * kv_rank, fp16
        float* pool = nullptr;    ///< max_context / kpool * idx_dim: the indexer's pools (kernels::idx_store())
        float* ring = nullptr;    ///< ring_ * 2 * idx_dim: its latest rows
        // dense FFN
        Dev ffn_gate, ffn_up, ffn_down;
        std::vector<float> probs_b;   ///< what the routing adds to each expert's gate before it picks
        Dev probs_b_dev;              ///< the same on the card, when the file has it
    };

    bool batch_heads() const override { return true; }
    void kda(Block& B, int n);
    void dsa(Block& B, int64_t pos, int n);
    void dsa_store(Block& B, int64_t pos, int n);
    void dsa_select(Block& B, int64_t pos, int first, int n);
    void dense_ffn(Block& B, int il, int n);

    std::vector<Block> blocks_;   ///< one a layer of layers_

    int n_head_ = 0, kda_dim_ = 0, d_inner_ = 0, n_kda_ = 0, n_dsa_ = 0;
    int q_rank_ = 0, kv_rank_ = 0, mla_k_ = 0, mla_v_ = 0;
    int idx_heads_ = 0, idx_dim_ = 0, idx_top_k_ = 0, kpool_ = 0;
    int sinkhorn_ = 0;
    int score_stride_ = 0, sel_stride_ = 0;   ///< one token's indexer scores, and its selected cache rows
    int score_rows_ = 0;   ///< the tokens whose scores score_ holds at once
    int ring_ = 0;   ///< the rows of a Block's ring: a call's and the pool before it
    int64_t gate_rank_ = 0;   ///< the widest low-rank gate projection of the KDA layers
    float hc_eps_ = 0, norm_eps_ = 0, kda_lb_ = 0, w_scale_ = 1;
    bool w_norm_ = false;

    // device scratch
    float *mix_ = nullptr, *hcw_ = nullptr;
    float *proj_ = nullptr, *qkv_ = nullptr, *fa_ = nullptr, *g1_ = nullptr, *ga_ = nullptr, *g2_ = nullptr,
          *beta_ = nullptr, *gated_ = nullptr;
    float *t1_ = nullptr, *qr_ = nullptr, *q_ = nullptr, *t2_ = nullptr, *kv_ = nullptr, *qabs_ = nullptr,
          *kqv_ = nullptr, *vout_ = nullptr, *ikr_ = nullptr, *ik_ = nullptr, *ig_ = nullptr, *iq_ = nullptr,
          *iw_ = nullptr, *score_ = nullptr, *akeys_ = nullptr, *ascore_ = nullptr;
    size_t akeys_n_ = 0;   ///< floats in akeys_
    int32_t* sel_ = nullptr;

};

}  // namespace strata::poly
