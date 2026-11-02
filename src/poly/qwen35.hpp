// src/poly/qwen35.hpp - Qwen3.5 and Qwen3.6 with routed experts (llama.cpp architecture "qwen35moe"), and the models
// trained from them.
//
// One residual stream.  Three layers in four have a recurrent attention, the gated delta net; the fourth has
// attention over the keys and values of every token so far, with a rotary embedding and a gate on its output.
// Every layer's FFN is routed experts with a gated shared one.  One more block after the model's, with the plain
// attention, guesses a token ahead.
#pragma once

#include "engine.hpp"
#include "gdn_mma_control.hpp"

namespace strata::poly {

class Qwen35 final : public Engine {
public:
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;
    bool sees() const override { return !clover_; }

protected:
    void load_geometry() override;
    void load_layer(int il) override;
    void begin_pass() override;
    void check_draft_block() override;
    size_t recurrent_floats() const override;
    bool snaps_spread() const override { return true; }
    // English answers write no token past 77,000, Chinese prose one in twenty past 131,072, and guesses from the
    // first 131,072 still write a Chinese answer faster than from all 248,320
    int guess_vocab() const override { return 131072; }
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
    void block_attend(int64_t pos, int n) override;
    void block_rows(int64_t pos, int n) override;
    int head_key(int il, int n) const override;
    bool batch_heads() const override { return true; }
    int64_t turn(int64_t row) const override;
    int block_mask() const override { return mask_; }
    void route_block(int il, float* scores, int n) override;

private:
    /// A layer's weights beside the ones every architecture has (Layer), and its places in the state and caches.
    struct Block {
        bool gdn = false;   ///< the gated delta net; else attention over the cache
        Dev wo;
        // gated delta net
        Dev wqkv, wz, conv, dt, a, alpha, beta, o_norm;
        // Clover's, for a block of tokens that see each other (gdn_block()): the convolution's taps over the next
        // three tokens, and a head's logit scale and the bias and log factor of its weight on the state
        Dev conv_right, bd_scale, bd_hist_b, bd_hist_c;
        float* past = nullptr;   ///< the convolution's: 3 * conv_dim
        float* s = nullptr;      ///< v_heads * s_dim * s_dim
        // attention
        Dev wq, wk, wv, q_norm, k_norm;
        uint16_t* keys = nullptr;   ///< max_context * n_kv * head_dim, fp16
        uint16_t* vals = nullptr;
        // q8_0 cache (opt-in): codes are n_kv*head_dim int8 a row; scales are one fp16 per 32-value block
        // (n_kv*(head_dim/32) a row).  Used instead of keys/vals when Options::kv_format == kQ8_0.
        int8_t*   k_codes  = nullptr;
        uint16_t* k_scales = nullptr;
        int8_t*   v_codes  = nullptr;
        uint16_t* v_scales = nullptr;
        Dev sh_gate_inp;   ///< the shared expert's gate
    };

    void gdn(Block& B, int n);
    void gdn_block_pass(Block& B, int n);
    void attn(Block& B, int64_t pos, int n);
    void attn_block_pass(Block& B, int64_t pos, int n);
    void attn_rows(Block& B, int n);
    void attn_store(Block& B, int64_t pos, int n);
    void turn_of(int64_t row, int32_t* thw) const;
    const int32_t* turns(int64_t pos, int n, int64_t* first);

    std::vector<Block> blocks_;   ///< one a layer of layers_
    bool clover_ = false;         ///< a Clover file (architecture clover_moe): text only, its ssm_a is A_log itself
    int mask_ = -1;               ///< Clover's mask token, which its block of guesses is filled with; -1: no block
    float coreset_ = 0.f;         ///< Clover: a block's tokens choose among the experts of this share of its weight
                                  ///< (moe_block_coreset_top_p); 0: among all
    int n_head_ = 0, n_kv_ = 0, head_dim_ = 0, n_rot_ = 0;
    float theta_scale_ = 0;   ///< a rotary pair turns by position * theta_scale^pair
    int sections_ = 1;        ///< the pairs the rotary embedding's three axes take in turn before they start over
    int32_t* axes_ = nullptr;         ///< a call's positions on the three axes, when it has a picture's rows
    std::vector<int32_t> h_axes_;
    int64_t axes_pos_ = -1;           ///< the call axes_ holds: its first row, its rows and the pictures it was
    int axes_n_ = 0;                  ///< made for (images_set_)
    uint64_t axes_set_ = 0;
    int s_dim_ = 0, k_heads_ = 0, v_heads_ = 0, d_inner_ = 0, conv_dim_ = 0;
    int n_gdn_ = 0, n_attn_ = 0;
    GdnMmaControl gdn_mma_;           ///< captured request and per-engine capability (STRATA_GDN_MMA=1)
    int chunk_ = 1;   ///< tokens the attention takes at a time

    // device scratch
    float *proj_ = nullptr, *qkv_ = nullptr, *z_ = nullptr, *alpha_ = nullptr, *beta_ = nullptr, *gated_ = nullptr;
    float *qg_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *v_ = nullptr, *att_ = nullptr,
          *score_ = nullptr, *sg_ = nullptr;
    float* awork_ = nullptr;   ///< a prompt batch's attention (attend_batch)
    float* q8w_ = nullptr;             ///< a q8 prompt batch's K and V blocks, dequantized to fp16 (kwork | vwork)
    size_t work_halves_ = 0;           ///< the halves of each of q8w_'s two halves (kQ8WorkRows rows at most)
    bool fattn_ = false;               ///< a prompt batch through fattn_q8() / fattn_f16() (STRATA_FATTN=0: never)
    bool fdec_ = false;                ///< a few tokens through fdec_q8() / fdec_f16() (STRATA_FDEC=0: never)
    int64_t fdec_from_ = 0;            ///< ...once the cache holds this many positions (STRATA_FDEC=1: from the first)
    bool hip_value_split_logged_ = false;   ///< STRATA_HIP_VALUE_SPLIT has reported its availability

};

}  // namespace strata::poly
