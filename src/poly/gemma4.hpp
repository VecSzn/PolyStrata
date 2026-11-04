// src/poly/gemma4.hpp - Gemma 4 with routed experts (llama.cpp architecture "gemma4"), as 26B-A4B is.
//
// One residual stream and attention in every layer.  Five layers in six attend only the last tokens (a sliding
// window) and keep their keys and values in a ring; the sixth attends every token so far, with wider heads.  A
// layer's FFN is a dense GELU MLP and routed GELU experts side by side, each normed before the two are added,
// and the router reads the layer's residual, not the experts' input.  The output matrix is the token table, and
// the logits are capped by a tanh.
//
// What guesses ahead for it is a model of its own in a file of its own (llama.cpp's "gemma4-assistant"): a few
// narrow layers with queries only, which read the keys and values this model's last window layer and last full
// layer have left.
#pragma once

#include "engine.hpp"
#include "strata/poly/kernels.hpp"   // KvInt8: what cache_int8() hands the attention under --kv int8

namespace strata::poly {

class Gemma4 final : public Engine {
public:
    ~Gemma4() override;
    size_t card_row_bytes() const override;
    void save_card_rows(int64_t first, int64_t n, uint8_t* dst) override;
    void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) override;
    bool sees() const override { return true; }
    int64_t whole(int64_t at, int64_t m) const override;

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
    void shift_rows(ptrdiff_t by) override;
    void output_rows(int first, int rows, float* h) override;
    void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const override;
    bool route_card(int il, int n, int32_t* ids, float* w) override;
    void moe_shared(const Layer& L, int il, int n) override;
    void router_scores(const Layer& L, int n) override;
    void finish_logits(float* logits, int rows) override;
    float embed_scale() const override;
    size_t extra_bytes() const override;
    void save_layer(int il, int64_t pos, uint8_t* dst) override;
    void load_extra(const uint8_t* src, int64_t pos) override;
    void load_drafter() override;
    void draft_step(int32_t token, int i) override;
    int head_key(int il, int n) const override;
    bool batch_heads() const override { return true; }

private:
    void mlp(const Layer& L, int il, int n, float* out);

    /// A layer's weights beside the ones every architecture has (Layer), and its caches.
    struct Block {
        bool swa = false;   ///< attends a sliding window; its caches are rings
        int dim = 0, n_kv = 0, n_rot = 0;   ///< a head's width, the heads that have keys of their own
        float theta_scale = 0;
        Dev wq, wk, wv, wo, q_norm, k_norm, post_attn;   ///< wv is absent where the values are the keys' projection
        uint16_t* keys = nullptr;   ///< rows of n_kv * dim, fp16: max_context of them, or ring_ in a window layer
        uint16_t* vals = nullptr;
        // --kv int8 keeps the same rows as codes with a delta a block of kQ8_0Values values, in two arrays each so
        // that a reader's four codes are one aligned load (store_int8).  One side of a layer is either the pair above
        // or the four below: they are not both filled.
        int8_t* k_code = nullptr;   ///< n_kv * dim codes a row, back to back
        uint16_t* k_delta = nullptr;   ///< n_kv * dim / kQ8_0Values halves a row, one a block
        int8_t* v_code = nullptr;
        uint16_t* v_delta = nullptr;
        // The halves a prompt batch's cuBLAS reads: the model's one pair, not this layer's (a pair a layer would
        // cost as much as the fp16 cache the quantization replaced), so a layer that is not the one that left it
        // converts its own codes in from row 0.  attend_q8 is a single-query attention - its readers are written
        // for a decode - so a batch of tokens goes through these.
        uint16_t* k_f16 = nullptr;
        uint16_t* v_f16 = nullptr;
        Dev ffn_gate, ffn_up, ffn_down;   ///< the dense MLP
        Dev post_mlp, pre_moe, post_moe, post_ffn, rscale;
        std::vector<float> escale;   ///< what each expert's output is multiplied by
        Dev escale_dev;              ///< the same on the card, when the file has it
        float out_scale = 1;
    };

    /// A layer of the draft model.  It has no keys or values: its queries read the rows of the model's layer `of`.
    struct Guess {
        int of = 0, dim = 0, n_rot = 0;
        bool swa = false;
        float theta_scale = 0, out_scale = 1;
        Dev attn_norm, wq, wo, q_norm, post_attn, ffn_norm, ffn_gate, ffn_up, ffn_down, post_ffn;
    };

    void load_draft_shape();
    void attn(Block& B, int64_t pos, int n);
    void attn_rows(Block& B, int n);
    void store(const float* x, const Block& B, int64_t pos, int n, bool value);
    /// One side of a layer's cache as the (code, delta) pair attend_int8 reads: the fp16 pointers under --kv int8
    /// would be null, so the callers take this instead of naming keys/vals themselves.
    KvInt8 cache_int8(const Block& B, bool value) const;
    template <class F>
    void window_runs(int64_t pos, F&& f) const;

    std::vector<Block> blocks_;   ///< one a layer of layers_
    std::vector<uint8_t> swa_;    ///< per layer: a window layer
    std::vector<int> kv_;         ///< per layer: the heads with keys of their own
    int n_head_ = 0, dim_full_ = 0, dim_swa_ = 0, rot_full_ = 0, rot_swa_ = 0;
    float base_full_ = 0, base_swa_ = 0;
    int window_ = 0;   ///< the tokens a window layer attends, the token itself among them
    int ring_ = 0;     ///< rows of a window layer's caches: the window and a call's tokens
    int keep_ = 0;     ///< rows of a window layer a sequence's state keeps
    float softcap_ = 0;
    Dev freq_;         ///< the full layers' rotary frequency factors
    int chunk_ = 1;    ///< tokens the attention takes at a time

    // device scratch
    float *qraw_ = nullptr, *q_ = nullptr, *kraw_ = nullptr, *k_ = nullptr, *vraw_ = nullptr, *v_ = nullptr,
          *att_ = nullptr, *score_ = nullptr, *mlp_ = nullptr, *rin_ = nullptr;
    float* awork_ = nullptr;   ///< a prompt batch's attention (attend_batch)
    int max_dim_ = 0;          ///< the wider of the two heads, which the int8 query scratch is sized for
    size_t twin_halves_ = 0;   ///< --kv int8: the halves one side of the fp16 twin holds (widest layer's rows)
    uint16_t* sh_k_f16_ = nullptr;   ///< the pair a prompt batch reads, one for the whole model
    uint16_t* sh_v_f16_ = nullptr;
    void twin_cover(size_t halves);

    int8_t* q8c_ = nullptr;    ///< --kv int8: this call's queries as codes, a chunk at a time, for attend_int8's dp4a
    uint16_t* q8d_ = nullptr;  ///< and their deltas


    // the draft model: its width and its MLPs', its layers, the projections in and out, its output norm and its
    // token table, which is its output matrix
    int d_embd_ = 0, d_ff_ = 0;
    float d_eps_ = 0;
    std::vector<Guess> guess_;
    Dev d_pre_, d_post_, d_out_norm_, d_table_, d_freq_;
    float *dxh_ = nullptr, *dg_gate_ = nullptr, *dg_up_ = nullptr, *dg_h_ = nullptr;   ///< its scratch
};

}  // namespace strata::poly
