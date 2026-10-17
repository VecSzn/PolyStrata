// src/poly/glm5next.cpp - GLM 5.3 Flash's layers.  The reference for the math is llama.cpp's glm5-next graph.
#include "glm5next.hpp"

#include "mmq.hpp"
#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <algorithm>
#include <bit>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr int kHc = 4, kConv = 4;   // what kernels.cu is written for; checked at load
constexpr int kAttnRows = 32;       // tokens one pass of the sparse attention takes; their keys are 4 MiB a token
constexpr size_t kScoreBytes = size_t(128) << 20;   // a prompt batch's indexer scores at most: 1,024 tokens at 128K
constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum

// The model's GGUFs come out of more than one converter.  `name` is what the layout of neurall's llama.cpp fork
// calls a thing, and what this file calls it.  `other` is what the same thing is called in the files made for
// patrickbdevaney's fork (the first, where it differs) and in antirez's (the second).
constexpr Names kTensorNames[] = {
    {"hc_attn_fn.weight", {"hc_attn_fn"}},
    {"hc_attn_scale.weight", {"hc_attn_scale"}},
    {"hc_attn_base.weight", {"hc_attn_base"}},
    {"hc_ffn_fn.weight", {"hc_ffn_fn"}},
    {"hc_ffn_scale.weight", {"hc_ffn_scale"}},
    {"hc_ffn_base.weight", {"hc_ffn_base"}},
    {"indexer_compressor_gate.weight", {"indexer.kpool_gate", "indexer.pool_gate.weight"}},
    {"indexer_compressor_ape.weight", {"indexer.kpool_ape", "indexer.pool_ape.weight"}},
    {"attn_q.weight", {nullptr, "kda_q.weight"}},
    {"attn_k.weight", {nullptr, "kda_k.weight"}},
    {"attn_v.weight", {nullptr, "kda_v.weight"}},
    {"attn_output.weight", {nullptr, "kda_output.weight"}},
    {"ssm_conv1d_q.weight", {nullptr, "kda_q_conv.weight"}},
    {"ssm_conv1d_k.weight", {nullptr, "kda_k_conv.weight"}},
    {"ssm_conv1d_v.weight", {nullptr, "kda_v_conv.weight"}},
    {"ssm_f_a.weight", {nullptr, "kda_f_a.weight"}},
    {"ssm_f_b.weight", {nullptr, "kda_f_b.weight"}},
    {"ssm_g_a.weight", {nullptr, "kda_g_a.weight"}},
    {"ssm_g_b.weight", {nullptr, "kda_g_b.weight"}},
    {"ssm_dt.bias", {nullptr, "kda_dt_bias.weight"}},
    {"ssm_beta.weight", {nullptr, "kda_beta.weight"}},
    {"ssm_a", {nullptr, "kda_a_log.weight"}},
    {"ssm_norm.weight", {nullptr, "kda_o_norm.weight"}},
};
constexpr Names kMetaNames[] = {
    {"kda.gate_lower_bound", {"ssm.gate_lower_bound", "linear_attention.gate_lower_bound"}},
    {"kda.head_dim", {nullptr, "linear_attention.head_dimension"}},
    {"ssm.conv_kernel", {nullptr, "linear_attention.conv_kernel"}},
    {"attention.key_length_mla", {nullptr, "attention.key_length"}},
    {"attention.value_length_mla", {nullptr, "attention.value_length"}},
    {"attention.indexer.kpool", {nullptr, "attention.indexer.pool_size"}},
    {"hyper_connection.count", {"attention.hc.mult"}},
    {"hyper_connection.sinkhorn_iterations", {"attention.hc.sinkhorn_iters"}},
    {"hyper_connection.epsilon", {"attention.hc.eps"}},
    {"swiglu_clamp_exp", {"swiglu_limit"}},   // one limit for every layer, and for the shared expert too
    {"swiglu_clamp_shexp", {"swiglu_limit"}},
};

}  // namespace

const Names* Glm5Next::tensor_alias(const char* name) const { return names_of(kTensorNames, name); }
const Names* Glm5Next::meta_alias(const char* key) const { return names_of(kMetaNames, key); }

void Glm5Next::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_dsa_ = n_kda_ = 0;
    gate_rank_ = 0;
}

void Glm5Next::check_draft_block() {
    if (!blocks_[(size_t) n_layer_].dsa || !layers_[(size_t) n_layer_].moe)
        throw std::runtime_error("the block that guesses ahead is not one this engine runs; --no-draft loads "
                                 "the model without it");
}

int Glm5Next::streams() const { return kHc; }

// The routed experts' sum goes back into the residual streams.
void Glm5Next::layer_tail(Layer&, int, int n) { hc_post(fo_, hcw_, r_, n_embd_, n, st_); }

// The weights a layer's head mixes the streams with are the ones its tail puts the experts' sum back with.
void Glm5Next::shift_rows(ptrdiff_t by) {
    Engine::shift_rows(by);
    hcw_ += by * (kHc + kHc * kHc);
}

// The model's output for `rows` tokens from row `first` of the call: the streams' mean, normed.
void Glm5Next::output_rows(int first, int rows, float* h) {
    hc_mean(r_ + (size_t) first * kHc * n_embd_, cur_, n_embd_, rows, st_);
    rms_norm(cur_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

void Glm5Next::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };

    n_ff_ = shape_i("feed_forward_length");
    n_head_ = shape_i("attention.head_count");
    // the indexer's LayerNorm; a converter that leaves the key out means the model's own 1e-6
    const MetaValue* norm_eps = find("attention.layer_norm_epsilon");
    norm_eps_ = norm_eps ? (float) norm_eps->f : 1e-6f;
    w_scale_ = shape_f("expert_weights_scale");
    w_norm_ = shape_i("expert_weights_norm") != 0;
    kda_dim_ = shape_i("kda.head_dim");
    kda_lb_ = shape_f("kda.gate_lower_bound");
    q_rank_ = shape_i("attention.q_lora_rank");
    kv_rank_ = shape_i("attention.kv_lora_rank");
    mla_k_ = shape_i("attention.key_length_mla");
    mla_v_ = shape_i("attention.value_length_mla");
    idx_heads_ = shape_i("attention.indexer.head_count");
    idx_dim_ = shape_i("attention.indexer.key_length");
    idx_top_k_ = shape_i("attention.indexer.top_k");
    // tokens to a pool: a key, or else the second axis of a layer's pooling table
    if (const MetaValue* kpool = find("attention.indexer.kpool")) kpool_ = (int) kpool->u;
    for (int il = 0; il < n_layer_ && kpool_ == 0; ++il)
        if (const TensorInfo* ape = tensor(il, "indexer_compressor_ape.weight"))
            kpool_ = ape->shape.size() == 2 ? (int) ape->shape[1] : 0;
    sinkhorn_ = shape_i("hyper_connection.sinkhorn_iterations");
    hc_eps_ = shape_f("hyper_connection.epsilon");
    d_inner_ = n_head_ * kda_dim_;

    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    need(shape_i("hyper_connection.count") == kHc, "hyper_connection.count is not 4");
    need(shape_i("ssm.conv_kernel") == kConv, "ssm.conv_kernel is not 4");
    const MetaValue* gating = find("expert_gating_func");   // left out by one converter: the model's sigmoid
    need(!gating || gating->u == 2, "the expert gate is not a sigmoid");
    need(shape_i("expert_shared_count") == 1, "expert_shared_count is not 1");
    need(kda_dim_ > 0 && kda_dim_ % 32 == 0 && kda_dim_ <= 256, "kda.head_dim is not a multiple of 32 up to 256");
    need(idx_dim_ > 0 && idx_dim_ <= 256, "the indexer key is longer than 256");
    need(kpool_ > 1 && idx_top_k_ > 0 && idx_top_k_ % kpool_ == 0, "indexer.top_k is not whole pools");

    clamp_exp_ = clamp_limits(find("swiglu_clamp_exp"), n_layer_);
    clamp_shexp_ = find("swiglu_clamp_shexp") ? clamp_limits(find("swiglu_clamp_shexp"), n_layer_) : clamp_exp_;
    if (draft_) {   // the draft block's limits are in the file the block is in: the model's own may end before it
        const MetaValue* routed = meta_in(*block_, "swiglu_clamp_exp");
        const MetaValue* shared = meta_in(*block_, "swiglu_clamp_shexp");
        clamp_exp_.push_back(clamp_limits(routed, n_layer_ + 1).back());
        clamp_shexp_.push_back(shared ? clamp_limits(shared, n_layer_ + 1).back() : clamp_exp_.back());
    }
}

void Glm5Next::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b, Want w = kAnyMat) {
        return upload(tensor(il, n), blk(il, n), a, b, w);
    };
    auto has = [&](const char* n) { return tensor(il, n) != nullptr; };
    // the low rank of a gate's two-matrix projection is the first matrix's row count
    auto rank = [&](const char* n) {
        const TensorInfo* t = tensor(il, n);
        if (!t || t->shape.size() < 2) throw std::runtime_error("the GGUF has no matrix " + blk(il, n));
        return (int64_t) t->shape[1];
    };
    const int n_mix = kHc * (2 + kHc);

    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    if (il < n_layer_) {   // the draft block has one residual stream and adds to it
        B.hc_attn_fn = f32("hc_attn_fn.weight", kHc * n_embd_, n_mix);
        B.hc_attn_scale = f32("hc_attn_scale.weight", 3);
        B.hc_attn_base = f32("hc_attn_base.weight", n_mix);
        B.hc_ffn_fn = f32("hc_ffn_fn.weight", kHc * n_embd_, n_mix);
        B.hc_ffn_scale = f32("hc_ffn_scale.weight", 3);
        B.hc_ffn_base = f32("hc_ffn_base.weight", n_mix);
    }

    B.dsa = has("attn_q_a.weight");
    L.fixed = !B.dsa;   // the sparse attention's launches take the position
    if (B.dsa) {
        B.q_a = mat("attn_q_a.weight", n_embd_, q_rank_);
        B.q_a_norm = f32("attn_q_a_norm.weight", q_rank_);
        B.q_b = mat("attn_q_b.weight", q_rank_, (int64_t) n_head_ * mla_k_);
        B.kv_a = mat("attn_kv_a_mqa.weight", n_embd_, kv_rank_);
        B.kv_a_norm = f32("attn_kv_a_norm.weight", kv_rank_);
        B.k_b = mat("attn_k_b.weight", mla_k_, (int64_t) kv_rank_ * n_head_, kFloatMat);
        B.v_b = mat("attn_v_b.weight", kv_rank_, (int64_t) mla_v_ * n_head_, kFloatMat);
        B.wo = mat("attn_output.weight", (int64_t) n_head_ * mla_v_, n_embd_);
        B.idx_k = mat("indexer.attn_k.weight", n_embd_, idx_dim_);
        B.idx_k_norm = f32("indexer.k_norm.weight", idx_dim_);
        B.idx_k_norm_b = f32("indexer.k_norm.bias", idx_dim_);
        B.idx_q_b = mat("indexer.attn_q_b.weight", q_rank_, (int64_t) idx_dim_ * idx_heads_);
        B.idx_proj = mat("indexer.proj.weight", n_embd_, idx_heads_);
        B.idx_gate = mat("indexer_compressor_gate.weight", n_embd_, idx_dim_);
        B.idx_ape = f32("indexer_compressor_ape.weight", idx_dim_, kpool_);
        ++n_dsa_;
    } else {
        const int64_t fr = rank("ssm_f_a.weight"), gr = rank("ssm_g_a.weight");
        gate_rank_ = std::max<int64_t>({gate_rank_, fr, gr});
        B.wq = mat("attn_q.weight", n_embd_, d_inner_);
        B.wk = mat("attn_k.weight", n_embd_, d_inner_);
        B.wv = mat("attn_v.weight", n_embd_, d_inner_);
        B.wo = mat("attn_output.weight", d_inner_, n_embd_);
        B.conv_q = f32("ssm_conv1d_q.weight", kConv, d_inner_);
        B.conv_k = f32("ssm_conv1d_k.weight", kConv, d_inner_);
        B.conv_v = f32("ssm_conv1d_v.weight", kConv, d_inner_);
        B.f_a = mat("ssm_f_a.weight", n_embd_, fr);
        B.f_b = mat("ssm_f_b.weight", fr, d_inner_);
        B.dt_b = f32("ssm_dt.bias", d_inner_);
        B.beta = mat("ssm_beta.weight", n_embd_, n_head_);
        // kda_step() takes -exp(A_log), which is what one converter stores.  Another stores exp(A_log), and the
        // sign tells those two apart.  The third stores A_log, under a name that says so.
        const TensorInfo* a = tensor(il, "ssm_a");
        float first = -1.f;
        if (a && a->type == kTypeF32 && home(*a).payload(*a) >= sizeof(first))
            std::memcpy(&first, home(*a).data(*a), sizeof(first));
        float (*fix)(float) = nullptr;
        if (a && a->name.find("a_log") != std::string::npos) fix = [](float v) { return -std::exp(v); };
        else if (first > 0.f) fix = [](float v) { return -v; };
        B.a = upload(a, blk(il, "ssm_a"), n_head_, 1, kF32, fix);
        B.g_a = mat("ssm_g_a.weight", n_embd_, gr);
        B.g_b = mat("ssm_g_b.weight", gr, d_inner_);
        B.o_norm = f32("ssm_norm.weight", kda_dim_);
        ++n_kda_;
    }

    if (!has("ffn_gate_inp.weight")) {
        B.ffn_gate = mat("ffn_gate.weight", n_embd_, n_ff_);
        B.ffn_up = mat("ffn_up.weight", n_embd_, n_ff_);
        B.ffn_down = mat("ffn_down.weight", n_ff_, n_embd_);
        return;
    }
    load_moe(il);
    B.probs_b.assign((size_t) n_expert_, 0.f);
    if (const TensorInfo* b = tensor(il, "exp_probs_b.bias")) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(b->name + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        B.probs_b_dev = upload(b, blk(il, "exp_probs_b.bias"), n_expert_, 1, kF32);
        home(*b).payload(*b);
        std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
    }
}

// Per-token buffers hold rows_ tokens, row after row.
void Glm5Next::assign_scratch() {
    const size_t n = (size_t) rows_, d = n * d_inner_;
    mix_ = fbuf(n * (kHc * (2 + kHc) + 1));
    hcw_ = fbuf(n * (kHc + kHc * kHc));

    proj_ = fbuf(3 * d);
    qkv_ = fbuf(3 * d);
    fa_ = fbuf(n * gate_rank_);
    ga_ = fbuf(n * gate_rank_);
    g1_ = fbuf(d);
    g2_ = fbuf(d);
    beta_ = fbuf(n * n_head_);
    gated_ = fbuf(d);

    t1_ = fbuf(n * q_rank_);
    qr_ = fbuf(n * q_rank_);
    q_ = fbuf(n * n_head_ * mla_k_);
    t2_ = fbuf(n * kv_rank_);
    kv_ = fbuf(n * kv_rank_);
    qabs_ = fbuf(n * n_head_ * kv_rank_);
    kqv_ = fbuf(n * n_head_ * kv_rank_);
    vout_ = fbuf(n * n_head_ * mla_v_);
    ikr_ = fbuf(n * idx_dim_);
    ik_ = fbuf(n * idx_dim_);
    ig_ = fbuf(n * idx_dim_);
    iq_ = fbuf(n * idx_heads_ * idx_dim_);
    iw_ = fbuf(n * idx_heads_);
    score_ = fbuf((size_t) score_rows_ * score_stride_);
    const size_t keys = (size_t) std::min(opt_.max_context, idx_top_k_ + kpool_ - 1);
    akeys_n_ = std::min(n, (size_t) kAttnRows) * keys * kv_rank_;
    akeys_ = fbuf(akeys_n_);
    ascore_ = fbuf(std::min(n, (size_t) kAttnRows) * keys * n_head_);
}

size_t Glm5Next::recurrent_floats() const {
    return size_t(n_kda_) * (size_t(3) * d_inner_ * (kConv - 1) + size_t(n_head_) * kda_dim_ * kda_dim_);
}

int Glm5Next::widest() const {
    return std::max({d_inner_, n_head_ * mla_v_, n_head_ * mla_k_, q_rank_, kv_rank_, (int) gate_rank_});
}

// The sparse layers' caches, and each recurrent layer's convolution and delta-rule state in state_.
//
// A call writes the indexer rows of [pos, pos + n), n <= rows_, and the next call starts after pos unless a state
// is loaded (load_extra()), so a ring of rows_ + kpool rows holds the rows of the pool the next call fills.
void Glm5Next::alloc_caches() {
    const size_t conv_n = size_t(3) * d_inner_ * (kConv - 1), s_n = size_t(n_head_) * kda_dim_ * kda_dim_;
    ring_ = rows_ + kpool_;
    const size_t ring_n = (size_t) ring_ * 2 * idx_dim_;
    float* ring = (float*) seq_alloc(n_dsa_ * ring_n * sizeof(float), "the indexer cache");
    float* st = state_;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.dsa) {
            B.lat = (uint16_t*) kv_alloc((size_t) kv_rank_ * sizeof(uint16_t), 1);
            B.pool = (float*) kv_alloc((size_t) idx_dim_ * sizeof(float), kpool_);
            B.ring = ring;
            ring += ring_n;
        } else {
            B.conv = st;
            st += conv_n;
            B.s = st;
            st += s_n;
        }
    }

    score_stride_ = opt_.max_context / kpool_ + 1;
    score_rows_ = (int) std::clamp<size_t>(kScoreBytes / (score_stride_ * sizeof(float)), 1, (size_t) rows_);
    sel_stride_ = idx_top_k_ + kpool_;
    const size_t n = (size_t) rows_;
    if (!sel_) sel_ = (int32_t*) dmalloc(n * sel_stride_ * sizeof(int32_t), "scratch");
}

// A sparse layer's part of a state: its indexer rows of the pool the position is in, which the ring no longer
// has when the sequence comes back to that position.  kpool - 1 rows each, behind the recurrent state.
size_t Glm5Next::extra_bytes() const { return (size_t) n_dsa_ * (kpool_ - 1) * 2 * idx_dim_ * sizeof(float); }

void Glm5Next::load_extra(const uint8_t* src, int64_t pos) {
    const size_t row = size_t(2) * idx_dim_ * sizeof(float);
    for (const Block& B : blocks_) {
        if (!B.dsa) continue;
        for (int64_t p = pos - pos % kpool_; p < pos; ++p)
            ck(cudaMemcpy(B.ring + (size_t) (p % ring_) * 2 * idx_dim_, src + (size_t) (p % kpool_) * row, row,
                          cudaMemcpyHostToDevice),
               "load state");
        src += (size_t) (kpool_ - 1) * row;
    }
}

// A recurrent layer's convolution and delta-rule state lie one behind the other in state_ (alloc_caches()).
void Glm5Next::save_layer(int il, int64_t pos, uint8_t* dst) {
    const Block& B = blocks_[(size_t) il];
    if (B.dsa) {
        const size_t row = size_t(2) * idx_dim_ * sizeof(float), part = (size_t) (kpool_ - 1) * row;
        dst += state_n_ * sizeof(float);
        for (int j = 0; j < il; ++j) dst += blocks_[(size_t) j].dsa ? part : 0;
        std::memset(dst, 0, part);
        for (int64_t p = pos - pos % kpool_; p < pos; ++p)
            ck(cudaMemcpy(dst + (size_t) (p % kpool_) * row, B.ring + (size_t) (p % ring_) * 2 * idx_dim_, row,
                          cudaMemcpyDeviceToHost),
               "save state");
        return;
    }
    const size_t n = size_t(3) * d_inner_ * (kConv - 1) + size_t(n_head_) * kda_dim_ * kda_dim_;
    ck(cudaMemcpy(dst + (size_t) (B.conv - state_) * sizeof(float), B.conv, n * sizeof(float),
                  cudaMemcpyDeviceToHost),
       "save state");
}

// A row: the latent the attention reads, and room for a pool's key, which the row that ends the pool has.
size_t Glm5Next::card_row_bytes() const {
    return (size_t) n_dsa_ * ((size_t) kv_rank_ * sizeof(uint16_t) + (size_t) idx_dim_ * sizeof(float));
}

void Glm5Next::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t lat = (size_t) kv_rank_ * sizeof(uint16_t), key = (size_t) idx_dim_ * sizeof(float);
    const size_t at = (size_t) first, rows = (size_t) n;
    const int64_t p0 = first / kpool_, p1 = (first + n) / kpool_;   // the pools that end in these rows
    for (const Block& B : blocks_) {
        if (!B.dsa || rows == 0) continue;
        ck(cudaMemcpy(dst, (const uint8_t*) B.lat + at * lat, rows * lat, cudaMemcpyDeviceToHost), "save rows");
        dst += rows * lat;
        std::memset(dst, 0, rows * key);
        if (p1 > p0)
            ck(cudaMemcpy2D(dst + (size_t) (p0 * kpool_ + kpool_ - 1 - first) * key, kpool_ * key,
                            B.pool + (size_t) p0 * idx_dim_, key, key, (size_t) (p1 - p0), cudaMemcpyDeviceToHost),
               "save rows");
        dst += rows * key;
    }
}

void Glm5Next::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t lat = (size_t) kv_rank_ * sizeof(uint16_t), key = (size_t) idx_dim_ * sizeof(float);
    const size_t at = (size_t) first, rows = (size_t) n, skip = (size_t) (first - from), all = (size_t) count;
    const int64_t p0 = first / kpool_, p1 = (first + n) / kpool_;
    for (const Block& B : blocks_) {
        if (!B.dsa || rows == 0) continue;
        ck(cudaMemcpy((uint8_t*) B.lat + at * lat, src + skip * lat, rows * lat, cudaMemcpyHostToDevice), "load rows");
        src += all * lat;
        if (p1 > p0)
            ck(cudaMemcpy2D(B.pool + (size_t) p0 * idx_dim_, key,
                            src + (size_t) (p0 * kpool_ + kpool_ - 1 - from) * key, kpool_ * key, key,
                            (size_t) (p1 - p0), cudaMemcpyHostToDevice),
               "load rows");
        src += all * key;
    }
}

// Its kernels take a count where the delta net's take a bit a token: they leave the state after a call's first
// tokens.  A window of this model has a place for each of its tokens (snaps_spread()), so the bits are those.
void Glm5Next::kda(Block& B, int n) {
    const size_t d = (size_t) n * d_inner_;
    lin(B.wq, xn_, proj_, n, xq_);
    lin(B.wk, xn_, proj_ + d, n, xq_);
    lin(B.wv, xn_, proj_ + 2 * d, n, xq_);
    if (rows_of_) {
        // step(): the products for all rows at once, and each row's convolution and step in its own sequence's
        // state.  The kernels take a row's q, k and v side by side, so they are copied together past the rows'.
        lin(B.f_a, xn_, fa_, n, xq_);
        lin(B.f_b, fa_, g1_, n);
        add_bias(fp(B.dt_b), g1_, d_inner_, n, st_);
        lin(B.beta, xn_, beta_, n, xq_);
        lin(B.g_a, xn_, ga_, n, xq_);
        lin(B.g_b, ga_, g2_, n);
        float* one = proj_ + 3 * d;
        const size_t row = (size_t) d_inner_;
        for (int i = 0; i < n; ++i) {
            const int s = rows_of_[i].seq;
            for (int p = 0; p < 3; ++p)
                ck(cudaMemcpyAsync(one + p * row, proj_ + p * d + i * row, row * sizeof(float), cudaMemcpyDeviceToDevice,
                                   cs(st_)),
                   "kda rows");
            float* out = qkv_ + 3 * i * row;
            kda_conv(one, seq_state(s, B.conv), fp(B.conv_q), fp(B.conv_k), fp(B.conv_v), out, d_inner_, 1, st_);
            kda_step(out, g1_ + i * row, fp(B.a), beta_ + (size_t) i * n_head_, seq_state(s, B.s), g2_ + i * row,
                     fp(B.o_norm), gated_ + i * row, kda_dim_, n_head_, 1, kda_lb_, rms_eps_, st_);
        }
        lin(B.wo, gated_, ao_, n);
        return;
    }
    kda_conv(proj_, B.conv, fp(B.conv_q), fp(B.conv_k), fp(B.conv_v), qkv_, d_inner_, n, st_,
             snap_bits_ ? snap_at_ + (B.conv - state_) : nullptr, rec_n_, std::popcount(snap_bits_));

    lin(B.f_a, xn_, fa_, n, xq_);
    lin(B.f_b, fa_, g1_, n);
    add_bias(fp(B.dt_b), g1_, d_inner_, n, st_);
    lin(B.beta, xn_, beta_, n, xq_);
    lin(B.g_a, xn_, ga_, n, xq_);
    lin(B.g_b, ga_, g2_, n);
    kda_step(qkv_, g1_, fp(B.a), beta_, B.s, g2_, fp(B.o_norm), gated_, kda_dim_, n_head_, n, kda_lb_, rms_eps_,
             st_, snap_bits_ ? snap_at_ + (B.s - state_) : nullptr, rec_n_, std::popcount(snap_bits_));
    lin(B.wo, gated_, ao_, n);
}

// For the n tokens from row `first` of a call whose first token is at `at`: the n_top pools with the highest
// indexer score, expanded to their rows, then the tail that is in no pool yet.  It only runs past top_k tokens.
void Glm5Next::dsa_select(Block& B, int64_t at, int first, int n) {
    const int64_t pos = at + first;
    const int n_top = idx_top_k_ / kpool_;
    lin(B.idx_q_b, qr_ + (size_t) first * q_rank_, iq_, n);
    lin(B.idx_proj, xn_ + (size_t) first * n_embd_, iw_, n,
        (const uint8_t*) xq_ + (size_t) first * kernels::native_q8_1_bytes(n_embd_));
    // A prompt batch's scores are products through cuBLAS, with the attention's scratch for them.  A few tokens
    // take the kernel that gives a token the same bits in any company.  At a long context score_ has room for
    // only some of a batch's tokens, and they take turns.
    for (int t = 0, c = 0; t < n; t += c) {
        c = std::min(score_rows_, n - t);
        const float *q = iq_ + (size_t) t * idx_heads_ * idx_dim_, *w = iw_ + (size_t) t * idx_heads_;
        if (n >= kStageRows)
            idx_score_batch(blas_, q, w, B.pool, score_, score_stride_, idx_dim_, idx_heads_, kpool_, pos + t, c,
                            akeys_, akeys_n_, st_);
        else
            idx_score(q, w, B.pool, score_, score_stride_, idx_dim_, idx_heads_, kpool_, pos + t, c, st_);
        idx_select(score_, score_stride_, sel_ + (size_t) t * sel_stride_, sel_stride_, n_top, kpool_, pos + t, c,
                   st_);
    }
}

// The rows n tokens from position `pos` leave in a sparse-attention layer's caches: the latent the attention
// reads, the indexer's key and gate, and the keys of the pools they fill.  xn_ and xq_ hold the layer's input.
void Glm5Next::dsa_store(Block& B, int64_t pos, int n) {
    lin(B.kv_a, xn_, t2_, n, xq_);
    rms_norm(t2_, fp(B.kv_a_norm), kv_, kv_rank_, n, rms_eps_, st_);
    store_f16(kv_, B.lat, kv_rank_, pos, n, st_);

    lin(B.idx_k, xn_, ikr_, n, xq_);
    layer_norm(ikr_, fp(B.idx_k_norm), fp(B.idx_k_norm_b), ik_, idx_dim_, n, norm_eps_, st_);
    lin(B.idx_gate, xn_, ig_, n, xq_);
    idx_store(ik_, ig_, B.ring, ring_, idx_dim_, pos, n, st_);
    idx_pool(B.ring, ring_, B.pool, fp(B.idx_ape), idx_dim_, kpool_, pos, n, st_);
}

void Glm5Next::dsa(Block& B, int64_t pos, int n) {
    if (rows_of_) {
        // step(): a row at a time, the whole layer, with the row's sequence's latents, pools and ring; its
        // kernels take their positions as arguments, so the layer is not recorded either way
        const Row* rows = std::exchange(rows_of_, nullptr);
        float *const xn = xn_, *const ao = ao_;
        void* const xq = xq_;
        const size_t q8 = kernels::native_q8_1_bytes(n_embd_);
        for (int i = 0; i < n; ++i) {
            const int s = rows[i].seq;
            Block one = B;
            one.lat = (uint16_t*) seq_buf(s, B.lat);
            one.pool = (float*) seq_buf(s, B.pool);
            one.ring = (float*) seq_buf(s, B.ring);
            xn_ = xn + (size_t) i * n_embd_;
            xq_ = (uint8_t*) xq + (size_t) i * q8;
            ao_ = ao + (size_t) i * n_embd_;
            dsa(one, rows[i].pos, 1);
        }
        xn_ = xn;
        xq_ = xq;
        ao_ = ao;
        rows_of_ = rows;
        return;
    }
    lin(B.q_a, xn_, t1_, n, xq_);
    rms_norm(t1_, fp(B.q_a_norm), qr_, q_rank_, n, rms_eps_, st_);
    lin(B.q_b, qr_, q_, n);
    lin(B.k_b, q_, qabs_, n, nullptr, kv_rank_);   // the key projection, absorbed into the query: per head

    dsa_store(B, pos, n);

    // While a token's pools all fit in top_k, every one is selected and its attention is plain causal attention.
    // That holds up to a position, so the tokens of a call are dense first and sparse after.
    const int64_t last_dense = (int64_t) idx_top_k_ + kpool_ - 2;
    const int dense = (int) std::clamp<int64_t>(last_dense - pos + 1, 0, n);
    if (dense < n) dsa_select(B, pos, dense, n - dense);
    // A prompt batch's tokens share their products, padded to the token with the most keys.  A few tokens get a
    // product each: a token of a window must come out as it would alone, to the last bit, and cuBLAS rounds a
    // product of another size another way.
    const int together = n < kStageRows ? 1 : std::min(rows_, kAttnRows);
    dsa_attend(blas_, qabs_, B.lat, sel_, sel_stride_, idx_top_k_, kpool_, pos, n, dense, akeys_, ascore_,
               together, kqv_, kv_rank_, n_head_, 1.f / std::sqrt((float) mla_k_), st_);
    lin(B.v_b, kqv_, vout_, n, nullptr, mla_v_);
    lin(B.wo, vout_, ao_, n);
}

void Glm5Next::dense_ffn(Block& B, int il, int n) {
    lin(B.ffn_gate, xn_, g_, n, xq_);
    lin(B.ffn_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_, clamp_shexp_[(size_t) il], st_);
    lin(B.ffn_down, h_, fo_, n);
}

// What ggml's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias, and weights from
// the bare gates.
void Glm5Next::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = probs + n_expert_;
    for (int e = 0; e < n_expert_; ++e) {
        probs[e] = 1.f / (1.f + std::exp(-logits[e]));
        pick[e] = probs[e] + bias[(size_t) e];
    }
    float sum = 0.f;
    for (int k = 0; k < n_used_; ++k) {
        int best = 0;
        for (int e = 1; e < n_expert_; ++e)
            if (pick[e] > pick[best]) best = e;
        pick[best] = -INFINITY;
        ids[k] = best;
        w[k] = probs[best];
        sum += w[k];
    }
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

bool Glm5Next::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, fp(B.probs_b_dev), nullptr, w_norm_, w_scale_, ids, w, st_);
    return true;
}

// The shared expert.  It runs on the GPU while the host routes.
void Glm5Next::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_exp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);
}

// A layer from its input to its router's copy (a dense layer: to its end).  A prompt batch's MoE (moe_rows) does
// its own routing.
void Glm5Next::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    hc_pre(r_, fp(B.hc_attn_fn), fp(B.hc_attn_scale), fp(B.hc_attn_base), mix_, hcw_, cur_, n_embd_, n, rms_eps_,
           hc_eps_, sinkhorn_, st_);
    rms_norm(cur_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("attn_in", il, xn_, n_embd_);
    if (B.dsa) dsa(B, pos_, n);
    else kda(B, n);
    dbg("attn", il, ao_, n_embd_);
    hc_post(ao_, hcw_, r_, n_embd_, n, st_);

    hc_pre(r_, fp(B.hc_ffn_fn), fp(B.hc_ffn_scale), fp(B.hc_ffn_base), mix_, hcw_, cur_, n_embd_, n, rms_eps_,
           hc_eps_, sinkhorn_, st_);
    rms_norm(cur_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!L.moe) {
        dense_ffn(B, il, n);
        dbg("ffn", il, fo_, n_embd_);
        hc_post(fo_, hcw_, r_, n_embd_, n, st_);
    } else if (!batched(L, n)) {
        moe_router(L, n);
    }
}

// The draft block is one more sparse-attention layer.
void Glm5Next::block_attend(int64_t pos, int n) { dsa(blocks_[(size_t) n_layer_], pos, n); }
void Glm5Next::block_rows(int64_t pos, int n) { dsa_store(blocks_[(size_t) n_layer_], pos, n); }

}  // namespace strata::poly
