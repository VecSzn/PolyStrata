// src/poly/lfm2moe.cpp - the layers of LFM2 with routed experts.  The reference for the math is llama.cpp's lfm2moe
// graph (src/models/lfm2.cpp, which lfm2moe.cpp builds on).
#include "lfm2moe.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr int kTaps = 3;   // what shortconv is written for; checked at load
constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time

// llama.cpp's LLM_TENSOR_OUTPUT_NORM_LFM2: the output's norm is called token_embd_norm in these files
const Names kAlias[] = {
    {"output_norm.weight", {"token_embd_norm.weight", nullptr}},
};

}  // namespace

const Names* Lfm2Moe::tensor_alias(const char* name) const {
    for (const Names& n : kAlias)
        if (std::strcmp(n.name, name) == 0) return &n;
    return nullptr;
}

void Lfm2Moe::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_conv_ = n_attn_ = 0;
}

void Lfm2Moe::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    n_head_ = shape_i("attention.head_count");
    need(n_head_ > 0 && n_embd_ % n_head_ == 0 && n_head_ <= 256, "bad attention.head_count");
    head_dim_ = n_embd_ / n_head_;
    n_rot_ = head_dim_;
    // a layer's key heads: 0 where the layer is a convolution
    const MetaValue* kv = find("attention.head_count_kv");
    need(kv && kv->type == MetaType::ARRAY && (int) kv->items.size() >= n_layer_,
         "attention.head_count_kv has no value for every layer");
    kv_heads_.assign((size_t) n_layer_, 0);
    for (int il = 0; il < n_layer_; ++il) {
        const int h = (int) kv->items[(size_t) il].u;
        need(h == 0 || (n_head_ % h == 0), "a layer's key heads do not divide its query heads");
        need(h == 0 || n_kv_ == 0 || h == n_kv_, "the attention layers have different key heads");
        if (h) n_kv_ = h;
        kv_heads_[(size_t) il] = h;
    }
    need(n_kv_ > 0, "no attention layer");
    need(head_dim_ % 2 == 0 && head_dim_ <= 256, "bad head size");
    need(shape_i("shortconv.l_cache") == kTaps, "shortconv.l_cache is not 3");
    need(shape_i("expert_gating_func") == 2, "the expert gate is not a sigmoid");
    const MetaValue* dense = find("leading_dense_block_count");
    n_dense_ = dense ? (int) dense->u : 0;
    need(n_dense_ >= 0 && n_dense_ < n_layer_, "bad leading_dense_block_count");
    n_ff_ = n_dense_ > 0 ? shape_i("feed_forward_length") : 0;
    const MetaValue* norm = find("expert_weights_norm");
    w_norm_ = norm ? norm->u != 0 : true;   // llama.cpp's graph normalizes them, key or not
    const MetaValue* scale = find("expert_weights_scale");
    w_scale_ = scale ? (float) scale->num() : 0.f;
    theta_scale_ = std::pow(shape_f("rope.freq_base"), -2.f / (float) n_rot_);

    // this architecture puts no limit on a SwiGLU's inputs
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;
}

void Lfm2Moe::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };

    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    B.attn = kv_heads_[(size_t) il] > 0;
    L.fixed = true;   // the attention's launches read the position on the card when they are recorded
    if (B.attn) {
        B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * head_dim_);
        B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
        B.q_norm = f32("attn_q_norm.weight", head_dim_);
        B.k_norm = f32("attn_k_norm.weight", head_dim_);
        B.wo = mat("attn_output.weight", (int64_t) n_head_ * head_dim_, n_embd_);
        ++n_attn_;
    } else {
        B.in_proj = mat("shortconv.in_proj.weight", n_embd_, (int64_t) 3 * n_embd_);
        B.conv = f32("shortconv.conv.weight", kTaps, n_embd_);
        B.wo = mat("shortconv.out_proj.weight", n_embd_, n_embd_);
        ++n_conv_;
    }
    if (il < n_dense_) {
        B.ffn_gate = mat("ffn_gate.weight", n_embd_, n_ff_);
        B.ffn_up = mat("ffn_up.weight", n_embd_, n_ff_);
        B.ffn_down = mat("ffn_down.weight", n_ff_, n_embd_);
        return;
    }
    load_moe(il, false);
    B.probs_b.assign((size_t) n_expert_, 0.f);
    const TensorInfo* b = tensor(il, "exp_probs_b.bias");
    if (!b || b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
        throw std::runtime_error(blk(il, "exp_probs_b.bias") + " is " + (b ? shape_of(*b) : "missing") +
                                 ", expected F32 [" + std::to_string(n_expert_) + "]");
    home(*b).payload(*b);
    std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
    B.probs_b_dev = upload(b, blk(il, "exp_probs_b.bias"), n_expert_, 1, kF32);
}

size_t Lfm2Moe::recurrent_floats() const { return size_t(n_conv_) * 2 * n_embd_; }

int Lfm2Moe::widest() const { return n_head_ * head_dim_; }

// The attention layers' caches, and each convolution layer's two past inputs in state_.
void Lfm2Moe::alloc_caches() {
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t);
    float* st = state_;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.attn) {
            B.keys = (uint16_t*) kv_alloc(row, 1);
            B.vals = (uint16_t*) kv_alloc(row, 1);
        } else {
            B.past = st;
            st += size_t(2) * n_embd_;
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

// Per-token buffers hold rows_ tokens, row after row.
void Lfm2Moe::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * head_dim_, kv = n * n_kv_ * head_dim_;
    proj_ = fbuf(n * 3 * n_embd_);
    cv_ = fbuf(n * n_embd_);
    qraw_ = fbuf(heads);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_n_ = std::max((size_t) chunk_ * n_head_ * opt_.max_context,
                        attend_split_floats(chunk_, n_head_, opt_.max_context, head_dim_));
    score_ = fbuf(score_n_);
    if (rows_ >= kStageRows) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);   // a prompt batch's attention
}

void Lfm2Moe::save_layer(int il, int64_t, uint8_t* dst) {
    const Block& B = blocks_[(size_t) il];
    if (!B.past) return;
    ck(cudaMemcpy(dst + (size_t) (B.past - state_) * sizeof(float), B.past, size_t(2) * n_embd_ * sizeof(float),
                  cudaMemcpyDeviceToHost),
       "save state");
}

size_t Lfm2Moe::card_row_bytes() const { return (size_t) n_attn_ * 2 * n_kv_ * head_dim_ * sizeof(uint16_t); }

void Lfm2Moe::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    for (const Block& B : blocks_) {
        if (!B.keys || bytes == 0) continue;
        for (const uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + at, bytes, cudaMemcpyDeviceToHost), "save rows");
            dst += bytes;
        }
    }
}

void Lfm2Moe::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), at = (size_t) first * row, bytes = (size_t) n * row;
    const size_t skip = (size_t) (first - from) * row, all = (size_t) count * row;
    for (const Block& B : blocks_) {
        if (!B.keys || bytes == 0) continue;
        for (uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + at, src + skip, bytes, cudaMemcpyHostToDevice), "load rows");
            src += all;
        }
    }
}

// The gated short convolution: one projection gives B, C and x; C times the convolution of B * x goes through the
// output projection.
void Lfm2Moe::conv(Block& B, int n) {
    lin(B.in_proj, xn_, proj_, n, xq_);
    if (rows_of_) {   // step(): each row with its own sequence's convolution state
        for (int i = 0; i < n; ++i)
            shortconv(proj_ + (size_t) i * 3 * n_embd_, seq_state(rows_of_[i].seq, B.past), fp(B.conv),
                      cv_ + (size_t) i * n_embd_, n_embd_, 1, st_);
    } else {
        shortconv(proj_, B.past, fp(B.conv), cv_, n_embd_, n, st_, snap_bits_ ? snap_at_ + (B.past - state_) : nullptr,
                  rec_n_, snap_bits_);
    }
    lin(B.wo, cv_, ao_, n);
}

// The attention: queries and keys normed a head and turned by the position, the keys and values into the caches.
void Lfm2Moe::attn(Block& B, int64_t pos, int n) {
    if (rows_of_) return attn_rows(B, n);
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
    lin(B.wq, xn_, qraw_, n, xq_);
    head_norm_rope(qraw_, head_dim_, fp(B.q_norm), q_, head_dim_, n_head_, n_rot_, theta_scale_, pos, n, rms_eps_, st_,
                   nullptr, nullptr, 1, at ? at + 1 : nullptr);
    lin(B.wk, xn_, kraw_, n, xq_);
    head_norm_rope(kraw_, head_dim_, fp(B.k_norm), k_, head_dim_, n_kv_, n_rot_, theta_scale_, pos, n, rms_eps_, st_,
                   nullptr, nullptr, 1, at ? at + 1 : nullptr);
    lin(B.wv, xn_, v_, n, xq_);
    if (at) {
        store_f16_at(k_, B.keys, n_kv_ * head_dim_, at, n, 0, st_);
        store_f16_at(v_, B.vals, n_kv_ * head_dim_, at, n, 0, st_);
    } else {
        store_f16(k_, B.keys, n_kv_ * head_dim_, pos, n, st_);
        store_f16(v_, B.vals, n_kv_ * head_dim_, pos, n, st_);
    }
    const float scale = 1.f / std::sqrt((float) head_dim_);
    // a prompt batch through cuBLAS; a few tokens with the kernels that give a token the same bits in any company
    if (n >= kStageRows)
        attend_batch(blas_, q_, B.keys, B.vals, pos, n, score_, (size_t) chunk_ * n_head_ * opt_.max_context, awork_,
                     att_, head_dim_, n_head_, n_kv_, scale, st_);
    else
        attend(q_, B.keys, B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_, n_kv_, scale, st_, 0, 0, false,
               at, at ? room(n) : 0, score_n_);
    lin(B.wo, att_, ao_, n);
}

// step()'s rows, each of its own sequence: the products for all of them at once, and a row at a time the turn (it
// takes one position and counts on from it), the stored keys and values and the attention, in the row's sequence.
void Lfm2Moe::attn_rows(Block& B, int n) {
    lin(B.wq, xn_, qraw_, n, xq_);
    lin(B.wk, xn_, kraw_, n, xq_);
    lin(B.wv, xn_, v_, n, xq_);
    const float scale = 1.f / std::sqrt((float) head_dim_);
    const size_t kv = (size_t) n_kv_ * head_dim_, hq = (size_t) n_head_ * head_dim_;
    for (int i = 0; i < n; ++i) {
        const Row& r = rows_of_[i];
        const long long* ai = at_ + 2 * i;
        head_norm_rope(qraw_ + i * hq, head_dim_, fp(B.q_norm), q_ + i * hq, head_dim_, n_head_, n_rot_, theta_scale_, 0,
                       1, rms_eps_, st_, nullptr, nullptr, 1, ai + 1);
        head_norm_rope(kraw_ + i * kv, head_dim_, fp(B.k_norm), k_ + i * kv, head_dim_, n_kv_, n_rot_, theta_scale_, 0,
                       1, rms_eps_, st_, nullptr, nullptr, 1, ai + 1);
        uint16_t* kc = (uint16_t*) seq_buf(r.seq, B.keys);
        uint16_t* vc = (uint16_t*) seq_buf(r.seq, B.vals);
        store_f16_at(k_ + i * kv, kc, (int) kv, ai, 1, 0, st_);
        store_f16_at(v_ + i * kv, vc, (int) kv, ai, 1, 0, st_);
        const int64_t keep = pos_;
        pos_ = r.pos;
        const int sized = room(1);
        pos_ = keep;
        attend(q_ + i * hq, kc, vc, r.pos, 1, score_, chunk_, att_ + i * hq, head_dim_, n_head_, n_kv_, scale, st_, 0, 0,
               false, ai, sized, score_n_);
    }
    lin(B.wo, att_, ao_, n);
}

// The recordings of an attention layer's head are numbered by the rows they are sized for.
int Lfm2Moe::head_key(int il, int n) const { return blocks_[(size_t) il].attn ? room_key(n) : 0; }

void Lfm2Moe::dense_ffn(Block& B, int il, int n) {
    lin(B.ffn_gate, xn_, g_, n, xq_);
    lin(B.ffn_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_, clamp_shexp_[(size_t) il], st_);
    lin(B.ffn_down, h_, fo_, n);
}

// A layer from its input to its router's copy (a dense layer: to its end).  A prompt batch's MoE (moe_rows) does
// its own routing.  A norm and its quantization are one launch, and so are the residual's sum and the next norm:
// a token's step is mostly small launches.
void Lfm2Moe::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    const int threads = sum_threads(n_embd_);
    kernels::native_add_rms_norm_q8_1(r_, nullptr, fp(L.attn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("attn_in", il, xn_, n_embd_);
    if (B.attn) attn(B, pos_, n);
    else conv(B, n);
    dbg("attn", il, ao_, n_embd_);

    kernels::native_add_rms_norm_q8_1(r_, ao_, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!L.moe) {
        dense_ffn(B, il, n);
        dbg("ffn", il, fo_, n_embd_);
        axpy(fo_, 1.f, r_, n * n_embd_, true, st_);
    } else if (!batched(L, n)) {
        moe_router(L, n);
    }
}

void Lfm2Moe::layer_tail(Layer&, int, int n) { axpy(fo_, 1.f, r_, n * n_embd_, true, st_); }

void Lfm2Moe::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// What ggml's MoE graph does for this model: sigmoid gates, the n_used largest of gate + bias, and weights from
// the bare gates, made to sum to one.
void Lfm2Moe::route(int il, const float* logits, int32_t, int* ids, float* w, float* probs) const {
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

bool Lfm2Moe::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, 1, 1.f, fp(B.probs_b_dev), nullptr, w_norm_, w_scale_, ids, w, st_);
    return true;
}

// No shared expert: the routed experts' sum starts from nothing.
void Lfm2Moe::moe_shared(const Layer&, int, int n) {
    ck(cudaMemsetAsync(fo_, 0, (size_t) n * n_embd_ * sizeof(float), cs(st_)), "experts");
}

}  // namespace strata::poly
