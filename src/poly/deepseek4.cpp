// src/poly/deepseek4.cpp - DeepSeek V4's layers.  The reference for the math is llama.cpp's deepseek4 graph
// (src/models/deepseek4.cpp) and its compressed caches (src/llama-kv-cache-dsv4.cpp).
#include "deepseek4.hpp"

#include "runtime.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr int kHc = 4;   // what kernels.cu is written for; checked at load
constexpr size_t kScoreBytes = size_t(128) << 20;   // a prompt batch's indexer scores at most
constexpr float kMinWeightSum = 6.103515625e-5f;   // the floor ggml's MoE graph puts under the gate sum
constexpr uint32_t kTypeI32 = 26;
constexpr float kPi = 3.14159265358979323846f;

// ggml_rope_ext's arguments as the deepseek4 graph gives them, and ggml_rope_yarn_corr_dims().
Rope make_rope(float base, int n_rot, float freq_scale, float ext_factor, int n_ctx_orig, float beta_fast,
               float beta_slow) {
    Rope r;
    r.theta_scale = powf(base, -2.0f / (float) n_rot);
    r.freq_scale = freq_scale;
    r.ext_factor = ext_factor;
    r.attn_factor = ext_factor == 0.0f ? 1.0f : 1.0f / (1.0f + 0.1f * logf(1.0f / freq_scale));
    const auto corr = [&](float rot) {
        return (float) n_rot * logf((float) n_ctx_orig / (rot * 2 * kPi)) / (2 * logf(base));
    };
    r.corr0 = std::max(0.0f, floorf(corr(beta_fast)));
    r.corr1 = std::min((float) (n_rot - 1), ceilf(corr(beta_slow)));
    return r;
}

// The positions [pos - keep, pos) as runs of a ring's rows: f(the run's first slot among the keep, its first ring
// row, its rows).  A position before the sequence's start has no run.
template <class F>
void ring_runs(int64_t pos, int keep, int ring, F&& f) {
    for (int64_t p = std::max<int64_t>(0, pos - keep); p < pos;) {
        const int64_t at = p % ring, m = std::min<int64_t>(pos - p, ring - at);
        f((size_t) (p - (pos - keep)), (size_t) at, (size_t) m);
        p += m;
    }
}

}  // namespace

void DeepSeek4::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    hc_head_fn_ = hc_head_scale_ = hc_head_base_ = Dev{};
}

int DeepSeek4::streams() const { return kHc; }

// The routed experts' sum goes back into the residual streams.
void DeepSeek4::layer_tail(Layer&, int, int n) { hc_post(fo_, hcw_, r_, n_embd_, n, st_); }

// The weights a layer's head mixes the streams with are the ones its tail puts the experts' sum back with.
void DeepSeek4::shift_rows(ptrdiff_t by) {
    Engine::shift_rows(by);
    hcw_ += by * (kHc + kHc * kHc);
}

// The model's output for `rows` tokens from row `first` of the call: the streams mixed by their own weights, normed.
void DeepSeek4::output_rows(int first, int rows, float* h) {
    hc_head(r_ + (size_t) first * kHc * n_embd_, fp(hc_head_fn_), fp(hc_head_scale_), fp(hc_head_base_), mix_, cur_,
            n_embd_, rows, rms_eps_, hc_eps_, st_);
    rms_norm(cur_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

void DeepSeek4::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    n_head_ = shape_i("attention.head_count");
    head_dim_ = shape_i("attention.key_length");
    n_rot_ = shape_i("rope.dimension_count");
    window_ = shape_i("attention.sliding_window");
    q_rank_ = shape_i("attention.q_lora_rank");
    o_groups_ = shape_i("attention.output_group_count");
    o_rank_ = shape_i("attention.output_lora_rank");
    idx_heads_ = shape_i("attention.indexer.head_count");
    idx_dim_ = shape_i("attention.indexer.key_length");
    idx_top_k_ = shape_i("attention.indexer.top_k");
    sinkhorn_ = shape_i("hyper_connection.sinkhorn_iterations");
    hc_eps_ = shape_f("hyper_connection.epsilon");
    hash_layers_ = shape_i("hash_layer_count");
    w_scale_ = shape_f("expert_weights_scale");
    w_norm_ = shape_i("expert_weights_norm") != 0;
    need(shape_i("hyper_connection.count") == kHc, "hyper_connection.count is not 4");
    need(shape_i("attention.head_count_kv") == 1, "more than one key head");
    need(shape_i("attention.value_length") == head_dim_, "the values are not as long as the keys");
    need(shape_i("expert_gating_func") == 4, "the expert gate is not sqrt(softplus)");
    need(shape_i("expert_shared_count") == 1, "expert_shared_count is not 1");
    need(head_dim_ % 2 == 0 && head_dim_ <= 2048 && n_rot_ % 2 == 0 && n_rot_ <= head_dim_ && n_rot_ <= idx_dim_,
         "bad attention.key_length or rope.dimension_count");
    need(idx_dim_ == 128 && idx_heads_ > 0 && idx_top_k_ > 0, "the indexer's keys are not 128 values");
    need(o_groups_ > 0 && n_head_ % o_groups_ == 0 && window_ > 0, "bad attention.output_group_count or window");

    const MetaValue* ratios = find("attention.compress_ratios");
    need(ratios && ratios->type == MetaType::ARRAY && (int) ratios->items.size() >= n_layer_,
         "attention.compress_ratios has no value for every layer");
    ratio_.assign((size_t) n_layer_, 0);
    for (int il = 0; il < n_layer_; ++il) {
        const int r = (int) ratios->items[(size_t) il].u;
        need(r == 0 || r == 4 || r == 128, "a compress ratio other than 0, 4 and 128");
        ratio_[(size_t) il] = r;
    }

    // The window layers turn by rope.freq_base alone.  The compressed ones, their compressed rows and the indexer
    // turn by the compressor's base with the file's YaRN scaling.
    rope_raw_ = make_rope(shape_f("rope.freq_base"), n_rot_, 1.0f, 0.0f, 0, 0.0f, 0.0f);
    const MetaValue* kind = find("rope.scaling.type");
    const MetaValue* factor = find("rope.scaling.factor");
    const MetaValue* orig = find("rope.scaling.original_context_length");
    const MetaValue* fast = find("rope.scaling.yarn_beta_fast");
    const MetaValue* slow = find("rope.scaling.yarn_beta_slow");
    const bool yarn_on = kind && kind->s == "yarn";
    const float freq_scale = factor && factor->f > 0 ? (float) (1.0 / factor->f) : 1.0f;
    rope_comp_ = make_rope(shape_f("attention.compress_rope_freq_base"), n_rot_, freq_scale, yarn_on ? 1.0f : 0.0f,
                           orig ? (int) orig->u : 0, fast ? (float) fast->f : 32.0f, slow ? (float) slow->f : 1.0f);

    clamp_exp_ = clamp_limits(find("swiglu_clamp_exp"), n_layer_);
    clamp_shexp_ = find("swiglu_clamp_shexp") ? clamp_limits(find("swiglu_clamp_shexp"), n_layer_) : clamp_exp_;
}

void DeepSeek4::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a, int64_t b = 1) { return upload(tensor(il, n), blk(il, n), a, b, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    const int n_mix = kHc * (2 + kHc);

    B.ratio = ratio_[(size_t) il];
    B.rope = B.ratio ? rope_comp_ : rope_raw_;
    L.fixed = true;   // the launches read the position on the card when recorded; head_key() numbers their sizes
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    B.hc_attn_fn = f32("hc_attn_fn.weight", kHc * n_embd_, n_mix);
    B.hc_attn_scale = f32("hc_attn_scale.weight", 3);
    B.hc_attn_base = f32("hc_attn_base.weight", n_mix);
    B.hc_ffn_fn = f32("hc_ffn_fn.weight", kHc * n_embd_, n_mix);
    B.hc_ffn_scale = f32("hc_ffn_scale.weight", 3);
    B.hc_ffn_base = f32("hc_ffn_base.weight", n_mix);

    B.q_a = mat("attn_q_a.weight", n_embd_, q_rank_);
    B.q_a_norm = f32("attn_q_a_norm.weight", q_rank_);
    B.q_b = mat("attn_q_b.weight", q_rank_, (int64_t) n_head_ * head_dim_);
    B.wkv = mat("attn_kv.weight", n_embd_, head_dim_);
    B.kv_norm = f32("attn_kv_a_norm.weight", head_dim_);
    B.wo_a = mat("attn_output_a.weight", (int64_t) n_head_ * head_dim_ / o_groups_, (int64_t) o_groups_ * o_rank_);
    B.wo_b = mat("attn_output_b.weight", (int64_t) o_groups_ * o_rank_, n_embd_);
    B.sinks = f32("attn_sinks.weight", n_head_);
    auto squeeze_of = [&](Squeeze& S, const char* kv, const char* gate, const char* ape, const char* norm, int dim) {
        S.dim = dim;
        S.width = B.ratio == 4 ? 2 * dim : dim;   // the 4-to-1 compressors overlap the block before
        S.wkv = mat(kv, n_embd_, S.width);
        S.wgate = mat(gate, n_embd_, S.width);
        S.ape = f32(ape, S.width, B.ratio);
        S.norm = f32(norm, dim);
    };
    if (B.ratio)
        squeeze_of(B.comp, "attn_compressor_kv.weight", "attn_compressor_gate.weight", "attn_compressor_ape.weight",
                   "attn_compressor_norm.weight", head_dim_);
    if (B.ratio == 4) {
        squeeze_of(B.idx, "indexer_compressor_kv.weight", "indexer_compressor_gate.weight",
                   "indexer_compressor_ape.weight", "indexer_compressor_norm.weight", idx_dim_);
        B.idx_q_b = mat("indexer.attn_q_b.weight", q_rank_, (int64_t) idx_heads_ * idx_dim_);
        B.idx_proj = mat("indexer.proj.weight", n_embd_, idx_heads_);
    }

    load_moe(il);
    B.probs_b.assign((size_t) n_expert_, 0.f);
    if (const TensorInfo* b = tensor(il, "exp_probs_b.bias")) {
        if (b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
            throw std::runtime_error(b->name + " is " + shape_of(*b) + ", expected F32 [" +
                                     std::to_string(n_expert_) + "]");
        home(*b).payload(*b);
        std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
        B.probs_b_dev = upload(b, blk(il, "exp_probs_b.bias"), n_expert_, 1, kF32);
    }
    if (il < hash_layers_) {
        const TensorInfo* t = tensor(il, "ffn_gate_tid2eid.weight");
        if (!t || t->type != kTypeI32 || t->shape.size() != 2 || (int64_t) t->shape[0] != n_used_ ||
            (int64_t) t->shape[1] < n_vocab_)
            throw std::runtime_error(blk(il, "ffn_gate_tid2eid.weight") + " is " + (t ? shape_of(*t) : "missing") +
                                     ", expected I32 [" + std::to_string(n_used_) + ", " + std::to_string(n_vocab_) +
                                     "]");
        home(*t).payload(*t);
        B.by_token.resize((size_t) n_used_ * n_vocab_);
        std::memcpy(B.by_token.data(), home(*t).data(*t), B.by_token.size() * sizeof(int32_t));
        for (int32_t e : B.by_token)
            if (e < 0 || e >= n_expert_) throw std::runtime_error(t->name + " names an expert the layer has not");
    }

    if (il == n_layer_ - 1) {   // the output's read of the streams, on the card that has the output
        auto top = [&](const char* n, int64_t a, int64_t b = 1) { return upload(gg_->find(n), n, a, b, kF32); };
        hc_head_fn_ = top("output_hc_fn.weight", kHc * n_embd_, kHc);
        hc_head_scale_ = top("output_hc_scale.weight", 1);
        hc_head_base_ = top("output_hc_base.weight", kHc);
    }
}

int DeepSeek4::widest() const {
    return std::max({q_rank_, o_groups_ * o_rank_, n_head_ * head_dim_ / o_groups_});
}

// A window ring for every layer, two rings for each compressor, and the compressed rows, which grow with the
// sequence: a row for each block of ratio positions.
void DeepSeek4::alloc_caches() {
    raw_ring_ = std::min(opt_.max_context, window_ + rows_);
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        B.raw = (uint16_t*) seq_alloc((size_t) raw_ring_ * head_dim_ * sizeof(uint16_t), "the attention cache");
        if (!B.ratio) continue;
        // a call's rows and the ones before it that its first block reads: the block's own, and with the overlap
        // the block before
        B.ring = rows_ + (B.ratio == 4 ? 2 : 1) * B.ratio;
        for (Squeeze* S : {&B.comp, &B.idx}) {
            if (!S->dim) continue;
            S->kv = (float*) seq_alloc((size_t) B.ring * S->width * sizeof(float), "the compressor's rows");
            S->score = (float*) seq_alloc((size_t) B.ring * S->width * sizeof(float), "the compressor's rows");
        }
        B.rows = (uint16_t*) kv_alloc((size_t) head_dim_ * sizeof(uint16_t), B.ratio);
        if (B.ratio == 4) B.keys = (uint16_t*) kv_alloc((size_t) idx_dim_ * sizeof(uint16_t), 4);
    }
    score_stride_ = opt_.max_context / 4 + 1;
    score_rows_ = (int) std::clamp<size_t>(kScoreBytes / (score_stride_ * sizeof(float)), 1, (size_t) rows_);
    if (!sel_) sel_ = (int32_t*) dmalloc((size_t) rows_ * idx_top_k_ * sizeof(int32_t), "scratch");
}

// Per-token buffers hold rows_ tokens, row after row.
void DeepSeek4::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * n_head_ * head_dim_;
    mix_ = fbuf(n * (kHc * (2 + kHc) + 1));
    hcw_ = fbuf(n * (kHc + kHc * kHc));
    t1_ = fbuf(n * q_rank_);
    qr_ = fbuf(n * q_rank_);
    q_ = fbuf(heads);   // the queries, and after the attention its output in groups (out_proj())
    kv_ = fbuf(n * head_dim_);
    att_ = fbuf(heads);
    oa_ = fbuf(n * o_groups_ * o_rank_);
    ob_ = fbuf(n * o_groups_ * o_rank_);
    ckv_ = fbuf(n * 2 * head_dim_);
    cscore_ = fbuf(n * 2 * head_dim_);
    iq_ = fbuf(n * idx_heads_ * idx_dim_);
    iw_ = fbuf(n * idx_heads_);
    score_ = fbuf((size_t) score_rows_ * score_stride_);
}

// A compressor's rows for n tokens from position `pos`, and the compressed row of each block they finish: normed,
// turned by the block's first position and stored as fp16.  xn_ and xq_ hold the layer's input.
void DeepSeek4::squeeze(Block& B, Squeeze& S, int64_t pos, int n) {
    const long long* at = recording_ ? at_ : nullptr;
    pair(S.wkv, S.wgate, ckv_, cscore_, n);
    dsv4_state(ckv_, cscore_, fp(S.ape), S.kv, S.score, B.ring, S.width, B.ratio, pos, n, st_, at);
    dsv4_squeeze(S.kv, S.score, B.ring, S.dim, B.ratio, S.width != S.dim, fp(S.norm), rms_eps_, rope_comp_, n_rot_,
                 pos, n, &S == &B.comp ? B.rows : B.keys, st_, at);
}

// A layer with compressed rows reads more of them as the sequence grows: its recordings are numbered by the rows
// they are sized for.
int DeepSeek4::head_key(int il, int n) const { return blocks_[(size_t) il].ratio ? room_key(n) : 0; }

// The output projection: the heads in groups, each group through its own low-rank matrix, then all of them through
// one.  A group of a token is n_head / groups heads that lie together.
void DeepSeek4::out_proj(Block& B, int n) {
    const int in = n_head_ * head_dim_ / o_groups_;
    const size_t row = ggml_row_size((ggml_type) B.wo_a.type, in);
    if (n == 1 && o_groups_ <= 8 && kernels::native_mmvq_supported((int) B.wo_a.type)) {
        // a token's groups lie one behind the other: quantized as one vector, then every group in one launch
        kernels::native_quantize_q8_1(att_, q8_, n_head_ * head_dim_, 1, st_);
        kernels::MmvqSlots slots{};
        const size_t xrow = kernels::native_q8_1_bytes(in);
        for (int g = 0; g < o_groups_; ++g) {
            slots.w[g] = (const uint8_t*) B.wo_a.p + (size_t) g * o_rank_ * row;
            slots.x[g] = (const uint8_t*) q8_ + (size_t) g * xrow;
            slots.y[g] = oa_ + (size_t) g * o_rank_;
        }
        if (kernels::native_mmvq_slots((int) B.wo_a.type, slots, o_groups_, in, o_rank_, st_))
            return lin(B.wo_b, oa_, ao_, 1);
    }
    const float* x = att_;
    if (n > 1) {   // a group's tokens together, for its matrix
        permute_groups(att_, q_, in, o_groups_, n, false, st_);
        x = q_;
    }
    for (int g = 0; g < o_groups_; ++g) {
        Dev w = B.wo_a;
        w.p = (uint8_t*) B.wo_a.p + (size_t) g * o_rank_ * row;
        w.n_out = o_rank_;
        lin(w, x + (size_t) g * n * in, oa_ + (size_t) g * n * o_rank_, n);
    }
    const float* y = oa_;
    if (n > 1) {
        permute_groups(oa_, ob_, o_rank_, o_groups_, n, true, st_);
        y = ob_;
    }
    lin(B.wo_b, y, ao_, n);
}

// Two matrices of the layer's input (xq_): for a token alone in one launch when they are of a shape and a format.
void DeepSeek4::pair(const Dev& a, const Dev& b, float* ya, float* yb, int n) {
    if (n == 1 && a.type == b.type && a.n_in == b.n_in && a.n_out == b.n_out &&
        kernels::native_mmvq_supported((int) a.type)) {
        kernels::MmvqSlots slots{};
        slots.w[0] = a.p, slots.w[1] = b.p;
        slots.x[0] = slots.x[1] = xq_;
        slots.y[0] = ya, slots.y[1] = yb;
        if (kernels::native_mmvq_slots((int) a.type, slots, 2, a.n_in, a.n_out, st_)) return;
    }
    lin(a, xn_, ya, n, xq_);
    lin(b, xn_, yb, n, xq_);
}

// The queries, and the key the token leaves in the window ring, are turned on their last n_rot values; the queries
// are normed a head without weights first.  The heads' output is turned back by the query's position.
void DeepSeek4::attention(Block& B, int64_t pos, int n) {
    if (rows_of_) {
        // step(): a row at a time, the whole attention, with the row's sequence's window ring, compressed rows,
        // indexer keys and compressors, at the row's buffers, position and recorded position
        const Row* rows = std::exchange(rows_of_, nullptr);
        float *const xn = xn_, *const ao = ao_;
        void* const xq = xq_;
        long long* const at = at_;
        const int64_t keep = pos_;
        const size_t q8 = kernels::native_q8_1_bytes(n_embd_);
        for (int i = 0; i < n; ++i) {
            const int s = rows[i].seq;
            Block one = B;
            one.raw = (uint16_t*) seq_buf(s, B.raw);
            if (B.rows) one.rows = (uint16_t*) seq_buf(s, B.rows);
            if (B.keys) one.keys = (uint16_t*) seq_buf(s, B.keys);
            for (Squeeze* S : {&one.comp, &one.idx})
                if (S->dim) {
                    S->kv = (float*) seq_buf(s, S->kv);
                    S->score = (float*) seq_buf(s, S->score);
                }
            xn_ = xn + (size_t) i * n_embd_;
            xq_ = (uint8_t*) xq + (size_t) i * q8;
            ao_ = ao + (size_t) i * n_embd_;
            at_ = at + 2 * i;
            pos_ = rows[i].pos;
            attention(one, rows[i].pos, 1);
        }
        xn_ = xn;
        xq_ = xq;
        ao_ = ao;
        at_ = at;
        pos_ = keep;
        rows_of_ = rows;
        return;
    }
    const int dim = head_dim_, off = dim - n_rot_;
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
    const int sized = at ? room(n) : 0;
    lin(B.q_a, xn_, t1_, n, xq_);
    rms_norm(t1_, fp(B.q_a_norm), qr_, q_rank_, n, rms_eps_, st_);
    lin(B.q_b, qr_, q_, n);
    dsv4_rope(q_, q_, dim, n_head_, off, n_rot_, B.rope, pos, 1, n, true, rms_eps_, false, st_, at);
    lin(B.wkv, xn_, kv_, n, xq_);
    rms_norm(kv_, fp(B.kv_norm), kv_, dim, n, rms_eps_, st_);
    dsv4_rope(kv_, kv_, dim, 1, off, n_rot_, B.rope, pos, 1, n, false, 0.f, false, st_, at);
    if (at) {
        store_f16_at(kv_, B.raw, dim, at, n, raw_ring_, st_);
    } else {
        const int row = (int) (pos % raw_ring_), head = std::min(n, raw_ring_ - row);
        store_f16(kv_, B.raw, dim, row, head, st_);
        if (head < n) store_f16(kv_ + (size_t) head * dim, B.raw, dim, 0, n - head, st_);
    }

    // The compressed rows the tokens finish are there before they attend: a token reads its own block's.
    if (B.ratio) squeeze(B, B.comp, pos, n);
    const float scale = 1.f / std::sqrt((float) dim);
    if (B.ratio == 4) {
        squeeze(B, B.idx, pos, n);
        lin(B.idx_q_b, qr_, iq_, n);
        dsv4_rope(iq_, iq_, idx_dim_, idx_heads_, idx_dim_ - n_rot_, n_rot_, rope_comp_, pos, 1, n, false, 0.f, false,
                  st_, at);
        lin(B.idx_proj, xn_, iw_, n);
        const float iscale = 1.f / std::sqrt((float) (idx_dim_ * idx_heads_));
        // score_ has room for some of a batch's tokens at a long context; they take turns.  A token that sees
        // top_k compressed rows or fewer takes them all and needs no scores; a recorded launch cannot tell.
        for (int t = 0, c = 0; t < n; t += c) {
            c = std::min(score_rows_, n - t);
            const long long* at_t = at ? at + 2 * t : nullptr;
            if (at || (pos + t + c) / 4 > idx_top_k_)
                dsv4_idx_score(iq_ + (size_t) t * idx_heads_ * idx_dim_, iw_ + (size_t) t * idx_heads_, B.keys, score_,
                               score_stride_, idx_dim_, idx_heads_, 4, pos + t, c, iscale, st_, at_t, sized);
            dsv4_select(score_, score_stride_, sel_ + (size_t) t * idx_top_k_, idx_top_k_, idx_top_k_, 4, pos + t, c,
                        st_, at_t);
        }
        dsv4_attend(q_, B.raw, raw_ring_, window_, B.rows, 4, sel_, idx_top_k_, idx_top_k_, fp(B.sinks), att_, dim,
                    n_head_, scale, pos, n, st_, at, sized);
    } else {
        dsv4_attend(q_, B.raw, raw_ring_, window_, B.ratio ? B.rows : nullptr, std::max(B.ratio, 1), nullptr, 0, 0,
                    fp(B.sinks), att_, dim, n_head_, scale, pos, n, st_, at, sized);
    }
    dsv4_rope(att_, att_, dim, n_head_, off, n_rot_, B.rope, pos, 1, n, false, 0.f, true, st_, at);
    out_proj(B, n);
}

// What ggml's MoE graph does for this model: sqrt(softplus) of the router's scores; the n_used largest of those
// plus the layer's bias, or in the first layers the token's own experts; their weights summed to one and scaled.
void DeepSeek4::route(int il, const float* logits, int32_t token, int* ids, float* w, float* probs) const {
    const Block& B = blocks_[(size_t) il];
    for (int e = 0; e < n_expert_; ++e) {
        const float x = logits[e];
        probs[e] = std::sqrt(x > 20.0f ? x : std::log(1.0f + std::exp(x)));
    }
    float sum = 0.f;
    if (!B.by_token.empty() && token >= 0 && token < n_vocab_) {
        for (int k = 0; k < n_used_; ++k) {
            ids[k] = B.by_token[(size_t) token * n_used_ + k];
            w[k] = probs[ids[k]];
            sum += w[k];
        }
    } else {
        float* pick = probs + n_expert_;
        for (int e = 0; e < n_expert_; ++e) pick[e] = probs[e] + B.probs_b[(size_t) e];
        for (int k = 0; k < n_used_; ++k) {
            int best = 0;
            for (int e = 1; e < n_expert_; ++e)
                if (pick[e] > pick[best]) best = e;
            pick[best] = -INFINITY;
            ids[k] = best;
            w[k] = probs[best];
            sum += w[k];
        }
    }
    const float denom = w_norm_ ? std::max(sum, kMinWeightSum) : 1.f;
    for (int k = 0; k < n_used_; ++k) {
        w[k] /= denom;
        if (w_scale_ != 0.f) w[k] *= w_scale_;
    }
}

bool DeepSeek4::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    if (!B.by_token.empty()) return false;
    route_rows(rl_, n_expert_, n_used_, n, 2, 1.f, fp(B.probs_b_dev), nullptr, w_norm_, w_scale_, ids, w, st_);
    return true;
}

// The shared expert.  It runs on the GPU while the host routes.
void DeepSeek4::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_shexp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);
}

// A layer from its input to its router's copy.  A prompt batch's MoE (moe_rows) does its own routing.
void DeepSeek4::layer_head(Layer& L, int il, int n) {
    Block& B = blocks_[(size_t) il];
    hc_pre(r_, fp(B.hc_attn_fn), fp(B.hc_attn_scale), fp(B.hc_attn_base), mix_, hcw_, cur_, n_embd_, n, rms_eps_,
           hc_eps_, sinkhorn_, st_);
    rms_norm(cur_, fp(L.attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("attn_in", il, xn_, n_embd_);
    attention(B, pos_, n);
    dbg("attn", il, ao_, n_embd_);
    hc_post(ao_, hcw_, r_, n_embd_, n, st_);

    hc_pre(r_, fp(B.hc_ffn_fn), fp(B.hc_ffn_scale), fp(B.hc_ffn_base), mix_, hcw_, cur_, n_embd_, n, rms_eps_,
           hc_eps_, sinkhorn_, st_);
    rms_norm(cur_, fp(L.ffn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!batched(L, n)) moe_router(L, n);
}

// The positions before the next one whose compressor rows a block still needs: its own, and with the overlap the
// block before.
int DeepSeek4::keep_rows(const Block& B) const { return B.ratio == 4 ? 2 * B.ratio - 1 : B.ratio - 1; }

// A layer's part of a state: the window's rows, and its compressors' rows of the blocks not done yet.
size_t DeepSeek4::layer_extra(const Block& B) const {
    if (!B.raw) return 0;   // another card's layer
    size_t b = (size_t) window_ * head_dim_ * sizeof(uint16_t);
    for (const Squeeze* S : {&B.comp, &B.idx})
        if (S->dim) b += (size_t) keep_rows(B) * S->width * 2 * sizeof(float);
    return b;
}

size_t DeepSeek4::extra_bytes() const {
    size_t b = 0;
    for (const Block& B : blocks_) b += layer_extra(B);
    return b;
}

// A layer's part, behind the recurrent state (which this architecture has none of) and the layers before it.  A
// position before the sequence's start leaves zeros.
void DeepSeek4::save_layer(int il, int64_t pos, uint8_t* dst) {
    dst += state_n_ * sizeof(float);
    for (int j = 0; j < il; ++j) dst += layer_extra(blocks_[(size_t) j]);
    const Block& B = blocks_[(size_t) il];
    if (!B.raw) return;
    const size_t raw_row = (size_t) head_dim_ * sizeof(uint16_t);
    std::memset(dst, 0, layer_extra(B));
    ring_runs(pos, window_, raw_ring_, [&](size_t slot, size_t at, size_t m) {
        ck(cudaMemcpy(dst + slot * raw_row, (const uint8_t*) B.raw + at * raw_row, m * raw_row,
                      cudaMemcpyDeviceToHost),
           "save state");
    });
    dst += (size_t) window_ * raw_row;
    for (const Squeeze* S : {&B.comp, &B.idx}) {
        if (!S->dim) continue;
        const size_t row = (size_t) S->width * sizeof(float);
        for (const float* ring : {S->kv, S->score}) {
            ring_runs(pos, keep_rows(B), B.ring, [&](size_t slot, size_t at, size_t m) {
                ck(cudaMemcpy(dst + slot * row, (const uint8_t*) ring + at * row, m * row, cudaMemcpyDeviceToHost),
                   "save state");
            });
            dst += (size_t) keep_rows(B) * row;
        }
    }
}

void DeepSeek4::load_extra(const uint8_t* src, int64_t pos) {
    const size_t raw_row = (size_t) head_dim_ * sizeof(uint16_t);
    for (const Block& B : blocks_) {
        if (!B.raw) continue;
        ring_runs(pos, window_, raw_ring_, [&](size_t slot, size_t at, size_t m) {
            ck(cudaMemcpy((uint8_t*) B.raw + at * raw_row, src + slot * raw_row, m * raw_row, cudaMemcpyHostToDevice),
               "load state");
        });
        src += (size_t) window_ * raw_row;
        for (const Squeeze* S : {&B.comp, &B.idx}) {
            if (!S->dim) continue;
            const size_t row = (size_t) S->width * sizeof(float);
            for (float* ring : {S->kv, S->score}) {
                ring_runs(pos, keep_rows(B), B.ring, [&](size_t slot, size_t at, size_t m) {
                    ck(cudaMemcpy((uint8_t*) ring + at * row, src + slot * row, m * row, cudaMemcpyHostToDevice),
                       "load state");
                });
                src += (size_t) keep_rows(B) * row;
            }
        }
    }
}

// A row: the compressed rows of the blocks it ends, which the row that ends a block has and the others have room
// for.  The window's rows are part of the state (save_layer()).
size_t DeepSeek4::card_row_bytes() const {
    size_t b = 0;
    for (const Block& B : blocks_) {
        if (!B.raw || !B.ratio) continue;
        b += (size_t) head_dim_ * sizeof(uint16_t);
        if (B.ratio == 4) b += (size_t) idx_dim_ * sizeof(uint16_t);
    }
    return b;
}

void DeepSeek4::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    for (const Block& B : blocks_) {
        if (!B.raw || !B.ratio || n == 0) continue;
        const int64_t r = B.ratio, b0 = (first + r) / r - 1, b1 = (first + n) / r;   // the blocks these rows end
        for (const auto& [cache, width] :
             {std::pair{(const uint16_t*) B.rows, head_dim_}, std::pair{(const uint16_t*) B.keys, idx_dim_}}) {
            if (!cache) continue;
            const size_t key = (size_t) width * sizeof(uint16_t);
            std::memset(dst, 0, (size_t) n * key);
            if (b1 > b0)
                ck(cudaMemcpy2D(dst + (size_t) (b0 * r + r - 1 - first) * key, (size_t) r * key,
                                (const uint8_t*) cache + (size_t) b0 * key, key, key, (size_t) (b1 - b0),
                                cudaMemcpyDeviceToHost),
                   "save rows");
            dst += (size_t) n * key;
        }
    }
}

void DeepSeek4::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    for (const Block& B : blocks_) {
        if (!B.raw || !B.ratio) continue;
        const int64_t r = B.ratio, b0 = (first + r) / r - 1, b1 = (first + n) / r;
        for (const auto& [cache, width] : {std::pair{B.rows, head_dim_}, std::pair{B.keys, idx_dim_}}) {
            if (!cache) continue;
            const size_t key = (size_t) width * sizeof(uint16_t);
            if (b1 > b0)
                ck(cudaMemcpy2D((uint8_t*) cache + (size_t) b0 * key, key, src + (size_t) (b0 * r + r - 1 - from) * key,
                                (size_t) r * key, key, (size_t) (b1 - b0), cudaMemcpyHostToDevice),
                   "load rows");
            src += (size_t) count * key;
        }
    }
}

}  // namespace strata::poly
