// src/poly/kolibri1.cpp - the layers of Kolibri 1.  The reference for the math is llama.cpp's kolibri1 graph
// (src/models/kolibri1.cpp).
#include "kolibri1.hpp"

#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <stdexcept>

namespace strata::poly {
namespace {

constexpr size_t kScoreFloats = size_t(16) << 20;  // the attention's scores of the tokens it takes at a time
// A q8 prompt batch's product reads a block of keys at a time; before each block its K and V rows are dequantized
// into an fp16 workspace of this many rows.  Bounded, so it does not grow with the context: the natural block is
// about 16/3 * head_dim rows, well under this, so real prompts take the same products the fp16 path takes.
constexpr size_t kQ8WorkRows = 2048;
// llama.cpp's LLM_EXPERT_GATING_FUNC_TYPE_SIGMOID_LOGIT_ADD as the GGUF counts it: the gates are the sigmoid of the
// scores, the pick is on the scores and the selection bias, and the weights are the gates alone.
constexpr int64_t kGatingSigmoidLogitAdd = 5;
// route_rows()'s gate for it, its own count of the gates.
constexpr int kGateSigmoidLogitAdd = 3;

// A value a file gives once for the model or once a layer.
int64_t of_layer(const MetaValue& v, int il) {
    if (v.type != MetaType::ARRAY) return (int64_t) v.u;
    if ((size_t) il >= v.items.size()) throw std::runtime_error("a per-layer value of the GGUF ends before its layers");
    return (int64_t) v.items[(size_t) il].u;
}

}  // namespace

void Kolibri1::begin_pass() {
    blocks_.assign(layers_.size(), Block{});
    n_full_ = 0;
}

void Kolibri1::load_geometry() {
    load_shape();
    auto find = [&](const char* key) { return meta_in(*gg_, key); };
    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    auto must = [&](const char* key) -> const MetaValue& {
        const MetaValue* v = find(key);
        need(v != nullptr, key);
        return *v;
    };

    // every layer is a mixture of experts with a shared expert beside them, gated by a sigmoid
    gelu_ = false;
    need(n_moe_ == 0 && n_expert_ > 0 && n_ff_exp_ > 0 && n_ff_shexp_ > 0, "no routed experts");
    const MetaValue* gating = find("expert_gating_func");
    need(gating && (int64_t) gating->u == kGatingSigmoidLogitAdd, "expert_gating_func is not 5 (sigmoid_logit_add)");
    const MetaValue* shared = find("expert_shared_count");
    need(!shared || (int) shared->u == 1, "expert_shared_count is not 1");
    const MetaValue* wnorm = find("expert_weights_norm");
    need(!wnorm || wnorm->u == 0, "expert_weights_norm is on, which this engine does not run");
    // the gates are not made to sum to one
    clamp_exp_.assign((size_t) n_layer_ + 1, INFINITY);
    clamp_shexp_ = clamp_exp_;

    // four window layers in five, the fifth of every five the whole sequence so far
    window_ = shape_i("attention.sliding_window");
    n_head_ = shape_i("attention.head_count");
    n_kv_ = shape_i("attention.head_count_kv");
    head_dim_ = shape_i("attention.key_length");
    need(window_ > 0, "bad attention.sliding_window");
    need(n_head_ > 0 && n_head_ <= 256, "bad attention.head_count");
    need(n_kv_ > 0 && n_head_ % n_kv_ == 0, "bad attention.head_count_kv");
    need(head_dim_ > 0 && head_dim_ % 2 == 0 && head_dim_ <= 256, "bad attention.key_length");
    need(shape_i("attention.value_length") == head_dim_, "the attention's keys and values differ in length");
    const MetaValue& pattern = must("attention.sliding_window_pattern");
    swa_.resize((size_t) n_layer_);
    for (int il = 0; il < n_layer_; ++il) swa_[(size_t) il] = of_layer(pattern, il) != 0;
    need(swa_.front(), "the model does not begin with a window layer");
    need(!q8() || (n_kv_ * head_dim_) % 32 == 0, "--kv q8_0 needs n_kv * head_dim to be a multiple of 32");
    if (q8()) {
        const size_t row_bytes = (size_t) n_kv_ * head_dim_ + (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
        std::fprintf(stderr, "strata-poly: q8_0 KV cache: %zu bytes a row\n", row_bytes);
    }
    theta_scale_ = std::pow(shape_f("rope.freq_base"), -2.f / (float) head_dim_);
}

void Kolibri1::load_layer(int il) {
    Layer& L = layers_[(size_t) il];
    Block& B = blocks_[(size_t) il];
    auto f32 = [&](const char* n, int64_t a) { return upload(tensor(il, n), blk(il, n), a, 1, kF32); };
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };

    L.fixed = true;   // the attention's launches read the position on the card when they are recorded
    // only the window layers turn their queries and keys: the full layers have no position at all (RNoPE)
    B.swa = swa_[(size_t) il] != 0;
    B.n_rot = B.swa ? head_dim_ : 0;
    if (!B.swa) ++n_full_;   // the layers whose rows the card's rows hold
    L.attn_norm = f32("attn_norm.weight", n_embd_);
    B.wq = mat("attn_q.weight", n_embd_, (int64_t) n_head_ * head_dim_);
    B.wk = mat("attn_k.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
    B.wv = mat("attn_v.weight", n_embd_, (int64_t) n_kv_ * head_dim_);
    B.wo = mat("attn_output.weight", (int64_t) n_head_ * head_dim_, n_embd_);
    B.q_norm = f32("attn_q_norm.weight", head_dim_);
    B.k_norm = f32("attn_k_norm.weight", head_dim_);
    // the sandwich norm: the attention's own output, before it goes into the residual
    B.post_attn = f32("post_attention_norm.weight", n_embd_);
    L.ffn_norm = f32("ffn_norm.weight", n_embd_);
    B.post_ffn = f32("post_ffw_norm.weight", n_embd_);

    load_moe(il, true);   // the shared expert beside the routed ones

    const TensorInfo* b = tensor(il, "exp_probs_b.bias");
    if (!b || b->type != kTypeF32 || (int64_t) b->elements() != n_expert_)
        throw std::runtime_error(blk(il, "exp_probs_b.bias") + " is " + (b ? shape_of(*b) : "missing") +
                                 ", expected F32 [" + std::to_string(n_expert_) + "]");
    home(*b).payload(*b);
    B.probs_b.assign((size_t) n_expert_, 0.f);
    std::memcpy(B.probs_b.data(), home(*b).data(*b), (size_t) n_expert_ * sizeof(float));
    B.probs_b_dev = upload(b, blk(il, "exp_probs_b.bias"), n_expert_, 1, kF32);
}

// The caches: a full layer keeps a row a position, a window layer a ring of the rows a call can read.  Under
// --kv q8_0 a row is its codes and, in a second array, an fp16 scale a 32-value block (store_q8); the rings, one
// block for the whole model as the fp16 ones are, take two arrays for that.
void Kolibri1::alloc_caches() {
    ring_ = std::min(opt_.max_context, window_ + rows_);
    keep_ = std::min(window_, ring_);
    const size_t row = (size_t) n_kv_ * head_dim_;                               // codes, or fp16 values
    const size_t scales = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);  // a q8_0 row's scales
    if (q8()) {
        size_t codes = 0, all_scales = 0;
        for (const Block& B : blocks_)
            if (B.swa) {
                codes += 2 * (size_t) ring_ * row;
                all_scales += 2 * (size_t) ring_ * scales;
            }
        // A window layer's rows are a ring, whatever the sequence's length, and the ring is the sequence's own
        // (seq_buf() reaches it): seq_alloc() gives each sequence its own and records it for the others.
        uint8_t* rc = codes ? (uint8_t*) seq_alloc(codes, "the attention cache") : nullptr;
        uint8_t* rs = all_scales ? (uint8_t*) seq_alloc(all_scales, "the attention cache") : nullptr;
        for (int il = 0; il < (int) blocks_.size(); ++il) {
            Block& B = blocks_[(size_t) il];
            if (!runs(il)) continue;
            if (B.swa) {
                B.k_codes = (int8_t*) rc;
                rc += (size_t) ring_ * row;
                B.k_scales = (uint16_t*) rs;
                rs += (size_t) ring_ * scales;
                B.v_codes = (int8_t*) rc;
                rc += (size_t) ring_ * row;
                B.v_scales = (uint16_t*) rs;
                rs += (size_t) ring_ * scales;
            } else {
                B.k_codes = (int8_t*) kv_alloc(row, 1);
                B.k_scales = (uint16_t*) kv_alloc(scales, 1);
                B.v_codes = (int8_t*) kv_alloc(row, 1);
                B.v_scales = (uint16_t*) kv_alloc(scales, 1);
            }
        }
        chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
        return;
    }
    size_t rings = 0;   // a window layer's rows are a ring, whatever the sequence's length
    for (const Block& B : blocks_)
        if (B.swa) rings += 2 * (size_t) ring_ * n_kv_ * head_dim_;
    uint16_t* kv = rings ? (uint16_t*) seq_alloc(rings * sizeof(uint16_t), "the attention cache") : nullptr;
    for (int il = 0; il < (int) blocks_.size(); ++il) {
        Block& B = blocks_[(size_t) il];
        if (!runs(il)) continue;
        if (B.swa) {
            B.keys = kv;
            kv += ring_ * row;
            B.vals = kv;
            kv += ring_ * row;
        } else {
            B.keys = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
            B.vals = (uint16_t*) kv_alloc(row * sizeof(uint16_t), 1);
        }
    }
    chunk_ = (int) std::clamp<size_t>(kScoreFloats / ((size_t) n_head_ * opt_.max_context), 1, (size_t) rows_);
}

// Per-token buffers hold rows_ tokens, row after row.
void Kolibri1::assign_scratch() {
    const size_t n = (size_t) rows_, heads = n * widest(), kv = n * n_kv_ * head_dim_;
    qraw_ = fbuf(heads);
    q_ = fbuf(heads);
    kraw_ = fbuf(kv);
    k_ = fbuf(kv);
    v_ = fbuf(kv);
    att_ = fbuf(heads);
    score_ = fbuf((size_t) chunk_ * n_head_ * opt_.max_context);
    // STRATA_KV_INT8=1 reads the q8_0 cache with attend_int8's dp4a instead of the float q8 readers and the
    // dequantized prompt batch.  Opt-in: it changes the arithmetic (the queries are quantized too).
    if (q8() && !q8c_) {
        const char* e = std::getenv("STRATA_KV_INT8");
        if (e && std::atoi(e) != 0) {
            q8c_ = (int8_t*) dmalloc((size_t) rows_ * n_head_ * head_dim_, "the queries as codes");
            q8d_ = (uint16_t*) dmalloc((size_t) rows_ * n_head_ * (head_dim_ / kQ8_0Values) * sizeof(uint16_t),
                                       "the queries' deltas");
        }
    }
    if (rows_ >= kStageRows && !q8c_) awork_ = fbuf(heads + heads / 2 + 1 + 3 * n * n_head_);  // prompt attention
    if (rows_ >= kStageRows && q8() && !q8c_) {
        // the keys and values a q8 prompt batch dequantizes, side by side; work_halves_ halves each
        work_halves_ = kQ8WorkRows * (size_t) n_kv_ * head_dim_;
        q8w_ = fbuf((2 * work_halves_ + 1) / 2);
    }
}

int Kolibri1::widest() const { return std::max({n_head_ * head_dim_, n_ff_exp_, n_ff_shexp_, 256}); }

// The recordings of a full layer's head are numbered by the rows they are sized for; a window layer's never reads
// more than its window.
int Kolibri1::head_key(int il, int n) const { return blocks_[(size_t) il].swa ? 0 : room_key(n); }

// A sequence's rows are the full layers': a window layer's are part of its state (save_layer()).  Under --kv q8_0
// a row is its codes and, in a second array, a scale every 32 of them; the two are saved one after the other, as
// the layer's other side is.
size_t Kolibri1::card_row_bytes() const {
    if (q8()) {
        const size_t codes = (size_t) n_kv_ * head_dim_,
                     scales = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
        return (size_t) n_full_ * 2 * (codes + scales);
    }
    return (size_t) n_full_ * 2 * n_kv_ * head_dim_ * sizeof(uint16_t);
}

void Kolibri1::save_card_rows(int64_t first, int64_t n, uint8_t* dst) {
    if (first < 0 || n < 0 || first + n > opt_.max_context)
        throw std::logic_error("save_rows: the rows are outside the context");
    ck(cudaStreamSynchronize(cs(st_)), "save rows");
    if (q8()) {
        // q8_0: a layer writes its keys' codes and scales, then the values', so that a load can walk the same runs
        // back.  card_row_bytes() sums to n_full_ * 2 * (codes + scales) a row.
        const size_t crow = (size_t) n_kv_ * head_dim_,
                     srow = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
        const size_t at_c = (size_t) first * crow, at_s = (size_t) first * srow;
        const size_t n_c = (size_t) n * crow, n_s = (size_t) n * srow;
        for (const Block& B : blocks_) {
            if (B.swa || !B.k_codes || n == 0) continue;   // a window layer's rows are its state, another's card's
            ck(cudaMemcpy(dst, (const uint8_t*) B.k_codes + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
            ck(cudaMemcpy(dst, (const uint8_t*) B.k_scales + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
            ck(cudaMemcpy(dst, (const uint8_t*) B.v_codes + at_c, n_c, cudaMemcpyDeviceToHost), "save rows");
            dst += n_c;
            ck(cudaMemcpy(dst, (const uint8_t*) B.v_scales + at_s, n_s, cudaMemcpyDeviceToHost), "save rows");
            dst += n_s;
        }
        return;
    }
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), bytes = (size_t) n * row;
    for (const Block& B : blocks_) {
        if (B.swa || !B.keys || bytes == 0) continue;   // a window layer's rows are its state, another's card's
        for (const uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy(dst, (const uint8_t*) cache + (size_t) first * row, bytes, cudaMemcpyDeviceToHost),
               "save rows");
            dst += bytes;
        }
    }
}

void Kolibri1::load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    if (from < 0 || first < from || n < 0 || first + n > from + count || first + n > opt_.max_context)
        throw std::logic_error("load_rows: the rows are outside what was saved");
    cover(first + n);
    ck(cudaStreamSynchronize(cs(st_)), "load rows");
    if (q8()) {
        // q8_0: mirror save_card_rows run for run, advancing by each run's own saved size (count * row) so
        // multi-layer blobs line up; the rows wanted start `first - from` rows into the run.
        const size_t crow = (size_t) n_kv_ * head_dim_,
                     srow = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
        const size_t at_c = (size_t) first * crow, at_s = (size_t) first * srow;
        const size_t n_c = (size_t) n * crow, n_s = (size_t) n * srow;
        const size_t sk_c = (size_t) (first - from) * crow, sk_s = (size_t) (first - from) * srow;
        const size_t all_c = (size_t) count * crow, all_s = (size_t) count * srow;
        for (const Block& B : blocks_) {
            if (B.swa || !B.k_codes || n == 0) continue;   // as save_card_rows() skips them
            ck(cudaMemcpy((uint8_t*) B.k_codes + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
            ck(cudaMemcpy((uint8_t*) B.k_scales + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
            ck(cudaMemcpy((uint8_t*) B.v_codes + at_c, src + sk_c, n_c, cudaMemcpyHostToDevice), "load rows");
            src += all_c;
            ck(cudaMemcpy((uint8_t*) B.v_scales + at_s, src + sk_s, n_s, cudaMemcpyHostToDevice), "load rows");
            src += all_s;
        }
        return;
    }
    const size_t row = (size_t) n_kv_ * head_dim_ * sizeof(uint16_t), bytes = (size_t) n * row;
    for (const Block& B : blocks_) {
        if (B.swa || !B.keys || bytes == 0) continue;   // as save_card_rows() skips them
        for (uint16_t* cache : {B.keys, B.vals}) {
            ck(cudaMemcpy((uint8_t*) cache + (size_t) first * row, src + (size_t) (first - from) * row, bytes,
                          cudaMemcpyHostToDevice),
               "load rows");
            src += (size_t) count * row;
        }
    }
}

// The last keep_ positions before `pos` as runs of a ring's rows: f(the run's place among the keep_, its first ring
// row, its rows).  A position before the sequence's start has no run.
template <class F>
void Kolibri1::window_runs(int64_t pos, F&& f) const {
    for (int64_t p = std::max<int64_t>(0, pos - keep_); p < pos;) {
        const int64_t at = p % ring_, m = std::min<int64_t>(pos - p, ring_ - at);
        f((size_t) (p - (pos - keep_)), (size_t) at, (size_t) m);
        p += m;
    }
}

// What a window layer has of the sequence: the rows the next token attends.  Under --kv q8_0 a row is its codes
// plus a scale every 32 of them, in two arrays, so one layer's part is the codes and the scales one after the other
// on each side.
size_t Kolibri1::extra_bytes() const {
    const size_t codes = (size_t) n_kv_ * head_dim_,
                 scales = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
    size_t b = 0;
    for (const Block& B : blocks_)
        if (B.swa) b += 2 * (size_t) keep_ * (q8() ? codes + scales : codes * sizeof(uint16_t));
    return b;
}

// A window layer's rows, which are no recurrent state here, so straight in, after the window layers before it.
void Kolibri1::save_layer(int il, int64_t pos, uint8_t* dst) {
    if (!blocks_[(size_t) il].swa) return;
    const size_t codes = (size_t) n_kv_ * head_dim_,
                 scales = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
    const size_t part = 2 * (size_t) keep_ * (q8() ? codes + scales : codes * sizeof(uint16_t));
    for (int j = 0; j < il; ++j)
        if (blocks_[(size_t) j].swa) dst += part;
    const Block& B = blocks_[(size_t) il];
    const auto save_ring = [&](const uint8_t* cache, size_t row) {
        std::memset(dst, 0, (size_t) keep_ * row);   // the places of positions before the sequence's start
        window_runs(pos, [&](size_t slot, size_t at, size_t m) {
            ck(cudaMemcpy(dst + slot * row, cache + at * row, m * row, cudaMemcpyDeviceToHost), "save state");
        });
        dst += (size_t) keep_ * row;
    };
    if (q8()) {
        save_ring((const uint8_t*) B.k_codes, codes);
        save_ring((const uint8_t*) B.k_scales, scales);
        save_ring((const uint8_t*) B.v_codes, codes);
        save_ring((const uint8_t*) B.v_scales, scales);
        return;
    }
    save_ring((const uint8_t*) B.keys, codes * sizeof(uint16_t));
    save_ring((const uint8_t*) B.vals, codes * sizeof(uint16_t));
}

void Kolibri1::load_extra(const uint8_t* src, int64_t pos) {
    const size_t codes = (size_t) n_kv_ * head_dim_,
                 scales = (size_t) n_kv_ * (head_dim_ / 32) * sizeof(uint16_t);
    const auto load_ring = [&](uint8_t* cache, size_t row) {
        window_runs(pos, [&](size_t slot, size_t at, size_t m) {
            ck(cudaMemcpy(cache + at * row, src + slot * row, m * row, cudaMemcpyHostToDevice), "load state");
        });
        src += (size_t) keep_ * row;
    };
    for (const Block& B : blocks_) {
        if (!B.swa) continue;
        if (q8()) {
            load_ring((uint8_t*) B.k_codes, codes);
            load_ring((uint8_t*) B.k_scales, scales);
            load_ring((uint8_t*) B.v_codes, codes);
            load_ring((uint8_t*) B.v_scales, scales);
            continue;
        }
        load_ring((uint8_t*) B.keys, codes * sizeof(uint16_t));
        load_ring((uint8_t*) B.vals, codes * sizeof(uint16_t));
    }
}

// n tokens' rows from position `pos` into a layer's cache.  A ring's rows may end and go on from its start.  A
// recorded launch stores them from the position at_ holds when it runs.
void Kolibri1::store(const float* x, uint16_t* cache, const Block& B, int64_t pos, int n) {
    const int width = n_kv_ * head_dim_;
    if (recording_) return store_f16_at(x, cache, width, at_, n, B.swa ? ring_ : 0, st_);
    if (!B.swa) return store_f16(x, cache, width, pos, n, st_);
    const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
    store_f16(x, cache, width, at, head, st_);
    if (head < n) store_f16(x + (size_t) head * width, cache, width, 0, n - head, st_);
}

// The same for a --kv q8_0 cache: the tokens' rows as codes and scales (store_q8).  A ring's rows may end and go
// on from its start, and a recorded launch stores them from the position at_ holds when it runs.
void Kolibri1::store_q8(const float* x, int8_t* codes, uint16_t* scales, const Block& B, int64_t pos, int n) {
    const int width = n_kv_ * head_dim_;
    if (recording_) return store_q8_at(x, codes, scales, width, at_, n, B.swa ? ring_ : 0, st_);
    if (!B.swa) return strata::poly::store_q8(x, codes, scales, width, pos, n, st_);
    const int at = (int) (pos % ring_), head = std::min(n, ring_ - at);
    strata::poly::store_q8(x, codes, scales, width, at, head, st_);
    if (head < n) strata::poly::store_q8(x + (size_t) head * width, codes, scales, width, 0, n - head, st_);
}

// Queries and keys are normed a head and, in a window layer, turned by the position; the values are normed
// without weights and go as they come.
void Kolibri1::attn(Block& B, int64_t pos, int n) {
    if (rows_of_) return attn_rows(B, n);   // step(): every row in its own sequence
    const float scale = 1.f / std::sqrt((float) head_dim_);   // llama.cpp's kq_scale
    const long long* at = recording_ ? at_ : nullptr;   // a recorded launch: the tokens from what at_ holds
    const long long* rot = at ? at + 1 : nullptr;   // and the turn each of them is given, read there too
    lin(B.wq, xn_, qraw_, n, xq_);
    head_norm_rope(qraw_, head_dim_, fp(B.q_norm), q_, head_dim_, n_head_, B.n_rot, theta_scale_, pos, n, rms_eps_,
                   st_, nullptr, nullptr, 1, rot);
    lin(B.wk, xn_, kraw_, n, xq_);
    head_norm_rope(kraw_, head_dim_, fp(B.k_norm), k_, head_dim_, n_kv_, B.n_rot, theta_scale_, pos, n, rms_eps_, st_,
                   nullptr, nullptr, 1, rot);
    lin(B.wv, xn_, v_, n, xq_);   // the values are not normed: the full layers have no position, and their queries
                                   // and keys have no turn there either (n_rot is 0)
    const bool is_q8 = q8();
    if (is_q8) {
        store_q8(k_, B.k_codes, B.k_scales, B, pos, n);
        store_q8(v_, B.v_codes, B.v_scales, B, pos, n);
    } else {
        store(k_, B.keys, B, pos, n);
        store(v_, B.vals, B, pos, n);
    }
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    const Kvq8 kq = is_q8 ? Kvq8{B.k_codes, B.k_scales} : Kvq8{},
                vq = is_q8 ? Kvq8{B.v_codes, B.v_scales} : Kvq8{};
    // a prompt batch through cuBLAS; a few tokens with the kernels that give a token the same bits in any company
    if (is_q8 && q8c_) {
        attend_int8(q_, KvInt8{q8c_, q8d_}, KvInt8{B.k_codes, B.k_scales}, KvInt8{B.v_codes, B.v_scales}, pos, n,
                    score_, chunk_, att_, head_dim_, n_head_, n_kv_, scale, st_, window, ring, false, at,
                    !at ? 0 : B.swa ? std::min(window_ + n - 1, opt_.max_context) : room(n));
    } else if (n >= kStageRows) {
        // the fp16 path reads the caches where they lie; under q8 each block of K and V the products read is
        // dequantized into q8w_ (kwork then vwork)
        uint16_t* kw = is_q8 ? (uint16_t*) q8w_ : nullptr;
        uint16_t* vw = is_q8 ? kw + work_halves_ : nullptr;
        attend_batch(blas_, q_, is_q8 ? nullptr : B.keys, is_q8 ? nullptr : B.vals, pos, n, score_,
                     (size_t) chunk_ * n_head_ * opt_.max_context, awork_, att_, head_dim_, n_head_, n_kv_, scale, st_,
                     window, ring, false, kq, vq, kw, vw, is_q8 ? work_halves_ : 0);
    } else {
        attend(q_, is_q8 ? nullptr : B.keys, is_q8 ? nullptr : B.vals, pos, n, score_, chunk_, att_, head_dim_, n_head_,
               n_kv_, scale, st_, window, ring, false, at,
               !at ? 0 : B.swa ? std::min(window_ + n - 1, opt_.max_context) : room(n), /*score_floats*/ 0, kq, vq);
    }
    lin(B.wo, att_, ao_, n);
}

// step()'s rows, each of its own sequence and past its pictures: the products for all of them at once, and a row at
// a time the turn (it takes one position and counts on from it), the stored keys and values and the attention, in
// the row's sequence.
void Kolibri1::attn_rows(Block& B, int n) {
    const float scale = 1.f / std::sqrt((float) head_dim_);
    lin(B.wq, xn_, qraw_, n, xq_);
    lin(B.wk, xn_, kraw_, n, xq_);
    lin(B.wv, xn_, v_, n, xq_);
    const size_t hq = (size_t) n_head_ * head_dim_, kv = (size_t) n_kv_ * head_dim_;
    const int window = B.swa ? window_ : 0, ring = B.swa ? ring_ : 0;
    const bool is_q8 = q8();
    for (int i = 0; i < n; ++i) {
        const Row& r = rows_of_[i];
        const long long* ai = at_ + 2 * i;
        head_norm_rope(qraw_ + i * hq, head_dim_, fp(B.q_norm), q_ + i * hq, head_dim_, n_head_, B.n_rot, theta_scale_, 0,
                       1, rms_eps_, st_, nullptr, nullptr, 1, ai + 1);
        head_norm_rope(kraw_ + i * kv, head_dim_, fp(B.k_norm), k_ + i * kv, head_dim_, n_kv_, B.n_rot, theta_scale_, 0,
                       1, rms_eps_, st_, nullptr, nullptr, 1, ai + 1);
        Kvq8 kq{}, vq{};
        uint16_t* kc = nullptr;
        uint16_t* vc = nullptr;
        if (is_q8) {
            kq = {(int8_t*) seq_buf(r.seq, B.k_codes), (uint16_t*) seq_buf(r.seq, B.k_scales)};
            vq = {(int8_t*) seq_buf(r.seq, B.v_codes), (uint16_t*) seq_buf(r.seq, B.v_scales)};
            store_q8_at(k_ + i * kv, (int8_t*) kq.codes, (uint16_t*) kq.scales, (int) kv, ai, 1, ring, st_);
            store_q8_at(v_ + i * kv, (int8_t*) vq.codes, (uint16_t*) vq.scales, (int) kv, ai, 1, ring, st_);
        } else {
            kc = (uint16_t*) seq_buf(r.seq, B.keys);
            vc = (uint16_t*) seq_buf(r.seq, B.vals);
            store_f16_at(k_ + i * kv, kc, (int) kv, ai, 1, ring, st_);
            store_f16_at(v_ + i * kv, vc, (int) kv, ai, 1, ring, st_);
        }
        const int64_t keep = pos_;
        pos_ = r.pos;   // room(1) sizes the launch for the row's own place in its sequence
        const int sized = B.swa ? std::min(window_, opt_.max_context) : room(1);
        pos_ = keep;
        if (is_q8 && q8c_)
            attend_int8(q_ + i * hq, KvInt8{q8c_, q8d_}, KvInt8{kq.codes, kq.scales}, KvInt8{vq.codes, vq.scales},
                        r.pos, 1, score_, chunk_, att_ + i * hq, head_dim_, n_head_, n_kv_, scale, st_, window, ring,
                        false, ai, sized);
        else
            attend(q_ + i * hq, kc, vc, r.pos, 1, score_, chunk_, att_ + i * hq, head_dim_, n_head_, n_kv_, scale, st_,
                   window, ring, false, ai, sized, 0, kq, vq);
    }
    lin(B.wo, att_, ao_, n);
}

// A layer from its input to its router's copy.  A norm and its quantization are one launch, and so are the
// attention's sum into the residual and the next norm.  A prompt batch's MoE (moe_rows) does its own routing.
void Kolibri1::layer_head(Layer& L, int il, int n) {
    const int threads = sum_threads(n_embd_);
    kernels::native_add_rms_norm_q8_1(r_, nullptr, fp(L.attn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("attn_in", il, xn_, n_embd_);
    attn(blocks_[(size_t) il], pos_, n);
    rms_norm(ao_, fp(blocks_[(size_t) il].post_attn), cur_, n_embd_, n, rms_eps_, st_);
    dbg("attn", il, cur_, n_embd_);
    // the sandwich norm the attention's output has joined the residual under
    kernels::native_add_rms_norm_q8_1(r_, cur_, fp(L.ffn_norm), xn_, xq_, n_embd_, n, rms_eps_, threads, st_);
    dbg("ffn_in", il, xn_, n_embd_);
    if (!batched(L, n)) moe_router(L, n);
}

// The shared expert's output and the routed experts' sum, normed, go into the residual.
void Kolibri1::layer_tail(Layer&, int il, int n) {
    rms_norm(fo_, fp(blocks_[(size_t) il].post_ffn), cur_, n_embd_, n, rms_eps_, st_);
    axpy(cur_, 1.f, r_, n * n_embd_, true, st_);
}

void Kolibri1::output_rows(int first, int rows, float* h) {
    rms_norm(r_ + (size_t) first * n_embd_, fp(out_norm_), h, n_embd_, rows, rms_eps_, st_);
}

// The shared expert: ungated, beside the routed ones, and its output is where their sum starts.
void Kolibri1::moe_shared(const Layer& L, int il, int n) {
    lin(L.sh_gate, xn_, g_, n, xq_);
    lin(L.sh_up, xn_, u_, n, xq_);
    swiglu_clamp(g_, u_, h_, n * n_ff_shexp_, clamp_shexp_[(size_t) il], st_);
    lin(L.sh_down, h_, fo_, n);
}

// What ggml's MoE graph does for this model: the n_used largest of the scores and the selection bias are picked,
// and a picked expert's weight is the sigmoid of its own score, which the weights of a token are not made to sum
// to one.  One pass keeps the picks in order as they come: a score is read once, and the sigmoids are six, where
// picking each of them out of all is six passes and a sigmoid for every expert.  A tie goes to the lower expert,
// as a scan from the lowest does.
void Kolibri1::route(int il, const float* logits, int32_t, int* ids, float* w, float* tmp) const {
    const std::vector<float>& bias = blocks_[(size_t) il].probs_b;
    float* pick = tmp + n_expert_;
    for (int e = 0; e < n_expert_; ++e) pick[e] = logits[e] + bias[(size_t) e];   // the pick is on the bare score, not on the gate
    int have = 0;
    for (int e = 0; e < n_expert_; ++e) {
        if (have == n_used_ && !(pick[e] > w[n_used_ - 1])) continue;
        int j = have;
        while (j > 0 && w[j - 1] < pick[e]) --j;   // after the ones it does not beat, an equal one included
        for (int t = have < n_used_ ? have : n_used_ - 1; t > j; --t) {
            w[t] = w[t - 1];
            ids[t] = ids[t - 1];
        }
        w[j] = pick[e];
        ids[j] = e;
        if (have < n_used_) ++have;
    }
    for (int k = 0; k < n_used_; ++k) w[k] = 1.f / (1.f + std::exp(-logits[ids[k]]));
}

bool Kolibri1::route_card(int il, int n, int32_t* ids, float* w) {
    const Block& B = blocks_[(size_t) il];
    route_rows(rl_, n_expert_, n_used_, n, kGateSigmoidLogitAdd, 1.f, fp(B.probs_b_dev), nullptr,
               /*norm*/ false, /*w_scale*/ 0.f, ids, w, st_);
    return true;
}

}  // namespace strata::poly
