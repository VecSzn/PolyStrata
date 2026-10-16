// include/strata/poly/kernels.hpp - the float kernels of the forward passes.
//
// Every call enqueues on the given CUDA stream and returns; nothing here but blas_create allocates or waits.
// Pointers are device pointers.  The math follows llama.cpp's graphs (src/models/glm5-next.cpp, qwen35moe.cpp and
// gemma4.cpp) and the ggml ops they build, which are the reference the engine is checked against.
//
// A call works on `rows` tokens at consecutive positions.  A buffer of per-token values holds them row after row,
// each row as long as the one-token case describes.
#pragma once

#include <cstddef>
#include <cstdint>

namespace strata::poly {

/// y[o] = sum_i w[o * n_in + i] * x[(o / rows_per_x) * n_in + i], o in [0, n_out).  With rows_per_x == n_out it is
/// a plain matrix-vector product; a smaller value gives each group of rows its own input (the per-head MLA
/// projections), so a token's x is n_in * n_out / rows_per_x long.
void gemv_f32(const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows, void* stream);
void gemv_bf16(const uint16_t* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows,
               void* stream);
/// The same for the many rows of a prompt batch, as a cuBLAS product on the stream of `blas` (blas_create), one
/// for each group of rows_per_x outputs.  It sums in another order than gemv_f32, so a row does not come out
/// to the last bit as it does there.  Throws std::runtime_error.
void gemm_f32(void* blas, const float* w, const float* x, float* y, int n_in, int n_out, int rows_per_x, int rows);
/// n BF16 values as floats, for gemm_f32.
void bf16_to_f32(const uint16_t* w, float* out, size_t n, void* stream);

/// RMS norm over each of `rows` groups of n values: y = x / sqrt(mean(x^2) + eps) * w.  `w` (n values, shared by the
/// rows) may be null.  A row of y starts y_stride values after the one before it (0: n), so two calls can fill
/// the two halves of rows twice as long.
void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* stream, int y_stride = 0);
/// The threads of a block that sums over n values (rms_norm()'s among them).
int sum_threads(int n);
/// y = (x - mean) / sqrt(var + eps) * w + b.
void layer_norm(const float* x, const float* w, const float* b, float* y, int n, int rows, float eps, void* stream);

/// The hyper-connection read.  r is a token's hc * n_embd residual streams.  Writes cur (n_embd) and hcw: post[hc]
/// then comb[hc * hc] (comb[dst + hc * src]), which hc_post reads.  `mix` is hc*(2+hc)+1 floats of scratch a token.
void hc_pre(const float* r, const float* fn, const float* scale, const float* base, float* mix, float* hcw,
            float* cur, int n_embd, int rows, float norm_eps, float hc_eps, int sinkhorn_iters, void* stream);
/// The hyper-connection write, in place on r: r[dst] = x * post[dst] + sum_src r[src] * comb[dst, src].
void hc_post(const float* x, const float* hcw, float* r, int n_embd, int rows, void* stream);
/// y = mean over the hc streams, token by token.
void hc_mean(const float* r, float* y, int n_embd, int rows, void* stream);
/// DeepSeek V4's output read of the streams: y = sum_h r[h] * (sigmoid(fn[h] . r * rs * scale + base[h]) + hc_eps),
/// rs the flat RMS norm's scale.  fn: hc rows of hc * n_embd.  mix: hc + 1 floats a token of scratch.
void hc_head(const float* r, const float* fn, const float* scale, const float* base, float* mix, float* y,
             int n_embd, int rows, float norm_eps, float hc_eps, void* stream);

/// The three causal convolutions of a KDA layer and their SiLU.  proj: the q projections of every row, then the k,
/// then the v (3 * rows * d_inner); out has the same layout.  state: 3 * d_inner * (d_conv - 1) past inputs, oldest
/// first per channel, advanced here by `rows` tokens.  w*: (d_conv, d_inner).
/// With snap, the state as it is after each of the first n_snap tokens is also written there: the one after
/// token i starts i * stride floats in.  That is what a window of guessed tokens goes back to (kda_step too).
void kda_conv(const float* proj, float* state, const float* wq, const float* wk, const float* wv, float* out,
              int d_inner, int rows, void* stream, float* snap = nullptr, size_t stride = 0, int n_snap = 0);
/// The delta-rule step of every head and the gated output, token after token.  qkv: the conv outputs.  g1:
/// f_b(f_a(x)) + dt bias (d_inner); it gates the state's decay per key dimension, as in ggml's gated_delta_net op.
/// a: per head.  beta: per head, before the sigmoid.  s: n_head * head_dim * head_dim, updated.  g2: g_b(g_a(x))
/// (d_inner).  out = rms_norm(o) * o_norm * sigmoid(g2), d_inner values.  A call of 32 rows or more without snap
/// takes three launches for the same bits, and leaves its work in qkv and g1.
void kda_step(float* qkv, float* g1, const float* a, const float* beta, float* s, const float* g2,
              const float* o_norm, float* out, int head_dim, int n_head, int rows, float gate_lower_bound,
              float norm_eps, void* stream, float* snap = nullptr, size_t stride = 0, int n_snap = 0);

/// Stores the tokens' MLA latents (n values each) as fp16 from row `pos` of the layer's cache.
void store_f16(const float* x, uint16_t* cache, int n, int64_t pos, int rows, void* stream);
/// The same from the row the card holds at `at` when the launch runs; with `ring`, a row is its number % ring.
void store_f16_at(const float* x, uint16_t* cache, int n, const long long* at, int rows, int ring, void* stream);
/// Attention over cached latents for the tokens from position `pos`.  q: n_head * rank absorbed queries a token.
/// The first dense_rows tokens attend to every cache row up to their own.  The others each have a list in sel
/// (sel_stride apart): top_rows cache rows, then the rows of the pool their own position is still filling.
/// out: n_head * rank a token, the attention-weighted latents per head.
/// It takes `chunk` tokens at a time: their keys are copied to `keys` as floats (rank a key), and two cuBLAS
/// products per chunk do the rest, with `score` (n_head a key) between them.  Both buffers hold a chunk of tokens
/// with the most keys a token has: top_rows + kpool - 1, or the context if that is less.  Throws
/// std::runtime_error.
void dsa_attend(void* blas, const float* q, const uint16_t* cache, const int32_t* sel, int sel_stride, int top_rows,
                int kpool, int64_t pos, int rows, int dense_rows, float* keys, float* score, int chunk, float* out,
                int rank, int n_head, float scale, void* stream);
/// A cuBLAS handle that enqueues on `stream`, for dsa_attend and gemm_f32.  Throws std::runtime_error.
void* blas_create(void* stream);
void blas_destroy(void* blas);

/// The indexer's cache: a pool's key once its kpool rows are in, dim floats a pool in `pool`, and the latest rows'
/// keys and gates in `ring`, key (dim) | gate (dim) a row, the row of position p at p % ring_rows.  fp32.
/// Stores the tokens' keys and gates from position `pos`.
void idx_store(const float* key, const float* gate, float* ring, int ring_rows, int dim, int64_t pos, int rows,
               void* stream);
/// For each of the positions from `pos` that ends a pool: the pool's key from its kpool rows.  ape: (dim, kpool).
void idx_pool(const float* ring, int ring_rows, float* pool, const float* ape, int dim, int kpool, int64_t pos,
              int rows, void* stream);
/// For the tokens from position `pos`: score[p] = sum_h relu(q[h] . pool[p]) * w[h] for the pools p that are
/// complete at the token's position.  A token's scores start score_stride apart.  w is the raw indexer_proj
/// output; the 1/sqrt(dim * n_head) factor is applied here.  Throws std::invalid_argument unless dim is 128 and
/// n_head 32 or less.
void idx_score(const float* q, const float* w, const float* pool, float* score, int score_stride, int dim,
               int n_head, int kpool, int64_t pos, int rows, void* stream);
/// idx_score() for a prompt batch: the products of the queries and the pools' keys through cuBLAS, as many tokens
/// at a time as have theirs in the work_floats floats of `work`, n_head for each pool of the call's last token.
/// A score has other last bits than idx_score()'s.  Throws std::runtime_error, and std::invalid_argument when
/// work has no room for a token.
void idx_score_batch(void* blas, const float* q, const float* w, const float* pool, float* score, int score_stride,
                     int dim, int n_head, int kpool, int64_t pos, int rows, float* work, size_t work_floats,
                     void* stream);
/// For each of the tokens from position `pos`, from its scores: the n_top pools with the highest, the lower pool
/// first among equals, as their kpool rows each in that order, then the rows that are in no whole pool yet.  A
/// token's list starts sel_stride apart and takes n_top * kpool + kpool - 1 at most.  Every token has n_top
/// whole pools or more.
void idx_select(const float* score, int score_stride, int32_t* sel, int sel_stride, int n_top, int kpool,
                int64_t pos, int rows, void* stream);

/// out = silu(min(gate, limit)) * clamp(up, -limit, limit).
void swiglu_clamp(const float* gate, const float* up, float* out, int n, float limit, void* stream);
/// swiglu_clamp for rows that each hold n gate values and then n up values; out has n values a row.
void swiglu_clamp_rows(const float* gate_up, float* out, int n, int rows, float limit, void* stream);
/// out = gelu(gate) * up, with the tanh form of GELU that ggml's CUDA backend computes.
void geglu(const float* gate, const float* up, float* out, int n, void* stream);
/// geglu for rows that each hold n gate values and then n up values; out has n values a row.
void geglu_rows(const float* gate_up, float* out, int n, int rows, void* stream);
/// The routing of `rows` tokens from their router scores (n_expert a token, times in_scale), as an architecture's
/// route() does it on the host: probabilities by `gate` (0 softmax, 1 sigmoid, 2 sqrt(softplus)), the n_used
/// largest of the probabilities plus `bias` (null: none), the lower expert first among equals; their weights are the
/// probabilities, with `norm` divided by their sum (ggml's floor under it), times escale[expert] (null: none), times
/// w_scale (0: not).  ids and w: n_used a token, on the card.  Throws std::invalid_argument past 512 experts.
void route_rows(const float* logits, int n_expert, int n_used, int rows, int gate, float in_scale, const float* bias,
                const float* escale, bool norm, float w_scale, int32_t* ids, float* w, void* stream);
/// acc[t] += sum over k < n_used of w[t * n_used + k] * x[t * n_used + k], for rows of n values: a token's
/// weighted experts added to its row.
void moe_sum(const float* x, const float* w, float* acc, int n, int n_used, int rows, void* stream);
/// acc = x * a, or acc += x * a when `add`.
void axpy(const float* x, float a, float* acc, int n, bool add, void* stream);
/// y = x for two runs of 4-byte words, n0 and n1 of them, in one launch.  Pinned host memory is the card's to read
/// and write too, and a kernel's copy to or from it has none of the wait that a copy engine's has on Windows,
/// some 25 microseconds between it and the launches beside it.
void copy2(const void* x0, void* y0, int n0, const void* x1, void* y1, int n1, void* stream);
/// Up to 32 vectors of n values, each times its weight into its own sum, one after the other: acc[k] += x[k] * w[k]
/// for k = 0, 1, ...  Two of them may have the same sum.
struct ScaledRows {
    const float* x[32];
    float* acc[32];
    float w[32];
};
void add_scaled(const ScaledRows& rows, int count, int n, void* stream);
/// acc[row] += b for each of the rows of n values.
void add_bias(const float* b, float* acc, int n, int rows, void* stream);

// ---- attention over a key and value cache (qwen35moe, gemma4), and the gated delta net with one decay a head

/// For `rows` tokens of `heads` heads each: a head's RMS norm with the weights w (dim values, the same for every
/// head; null: none), then ggml's rotary embedding on its first n_rot values in the NeoX pairing: value i goes
/// with value i + n_rot / 2, and the pair turns by position * theta_scale^i, divided by freq[i] when freq is
/// given.  A head's input starts x_stride floats after the one before it; y has dim values a head.
/// The position is the token's place in the call counted from `pos`.  With `axes` (three numbers a token) it is
/// the token's position on one of three axes instead: pair i takes axis (i % sections) % 3, ggml's interleaved
/// rope_multi, which is how a picture's rows get a line and a column.
/// With `at`, the first token's position is not `pos` but what the card holds at `at` when the launch runs: a
/// launch recorded once can then be replayed for every position (attend() and store_f16_at() take it too).
void head_norm_rope(const float* x, int x_stride, const float* w, float* y, int dim, int heads, int n_rot,
                    float theta_scale, int64_t pos, int rows, float eps, void* stream, const float* freq = nullptr,
                    const int32_t* axes = nullptr, int sections = 1, const long long* at = nullptr);
/// Causal attention for the tokens from position `pos`: token t attends the rows 0 to pos + t, or with `window`
/// only the last `window` of them.  q: n_head * dim a token.  kcache and vcache: fp16, kv_heads * dim a row; the
/// heads read a row's keys and values in turn, n_head / kv_heads heads each.  With `ring`, the sequence's row p
/// is cache row p % ring.  out: n_head * dim a token.  It takes `chunk` tokens a launch, which read the caches
/// once between them, and score is scratch of chunk * n_head floats for each row a launch's tokens attend
/// between them: pos + rows, or window - 1 + chunk if that is less.  A token comes out the same whatever tokens
/// the call has beside it.  Queries and keys are read four values at a time: dim is a multiple of 4, q lies at
/// a multiple of 16 bytes and kcache of 8, or it throws std::invalid_argument.
/// With `whole` the call's tokens see each other: each attends up to the call's last row, so a token can attend
/// window - 1 + rows of them.
/// With `at`, the tokens start at the position the card holds there; the launches are sized for `room` rows,
/// which must be no fewer than a launch's tokens attend between them, and score has room for them.
/// score_floats: the floats score holds, when more than the above.  Heads of 64 values in groups of 4 over the
/// whole conversation then take a launch block each block of rows, where score has room for those blocks' sums
/// of chunk tokens at the longest the call can attend (attend_split_floats()); the results differ from the other
/// kernels' in the last bits.
void attend(const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows, float* score,
            int chunk, float* out, int dim, int n_head, int kv_heads, float scale, void* stream, int window = 0,
            int ring = 0, bool whole = false, const long long* at = nullptr, int room = 0,
            size_t score_floats = 0);
/// The floats of score that attend()'s blocks of rows take for calls of up to chunk tokens attending up to max_rows
/// rows (score_floats): with that much a model's every call takes them.
size_t attend_split_floats(int chunk, int n_head, int max_rows, int dim);
/// attend() for a prompt batch: the same attention with cuBLAS products on the tensor cores, several times
/// faster on many tokens.  The products read the cache's halves where they lie, take the queries and the
/// softmax's weights rounded to halves, and sum floats.  A token's result depends on the tokens beside it, so
/// the windows of guessed tokens use attend().
/// It takes the call's tokens together and the rows they attend a block at a time: as many rows as have their
/// scores and weights, 6 bytes for each of a token's n_head heads, in the score_floats floats of `score`, which
/// must hold 2 * n_head floats.  work: rows * n_head * (dim + dim / 2 + 3) + 1 floats.  Throws
/// std::runtime_error.
/// Heads of 64 values in groups of 4 over the whole conversation take one kernel instead, which keeps the scores in
/// registers, on NVIDIA cards from Turing on; it uses neither blas, score nor work.
void attend_batch(void* blas, const float* q, const uint16_t* kcache, const uint16_t* vcache, int64_t pos, int rows,
                  float* score, size_t score_floats, float* work, float* out, int dim, int n_head, int kv_heads,
                  float scale, void* stream, int window = 0, int ring = 0, bool whole = false);
/// x = cap * tanh(x / cap).
void softcap(float* x, size_t n, float cap, void* stream);
/// The first of the largest of n logits and the softmax's probability of it, summed without the logits more than
/// 20 below it.
struct Likeliest {
    int32_t token;
    float prob;
};
void likeliest(const float* logits, int n, Likeliest* out, void* stream);
/// acc[r][i] *= sigmoid(g[r][i]) for `rows` groups of n values; group r's gates start r * g_stride floats into g.
/// With one_gate a group has one gate for all its values.
void mul_sigmoid(float* acc, const float* g, int n, int rows, int g_stride, bool one_gate, void* stream);
/// A causal convolution of four taps over each of `channels` channels, and its SiLU.  x and out: `channels` values
/// a token.  state: 3 past inputs a channel, oldest first, advanced here by `rows` tokens.  w: (4, channels).
/// With snap, the state as it is after some of the call's tokens is also written there, for a window of guessed
/// tokens to go back to (gdn_step too): `snaps` has a bit for each token that leaves it, and the states lie in
/// the tokens' order, `stride` floats apart.
void conv4_silu(const float* x, float* state, const float* w, float* out, int channels, int rows, void* stream,
                float* snap = nullptr, size_t stride = 0, unsigned snaps = 0);
/// LFM2's gated short convolution (llama.cpp's shortconv block between its two projections).  bcx: B, C and x of a
/// token, `channels` values each, as the in_proj writes them.  out = C * conv3(B * x), `channels` values a token,
/// a causal convolution of three taps w (3, channels).  state: the 2 B * x before the call a channel, oldest
/// first, advanced here by `rows` tokens.  snap, stride, snaps: as conv4_silu's.
void shortconv(const float* bcx, float* state, const float* w, float* out, int channels, int rows, void* stream,
               float* snap = nullptr, size_t stride = 0, unsigned snaps = 0);
/// The delta-rule step of every value head and its gated output, token after token: ggml's gated_delta_net op
/// with one decay a head.  qkv: a token's conv outputs, the k_heads key heads' q, their k, then the n_head value
/// heads' v, dim values each; value head h reads key head h % k_heads.  alpha and beta: n_head a token, as their
/// projections give them.  dt, a: n_head.  The state decays by exp(softplus(alpha + dt) * a) and takes
/// sigmoid(beta) of the error.  s: n_head * dim * dim, updated.  z: n_head * dim a token.
/// out = rms_norm(o) * o_norm * silu(z), n_head * dim values a token.  qkv, alpha and beta are overwritten.
void gdn_step(float* qkv, float* alpha, float* beta, const float* dt, const float* a, float* s,
              const float* z, const float* o_norm, float* out, int dim, int n_head, int k_heads, int rows,
              float norm_eps, void* stream, float* snap = nullptr, size_t stride = 0, unsigned snaps = 0);

// ---- DeepSeek V4 (deepseek4): one key head that is its value too, a window of raw rows, rows compressed 4 or 128
// to one, and an indexer that picks among the 4-to-1 rows

/// ggml_rope_ext's turn in the normal pairing (value i with i + 1), with YaRN.  corr0 and corr1 are what
/// ggml_rope_yarn_corr_dims() gives; attn_factor is the magnitude the turn starts from.
struct Rope {
    float theta_scale = 1.f;   ///< freq_base^(-2 / n_rot)
    float freq_scale = 1.f, ext_factor = 0.f, attn_factor = 1.f;
    float corr0 = 0.f, corr1 = 0.f;
};
/// For `rows` tokens of `heads` heads of dim values: with norm, each head RMS-normed without weights; then the
/// n_rot values from `off` turned by the position and the others kept.  Token t's position is pos + t * step.
/// back: the turn the other way (ggml_rope_ext_back).  y may be x.  With `at` (step 1), token t's position is what
/// the card holds at at[2 * t] when the launch runs: a launch recorded once is replayed for every position.  The
/// kernels below take `at` the same way.
void dsv4_rope(const float* x, float* y, int dim, int heads, int off, int n_rot, const Rope& r, int64_t pos,
               int step, int rows, bool norm, float eps, bool back, void* stream, const long long* at = nullptr);
/// A compressor's rows into its rings, position p at ring row p % ring: kv as it is, and score plus
/// ape[p % ratio].  width values a row.
void dsv4_state(const float* kv, const float* score, const float* ape, float* ring_kv, float* ring_score, int ring,
                int width, int ratio, int64_t pos, int rows, void* stream, const long long* at = nullptr);
/// For each of the `rows` tokens from position pos that ends a block of ratio positions, the block's compressed row
/// into `cache` (fp16, dim values a block, at the block's index): value d is the sum over the block's rows of
/// softmax(score) * kv, RMS-normed with `norm` (dim weights), its last n_rot values turned by the block's first
/// position.  Block b's rows are the ratio from position b * ratio, with their first dim values.  With overlap a
/// row is 2 * dim values: the block's rows give their second dim, and the ratio rows before the block their first
/// dim (a row before the sequence is left out).  dim is a multiple of 32 up to 1024.
void dsv4_squeeze(const float* ring_kv, const float* ring_score, int ring, int dim, int ratio, bool overlap,
                  const float* norm, float eps, const Rope& r, int n_rot, int64_t pos, int rows, uint16_t* cache,
                  void* stream, const long long* at = nullptr);
/// The indexer's scores of the tokens from position pos: score[b] = scale * sum_h w[h] * relu(q[h] . key[b]) for
/// the compressed rows b < (pos + t + 1) / ratio.  q: n_head * dim a token; w: n_head; keys: fp16, dim a row.  A
/// token's scores start score_stride apart.  dim must be 128.  Throws std::invalid_argument, and
/// std::runtime_error when the card's shared memory has no room for a token's queries.
/// With `at`, the launch is sized for the tokens of positions below `room`.
void dsv4_idx_score(const float* q, const float* w, const uint16_t* keys, float* score, int score_stride, int dim,
                    int n_head, int ratio, int64_t pos, int rows, float scale, void* stream,
                    const long long* at = nullptr, int room = 0);
/// For each token from position pos, its compressed rows to attend, in their order: the n_top with the highest
/// scores, the lower row first among equals, or all of them when it sees n_top or fewer.  sel_stride apart.
void dsv4_select(const float* score, int score_stride, int32_t* sel, int sel_stride, int n_top, int ratio,
                 int64_t pos, int rows, void* stream, const long long* at = nullptr);
/// The attention of the tokens from position pos.  Token t (position p) attends the raw rows of the `window`
/// positions up to p (row p at p % ring of `raw`), then compressed rows of `comp`: with sel, the
/// min((p + 1) / ratio, top) of its list; without, every one below (p + 1) / ratio; with comp null, none.  The
/// keys are fp16, dim values a row, and each key is its value too.  q: n_head * dim a token; sinks: a logit a
/// head that takes part in the softmax with no value.  out: n_head * dim a token.  Throws std::runtime_error when
/// a token's keys do not fit the card's shared memory.  With `at`, the launch is sized for positions below `room`.
void dsv4_attend(const float* q, const uint16_t* raw, int ring, int window, const uint16_t* comp, int ratio,
                 const int32_t* sel, int sel_stride, int top, const float* sinks, float* out, int dim, int n_head,
                 float scale, int64_t pos, int rows, void* stream, const long long* at = nullptr, int room = 0);
/// y[g][t] = x[t][g] for `rows` tokens of `groups` groups of n values; with back, y[t][g] = x[g][t].
void permute_groups(const float* x, float* y, int n, int groups, int rows, bool back, void* stream);

}  // namespace strata::poly
