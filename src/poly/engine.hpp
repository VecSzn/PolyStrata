// src/poly/engine.hpp - the engine that runs a model from its GGUF.
//
// The engine reads the GGUF in place, in one file or in the numbered files of a split one.  Everything but the
// routed experts goes to VRAM; as many routed experts as fit go there too, and the rest are computed on the CPU
// from the mapped file while the GPU does its share.
//
// Engine is what every architecture shares.  An architecture is a class derived from it (glm5next.hpp, qwen35.hpp,
// gemma4.hpp) that reads the model's shape, loads a layer's weights and composes a layer from the kernels;
// Engine::open() picks it by the file's general.architecture.
//
// On several cards (Options::layer_split) each card has an engine of its own for a run of the layers, with their
// weights, caches, state and expert cache, and the last one has the output and the draft block.  The first one is
// the one the caller holds: it runs its layers, hands the rows of the residual on to the next one through host
// memory, and the next one goes on from them.
#pragma once

#include "cpu_experts.hpp"

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <thread>
#include <utility>
#include <vector>

namespace strata {
struct TensorInfo;
struct MetaValue;
}

namespace strata::poly {

class Shards;
struct Likeliest;

/// The attention caches' storing format, opt-in via --kv.  q8_0 is ggml's blocks (one fp16 scale a 32-value
/// block of int8 codes) and reads them back element by element; int8 is the same rule with the codes and the
/// deltas in two arrays, which a reader can take four codes to a load, and int8k keeps only the keys' side
/// quantized.  q8_0 is qwen35moe's; int8 and int8k are Gemma 4's - the two readers are not the same arithmetic,
/// so a model has one or the other rather than a choice between them.
enum class KvFmt { kFp16 = 0, kQ8_0 = 1, kInt8 = 2, kInt8K = 3 };

struct Options {
    std::string model;
    int max_context = 8192;
    int threads = 0;              ///< 0: the CPUs this process may use
    int vram_reserve_mib = -1;    ///< left free after the expert cache is filled; below 0: by the card's size
    std::string profile;          ///< expert use counts; orders the expert cache, and save_profile() adds to it
    bool gpu_experts = true;
    bool dump = false;            ///< print a checksum of every stage, for comparing against llama.cpp
    // A prompt batch.  More tokens a call and a larger staging buffer read a prompt faster, and both take VRAM
    // that the expert cache, and so decoding, does without.
    int prompt_batch = 1024;      ///< prompt tokens one eval() takes; the scratch is about 1.4 MiB of VRAM a token
    int stage_mib = 0;            ///< the buffer the experts that are not cached pass through, never more than a
                                  ///< layer's experts take; 0: 128 where a read can widen it (stage_read_mib) on
                                  ///< CUDA, else 2048, about what a layer of Q2_K experts leaves out of the cache
    int stage_read_mib = -1;      ///< a read of several calls: the staging buffer grows to this, at most what a
                                  ///< layer's experts take, with VRAM the expert cache lends it until the read is
                                  ///< done (end_read()), so that each layer's experts are copied once a read, not
                                  ///< once a call; 0 (or no more than stage_mib): never; below 0: 2048 with the
                                  ///< 128 MiB default of stage_mib, else never
    int stage_ahead = 1024;       ///< from this many tokens, a call routes to most of a layer's experts, and the
                                  ///< ones that are not cached start their copy before the layer has routed; in a
                                  ///< read of several calls and this many tokens every call does; 0: never.  Under
                                  ///< it a call copies only the experts it routes to, which with 8 GB of VRAM was
                                  ///< the faster of the two up to about 1,000 tokens
    float cpu_share = -1;         ///< of the experts outside VRAM that such a call routes to, the share the CPU
                                  ///< computes while the others are copied: the ones the fewest tokens chose;
                                  ///< below 0: as the two took the time between them; 0: none
    int expert_swaps = 64;        ///< experts on their way into the cache at a time while an answer is written, each
                                  ///< to the slot of one the answer uses less; 0: the cache stays as it was loaded
    int64_t ram_experts_mib = -2; ///< the memory this process holds for the experts outside VRAM, answer or not:
                                  ///< the ones used most lately are read into it from the files; the others are left
                                  ///< to the file cache, which gives them to whatever needs the memory, and come back
                                  ///< from it or the disk when a token routes to them.  -1: all of them.  -2 (the
                                  ///< default): 6144 when they pass it and are at most twice that, else all
    bool stay_near = true;        ///< with several NUMA nodes, keep the CPU threads on the one that holds the experts
    bool graphs = true;           ///< a token's fixed launches are recorded once, a CUDA graph a layer, and replayed
    bool kv_grow = true;          ///< the attention caches get VRAM as the sequence grows, where the card can map it
                                  ///< so; false: for the whole context at load, as the expert cache gets the rest
    KvFmt kv_format = KvFmt::kFp16;   ///< the K and V caches' format; fp16 by default (byte-identical path)
    int window = 0;               ///< the most tokens window() takes: the known one and guesses after it; 1: none;
                                  ///< 0: 8, or 4 where every guess keeps a copy of the recurrent state
    bool draft = true;            ///< run the block that guesses ahead, when there is one and window is above 1
    int draft_vocab = -1;         ///< the guesses come from the vocabulary's first this many tokens; 0: all of it;
                                  ///< -1: as many as the architecture's vocabulary has in order of use (guess_vocab())
    std::string draft_model;      ///< a file of the same model to take that block from, when the model's has none;
                                  ///< for an architecture that guesses with a model of its own, that model's file
    int tail_skip = 0;            ///< a routed expert outside VRAM that every token of a call ranks this far down or
                                  ///< further (the router's first choice is 0) is left out (--tail-skip,
                                  ///< STRATA_ROUTE_TAIL_SKIP); 0: none is
    float skip_weight = 0;        ///< in place of tail_skip: a routed expert outside VRAM whose router weight is under
                                  ///< this share of its token's experts' weight, for every token of the call, is left
                                  ///< out (--skip-weight); 0: tail_skip decides
    float cache_bias = 0;         ///< a token's few-at-a-time routing adds this to the score of each expert in VRAM
                                  ///< before it picks its experts; their weights stay the softmax of the scores the
                                  ///< router gave (--cache-bias, qwen35moe only); 0: the router's own picks
    std::string layer_split;      ///< the layers across the CUDA devices this process sees, in their order: "auto"
                                  ///< (by each card's free VRAM) or the first layer of each later card ("20,30");
                                  ///< empty, or one device: the current card alone
    int batch = 0;                ///< sequences beside the first, each with a state and caches of its own (bind(),
                                  ///< step()); 0: one sequence
};

/// A thing a model's file may call by other names than the engine does: converters differ.
struct Names {
    const char* name;
    const char* other[2];
};

/// A tensor in VRAM in its GGUF form.  n_in is the contiguous axis.
struct Dev {
    void* p = nullptr;
    uint32_t type = 0;
    int n_in = 0, n_out = 0;
};

/// What a layer has in every architecture.  Its attention's weights, and a dense FFN's, are the architecture's.
struct Layer {
    bool moe = false;
    bool mmq = false;     ///< the card has MMQ tiles for its routed experts' types (Engine::batched())
    bool fixed = false;   ///< one token's launches can be recorded and replayed for the next ones (eval()): they
                          ///< are the same for every token, or read the position on the card (Engine::at_)
    Dev attn_norm, ffn_norm;   ///< the norms ahead of the attention and of the FFN
    // MoE
    Dev gate_inp, sh_gate, sh_up, sh_down;
    const uint8_t* exp_gate = nullptr;  ///< n_expert experts back to back, in the mapped file
    const uint8_t* exp_up = nullptr;
    const uint8_t* exp_down = nullptr;
    uint32_t gu_type = 0, d_type = 0;
    size_t gu_bytes = 0, d_bytes = 0;  ///< one expert's gate (or up), and its down
    size_t gu_stride = 0;              ///< from an expert's gate to the next one's: twice gu_bytes in a file that
                                       ///< keeps an expert's gate and up in one tensor
    // up's own format (issue 103: IsValorum's NanoPlus files quantize gate and up differently); in every other file
    // these equal gu_type, gu_bytes and gu_stride, and nothing about the layer changes
    uint32_t u_type = 0;
    size_t u_bytes = 0, u_stride = 0;
    bool mixed() const { return u_type != gu_type; }
    size_t gu2() const { return gu_bytes + u_bytes; }   ///< an expert's gate and up: where its down starts in a slot
    size_t gu_step = 0;   ///< from a staged expert's gate|up to the next one's: gu2(), and in a mixed layer whole
                          ///< blocks of both types, since MMQ takes the stride in blocks (Mmq::mul)
    std::vector<uint8_t*> slot;        ///< per expert: gate | up | down in VRAM, or null
    size_t slot_bytes = 0;             ///< from a slot of the layer to the one behind it: whole blocks of both types
    std::vector<int32_t> ahead;        ///< experts a full prompt batch stages before it routes
    int trade = -1;                    ///< layers with the same number can take each other's slots: same size
    int8_t card = -1;                  ///< a few tokens' experts routed and found on the card (moe_card()): 1 yes,
                                       ///< 0 no (the architecture routes this layer on the host), -1 not tried
    // What a call of a few tokens hands the card, recorded (Engine::run()): by the call's tokens, and the head
    // also by Engine::head_key()
    std::vector<void*> heads;          ///< up to the router
    std::vector<void*> shared;         ///< the shared expert
    std::vector<void*> tail;           ///< after the experts
    std::vector<void*> card_moe;       ///< the routed experts, routed on the card (moe_card())
    /// the head of step()'s rows in one call (Engine::batch_heads()): by the rows, their sequences and head keys
    std::map<std::vector<int>, void*> rows_heads;
};

/// A routed expert is run from the VRAM cache or on the CPU when tokens come a few at a time, as an answer's do
/// (gpu_experts, cpu_experts), and in a prompt batch from the cache or the staging buffer (batch_*).
struct Stats {
    int64_t tokens = 0, gpu_experts = 0, cpu_experts = 0, batch_cached = 0, batch_staged = 0;
    int64_t swapped_experts = 0;   ///< experts that took a cached one's slot (follow())
    int64_t tail_experts = 0;      ///< a token's experts left out by Options::tail_skip
    double cpu_ms = 0, route_wait_ms = 0;
};

/// The most likely token of n logits: the first of the largest.
int most_likely(const float* logits, int n);

class Mmq;

/// A picture in a sequence: n rows that stand where n tokens would, each the vision encoder's output for a patch,
/// as wide as a token's embedding.  The patches come line after line, nx to a line.
struct Image {
    int64_t at = 0;   ///< the sequence's row its first one takes
    int n = 0, nx = 0;
    const float* rows = nullptr;
};

class Engine {
public:
    Engine();
    virtual ~Engine();
    Engine(const Engine&) = delete;
    Engine& operator=(const Engine&) = delete;

    /// The engine of the architecture the model file names, loaded.  Throws std::runtime_error, as do eval and
    /// the other calls that touch the device.
    static std::unique_ptr<Engine> open(const Options& o);
    void load(const Options& o);

    int n_vocab() const { return n_vocab_; }
    /// The rows of the output a guess is picked from.  A guess that is never right costs a window's place and no
    /// token, so the rare tokens may be left out, where the vocabulary has the common ones first.
    int draft_rows() const {
        const int v = opt_.draft_vocab >= 0 ? opt_.draft_vocab : guess_vocab();
        return v > 0 ? std::min(v, n_vocab_) : n_vocab_;
    }
    int n_embd() const { return n_embd_; }
    int max_context() const { return opt_.max_context; }
    int threads() const { return workers_ ? workers_->threads() : 0; }
    /// The NUMA node the CPU threads were kept on, and the share of the experts' pages it holds; -1: no node chosen.
    int numa_node() const { return node_; }
    double numa_share() const { return node_share_; }
    int64_t pos() const { return pos_; }
    int64_t meta_u(const char* key, int64_t fallback) const;
    /// What --kv asked for, in the two short forms an architecture's code reads: the int8 codes/delta cache
    /// (both sides, or the keys' side alone under int8k).
    bool int8() const { return opt_.kv_format == KvFmt::kInt8 || opt_.kv_format == KvFmt::kInt8K; }
    bool int8k() const { return opt_.kv_format == KvFmt::kInt8K; }

    /// Forget the sequence.
    void reset();
    /// STRATA_FREEZE_EXPERTS: lift the freeze for the caller's own cache change (the serve VRAM
    /// resize); refreeze_experts() puts it back, and the resized cache is what every request reads.
    void unfreeze_experts();
    void refreeze_experts();
    /// The most tokens one eval() takes.
    int max_rows() const { return rows_; }
    /// Feed n tokens from position pos().  With want_logits, logits() then holds the scores of the token after the
    /// last one.
    void eval(const int32_t* tokens, int n, bool want_logits);
    const float* logits() const { return next_ ? next_->logits() : h_logits_; }
    /// Feed n tokens of a prompt, none of them a picture's row, as eval() without logits would in calls of
    /// max_rows(): the same tokens a call.  With experts outside the cache it is faster than those calls, the
    /// more the more of them n is; under stage_ahead tokens it makes those calls.  The result is the calls' own
    /// bit for bit, but for a call of fewer than stage_ahead tokens beside others: its experts' sums are split as a
    /// larger call's, not as on its own.
    ///
    /// A mark is a place inside the read where the caller would have stopped between two calls: `at` of the
    /// read's tokens come before it.  The calls end there as they would have, and `state`, when it is not null,
    /// gets what save_state() gives at that place.  The marks rise, each above 0 and below n.
    struct Mark {
        int at;
        uint8_t* state;
    };
    int max_read() const { return read_rows_; }
    /// `turn`, when given, is asked each time the read's tokens are through one more layer (or, where the read
    /// is its calls one by one, through one more call), with how many of how many that is.  False gives the read
    /// up: read() returns false, and what the engine holds from pos() on is spoiled, its recurrent state and a
    /// window layer's rows among it.  The caller goes back to a state saved before (load_state()) or resets.
    bool read(const int32_t* tokens, int n, const Mark* marks = nullptr, int n_marks = 0,
              const std::function<bool(int, int)>* turn = nullptr);
    /// The prompt's reads are done: the expert cache takes back what Options::stage_read_mib lent the staging
    /// buffer.  window() and step() do it too, if the caller has not.
    void end_read() {
        if (stage_lent_) narrow_stage();
    }

    /// Whether the architecture reads pictures.
    virtual bool sees() const { return false; }
    /// The pictures of the sequence, in its order, in place of the ones set before.  eval() reads a row that lies
    /// in one from the picture and not from its token, so their rows must stay where they are while tokens are
    /// read.  A sequence that is come back to must get the pictures it was read with.
    void images(const Image* im, int n);
    /// Of m rows from row `at`, how many the next eval() takes.  An architecture whose pictures' rows see each
    /// other takes a picture in a call of its own, which can be more than m.
    virtual int64_t whole(int64_t at, int64_t m) const { (void) at; return m; }

    /// Guessed tokens.  window() feeds t tokens from pos(): the first is the one that is known, each of the others a
    /// guess at what follows the one before it.  logits_of(i) then holds the scores of the token after the i-th, so
    /// the guesses can be checked, and keep(n) makes the first n tokens part of the sequence and forgets the rest.
    /// What a kept token computed is what eval() would have computed for it.
    int max_window() const { return opt_.window; }
    void window(const int32_t* tokens, int t);
    const float* logits_of(int i) const { return next_ ? next_->logits_of(i) : h_wlogits_ + (size_t) i * n_vocab_; }
    void keep(int n);

    /// Some files carry a block that guesses ahead: one layer that takes a token and the model's output for the
    /// token before it, and gives what the model would likely say after the token.  drafts(): it is there and runs.
    /// draft(token, 0, &p) takes the token at pos(), picked and not fed yet, and returns the guess at the one after
    /// it; draft(guess, i, &p) for i = 1, 2, ... then guesses on from the i-th guess, with the block's own output
    /// in the model's place.  p is the probability the block gives what it returns.  pos() + i must be inside the
    /// context.  logits() is left alone.  An architecture may guess with a small model of its own instead of a
    /// block (draft_step()); the calls are the same.  A model trained to guess a block of tokens at once may guess
    /// them itself (block_mask(), under STRATA_BLOCK_GUESS=1): draft(token, 0, &p) makes a window's guesses in one
    /// call, and the calls after it return them, whatever guess they are given.  It does so only while the card holds
    /// every expert: the block runs the whole model, and with experts on the CPU it cost more than its right guesses
    /// saved (Clover on an 8 GB laptop wrote 57.7 tokens/s after a 32K prompt with its guesses, 61.6 without).
    bool drafts() const { return next_ ? next_->drafts() : draft_ || (block_guess_ && all_cached()); }
    /// The guesses of a window come out of draft(token, 0, &p) together: asking for one more costs nothing.
    bool guesses_at_once() const { return next_ ? next_->guesses_at_once() : block_guess_; }
    /// Test only: the calls read()'s layer-major path gave the draft block's rows (draft_rows()) on the card that has
    /// the block, since load.  Two or more in one read are back-to-back refills of its pinned input rows.
    int64_t read_draft_calls_for_test() const { return next_ ? next_->read_draft_calls_for_test() : read_draft_calls_; }
    int32_t draft(int32_t token, int i, float* prob);
    /// Whether a call since the last time this was asked recorded launches for replaying, which takes far longer
    /// than the call does afterwards: its time says nothing of the calls to come.
    bool took_record() {
        const bool later = next_ && next_->took_record();
        return std::exchange(recorded_, false) || later;
    }

    // A sequence's state, to leave the sequence and come back to it.  The recurrent layers have a state that
    // every token changes.  Each token also leaves a row in the attention caches, which stays as it was written.
    // On several cards: the first card's part, then the next one's.
    size_t state_bytes() const { return card_state_bytes() + (next_ ? next_->state_bytes() : 0); }
    size_t row_bytes() const { return card_row_bytes() + (next_ ? next_->row_bytes() : 0); }
    /// The recurrent state at pos(), state_bytes() long.
    void save_state(uint8_t* dst);
    /// Back to position `pos` with the recurrent state it had there.  The rows before pos must be the sequence's.
    void load_state(const uint8_t* src, int64_t pos);
    /// The rows of positions [first, first + n), n * row_bytes() long: layer after layer, a layer's rows together.
    void save_rows(int64_t first, int64_t n, uint8_t* dst);
    /// Rows [first, first + n) out of what save_rows wrote for [from, from + count).
    void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n);

    const Stats& stats() const;
    /// Between requests, and once after the load: of the experts the card has, the pages of the half used most
    /// lately leave this process's working set (after the load only under Options::ram_experts_mib), and so do the
    /// CPU's experts' under Options::ram_experts_mib and cuBLAS's pages; the pinned memory only a long prompt needs
    /// is given back.
    void trim();
    int cached_experts() const { return cached_ + (next_ ? next_->cached_experts() : 0); }
    /// The RAM held for the experts outside VRAM, in MiB, as load() settled Options::ram_experts_mib (-1: all of
    /// them), and those experts' GiB.
    int64_t ram_experts_mib() const { return opt_.ram_experts_mib; }
    double outside_gib() const { return double(outside_bytes_) / double(1ull << 30); }
    int total_experts() const { return n_moe_ * n_expert_ + (next_ ? next_->total_experts() : 0); }
    double cached_gib() const { return double(exp_at_) / double(1ull << 30) + (next_ ? next_->cached_gib() : 0); }
    /// The cache as load() filled it, which is as large as resize_cache() makes it again.
    int full_experts() const { return full_cached_ + (next_ ? next_->full_experts() : 0); }
    double full_gib() const { return double(full_bytes_) / double(1ull << 30) + (next_ ? next_->full_gib() : 0); }
    /// The VRAM the cache was last sized to leave free, and what is free now, in MiB: on several cards, the
    /// first card's reserve and the least that one of them has free.
    int reserve_mib() const { return reserve_mib_; }
    int64_t vram_free_mib() const;
    /// The expert cache sized again, to leave `mib` of VRAM free (below 0: what load() left), and filled with the
    /// experts used most so far.  Without VMM it gets no larger than load() made it.  Between calls that feed
    /// tokens: the experts it gives up are computed on the CPU from then on, as any expert outside it.
    void resize_cache(int mib);
    /// Between requests: the cache gives up VRAM when little is free, and takes VRAM that came free (see the
    /// definition for how much stays free).  True when it changed.
    bool mind_vram();
    /// At a request's start, the engine at the furthest point of its `end` tokens it could come back to: the
    /// attention caches give back the rows past `end` that the request before did not reach either.  One long
    /// request leaves them grown, and the expert cache would stay that much smaller for every request after it.
    /// Returns the bytes given back, which mind_vram() hands to the expert cache.
    size_t trim_kv(int64_t end);
    void save_profile() const;

    /// Options::batch + 1 sequences, each with its own position, recurrent state, attention rows and pictures.  One
    /// is bound at a time, and every call above but step() is the bound one's.  bind(s) leaves the bound one as it
    /// is and goes on with sequence s where it was left.  Not between window() and keep().
    int sequences() const { return (int) seqs_.size(); }
    int bound() const { return bound_; }
    void bind(int s);
    /// One token for each of n sequences, together: tokens[i] fed to sequence seqs[i] at its position, as eval()
    /// with logits would on its own.  logits_of(i) then holds the scores of what follows tokens[i].  The sequences
    /// differ; none of them has pictures past its position.  The bound sequence stays bound.
    void step(const int* seqs, const int32_t* tokens, int n);
    /// Sequence `to` becomes what sequence `from` is: its position, state, rows and pictures, copied on the card.
    void copy_seq(int from, int to);

protected:
    // ---- what an architecture supplies ----
    /// The model's shape from its metadata: load_shape() for what every architecture has, then its own.
    virtual void load_geometry() = 0;
    /// A layer's weights, to VRAM (upload()) or as places in the mapped file (load_moe()).  load() runs the layers
    /// through twice: begin_pass() is called before each.
    virtual void load_layer(int il) = 0;
    virtual void begin_pass() {}
    /// Throws when the draft block is not one the architecture runs.
    virtual void check_draft_block() {}
    /// The weights of what guesses ahead, to VRAM.  The engine's own: the draft block's parts beside its layer.
    virtual void load_drafter();
    /// One guess (draft()): the token through what guesses ahead, its scores into logits_.  hn_'s first row is the
    /// model's output for the token before pos(); the second is this step's to leave and the next step's to read.
    /// The engine's own runs the draft block.
    virtual void draft_step(int32_t token, int i);
    /// The floats of every recurrent layer's state together.  They are state_'s first.
    virtual size_t recurrent_floats() const = 0;
    /// Whether the recurrent layers leave their state after any tokens of a call, and not only after its first
    /// ones.  A window then takes more tokens than it has places for their states (snap_tokens()).
    virtual bool snaps_spread() const { return false; }
    /// How many of the vocabulary's first tokens hold nearly every token an answer writes; 0: the order says
    /// nothing of that.
    virtual int guess_vocab() const { return 0; }
    /// The token a model trained to guess a block of tokens at once fills the block with after the known one
    /// (guess_block()); -1: the model is not.  Its layers then run a call of block_pass_ as that block.
    virtual int block_mask() const { return -1; }
    /// In a call of block_pass_: what is done to the n tokens' scores at layer il's router, together, before each
    /// token is routed (route()).
    virtual void route_block(int il, float* scores, int n) { (void) il, (void) scores, (void) n; }
    /// The layers' places in state_, and the attention's caches: a row a position.  Only for the layers this
    /// engine runs (runs()).
    virtual void alloc_caches() = 0;
    /// What row_bytes(), save_rows() and load_rows() are for the layers this engine runs.
    virtual size_t card_row_bytes() const = 0;
    virtual void save_card_rows(int64_t first, int64_t n, uint8_t* dst) = 0;
    virtual void load_card_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) = 0;
    /// The scratch of the architecture's own kernels, a fbuf() each.  Called twice: the first time only counts.
    virtual void assign_scratch() = 0;
    /// The widest input lin() gets, beside the model's width and the FFNs'.
    virtual int widest() const = 0;
    /// The copies of a token's embedding a layer's input starts from.
    virtual int streams() const = 0;
    /// A layer from its input to its router's copy (a layer without routed experts: to its end).
    virtual void layer_head(Layer& L, int il, int n) = 0;
    /// The routed experts' sum (fo_) into the layer's output.
    virtual void layer_tail(Layer& L, int il, int n) = 0;
    /// The model's output for `rows` tokens from row `first` of the call, normed: what the output matrix takes.
    virtual void output_rows(int first, int rows, float* h) = 0;
    /// A token's experts and their weights from its scores at layer il's router.  `token` is the token the row is
    /// for, or -1 where it has none (a picture's row, the draft block).  tmp: 2 * n_expert_ floats of scratch.  A
    /// prompt batch's tokens are routed by several threads at once: it changes nothing of the engine's.
    virtual void route(int il, const float* logits, int32_t token, int* ids, float* w, float* tmp) const = 0;
    /// A prompt batch's routing on the card: n tokens' experts and weights from their scores in rl_, n_used_ a token
    /// into ids and w (on the card), what route() gives them up to the last bits of a weight.  False: the host routes.
    virtual bool route_card(int il, int n, int32_t* ids, float* w) {
        (void) il, (void) n, (void) ids, (void) w;
        return false;
    }
    /// The shared expert, into fo_.  It runs on the GPU while the host routes.
    virtual void moe_shared(const Layer& L, int il, int n) = 0;
    /// The scores of a layer's router for n tokens, into rl_.  The engine's own: the router's matrix times the
    /// experts' input (xn_).
    virtual void router_scores(const Layer& L, int n);
    /// What is done to `rows` rows of logits on the card before they are copied out.
    virtual void finish_logits(float* logits, int rows) { (void) logits, (void) rows; }
    /// What a token's embedding is multiplied by on its way in.
    virtual float embed_scale() const { return 1.f; }
    /// What a sequence has beside its recurrent state and its rows, in bytes: it is saved and brought back with
    /// the state (save_state(), load_state()).  load_extra() gets the position the state is for.
    virtual size_t extra_bytes() const { return 0; }
    virtual void load_extra(const uint8_t* src, int64_t pos) { (void) src, (void) pos; }
    /// Layer il's part of what save_state() writes, to its place in dst: the layer's recurrent state, and what
    /// extra_bytes() counts for it, as they are once the layer has run every token before `pos`.  The card is
    /// done with the layer when this is called.  save_state() is this for every layer; read() calls it for a
    /// layer when the layer gets to a checkpoint, which is before the layers after it do.
    virtual void save_layer(int il, int64_t pos, uint8_t* dst) = 0;
    /// The draft block's attention for n tokens from position `pos`.  xn_ and xq_ hold its normed input; ao_ gets
    /// its output, and the tokens leave their rows in its caches.  block_rows() only leaves the rows.
    virtual void block_attend(int64_t pos, int n) { (void) pos, (void) n; }
    virtual void block_rows(int64_t pos, int n) { (void) pos, (void) n; }
    /// Which recording of a layer's head fits n tokens from pos().  Launches that are sized by the rows they read
    /// are recorded once for each size (room()), and the architecture numbers the sizes from 0.
    virtual int head_key(int il, int n) const { (void) il, (void) n; return 0; }
    /// layer_head() takes step()'s rows in one call: the products for every row at once, and each row's recurrent
    /// state and attention rows from its own sequence (rows_of_, seq_state(), seq_buf()), its position from at_.
    /// Otherwise step() runs the head a row at a time, with the row's sequence bound.
    virtual bool batch_heads() const { return false; }
    /// While step() runs a head in one call: each row's sequence and position, row after row; else null.
    struct Row {
        int seq;
        int64_t pos;
    };
    const Row* rows_of_ = nullptr;
    int n_rows_of_ = 0;
    /// p, a place in the bound sequence's recurrent state, at the same place in sequence s's.
    float* seq_state(int s, const float* p) const;
    /// p, a place in one of the bound sequence's buffers (kv_alloc(), seq_alloc()), at the same place in s's.
    void* seq_buf(int s, const void* p) const;
    /// The rotary position of a row of text: its place in the sequence, unless pictures before it took fewer
    /// positions than rows.
    virtual int64_t turn(int64_t row) const { return row; }
    /// The buffers that hold a row a token, and pos(), moved `by` rows on (back, when negative): a call for one
    /// token then works in the place of another row of a window (run()).  An architecture adds its own buffers
    /// that a layer's head leaves for the layer's tail.
    virtual void shift_rows(ptrdiff_t by);
    /// The tokens of a window of t that leave the recurrent state as it is after them, a bit a token.
    unsigned snap_tokens(int t) const;
    /// The other names of a tensor or a metadata key in the files of this architecture, or null.
    virtual const Names* tensor_alias(const char* name) const { (void) name; return nullptr; }
    virtual const Names* meta_alias(const char* key) const { (void) key; return nullptr; }

    // ---- what it builds on ----
    /// Whether layer il is this engine's: one of [lo_, hi_), or the draft block on the card that ends the model.
    bool runs(int il) const { return il >= lo_ && (il < hi_ || hi_ == n_layer_); }
    size_t card_state_bytes() const { return state_n_ * sizeof(float) + extra_bytes(); }
    /// Every expert of this engine's layers is in its cache.
    bool all_cached() const { return cached_ == n_moe_ * n_expert_; }
    /// The rows a call of n tokens leaves in r_, to h (pinned) for the next card's engine, which goes on from them.
    void hand_on(int64_t pos, int n, float* h);
    float* pass_rows(int n);
    /// kGuessMat: quantized again to Q4_0, for a matrix only guesses are made with: a guess is checked, so a less
    /// exact one costs a wrong guess now and then and no token.
    enum Want { kF32, kFloatMat, kAnyMat, kGuessMat };
    Dev upload(const TensorInfo* t, const std::string& name, int64_t n_in, int64_t n_out, Want want,
               float (*fix)(float) = nullptr);
    const TensorInfo* tensor(int il, const char* name) const;
    const Shards& home(const TensorInfo& t) const;
    const MetaValue* meta_in(const Shards& g, const char* key) const;
    /// A number of the model's shape, which its file must have.
    int shape_i(const char* key) const;
    float shape_f(const char* key) const;
    void load_shape();
    void load_moe(int il, bool shared = true);
    void alloc();
    void shared_scratch();
    float* draft_input(const int32_t* tokens, int64_t row, const float* h, int n);
    void draft_rows(const int32_t* tokens, int first, int n);
    void guess_block(int32_t token);
    const Image* image_at(int64_t row) const;
    bool input(int32_t token, int64_t row, float* dst) const;
    void lin(const Dev& w, const float* x, float* y, int n, const void* xq = nullptr, int rows_per_x = 0);
    void fill_expert_cache();
    void start_swaps();
    void plan_ahead(Layer& L, int il);
    void run(const int32_t* tokens, int n);
    bool pick_way(int t);
    void took_way(int t, double ms);
    void layer(int il, int n);
    void save_last(uint8_t* dst);
    void moe_router(const Layer& L, int n);
    bool moe(Layer& L, int il, int n, const std::function<void()>* first = nullptr);
    bool card_moe(int n);
    bool moe_card(Layer& L, int il, int n);
    const float* biased(const Layer& L, const float* scores);
    /// Whether n tokens take layer L's routed experts as a prompt batch, through MMQ (moe_rows()), and not a few at
    /// a time (moe()).
    bool batched(const Layer& L, int n) const;
    /// The routed experts' rows a call routed on the card has room for (d_): a window of a model that guesses its
    /// own block, or kSumRows.  The others keep kSumRows, whose windows past it are routed on the host as before.
    int card_rows() const;
    void moe_add(int n);
    template <class F>
    void* record(F&& work);
    void moe_rows(Layer& L, int il, int n);
    void stage(const Layer& L, const int32_t* experts, int m, void* stream, bool alone = false);
    void mind_memory();
    void follow(int written);
    void follow_loop();
    void embed(int32_t token, float* dst) const;
    void dbg(const char* name, int il, const float* dev, int n);
    float* fbuf(size_t n);
    int room(int n) const;
    int room_key(int n) const;
    void* dmalloc(size_t bytes, const char* what);
    void* hmalloc(size_t bytes, const char* what);
    void hfree(void* p);
    std::vector<uint8_t> ram_keep(const std::vector<float>* lately = nullptr) const;
    bool read_file(uint8_t* dst, const uint8_t* src, size_t n) const;
    /// A write into VRAM, on st_.  Every one of them goes through these two or is an async call on a stream of
    /// its own: see the note on h2d() in engine.cpp.
    void h2d(void* dst, const void* src, size_t bytes);
    void zero_vram(void* dst, size_t bytes);
    /// An attention cache for alloc_caches(): `bytes` for every `per` rows, with addresses for the whole context
    /// and VRAM for the rows the sequence has reached (cover()).
    void* kv_alloc(size_t bytes, int per);
    /// A buffer of the sequence's own that does not grow with it, such as a window layer's ring of rows.
    void* seq_alloc(size_t bytes, const char* what);
    /// VRAM in every attention cache for the rows below `end`, before a call writes them.
    void cover(int64_t end);

    // VRAM mapped onto addresses reserved for the most it can be, a piece at a time (CUDA's virtual memory
    // management): it grows and shrinks at its end and keeps its address, which recorded launches hold.
    struct Mapped {
        unsigned long long base = 0;   ///< a CUdeviceptr
        size_t reserved = 0, mapped = 0;
        std::vector<std::pair<unsigned long long, size_t>> pieces;   ///< allocation handles and their bytes, in order
    };
    struct Kv : Mapped {
        size_t bytes = 0;   ///< for every `per` rows
        int per = 1;
    };
    void vmm_reserve(Mapped& m, size_t bytes);
    void vmm_map(Mapped& m, size_t bytes, size_t piece, const char* what);
    void vmm_unmap(Mapped& m, size_t bytes);
    void vmm_free(Mapped& m);
    void shrink_cache(size_t bytes);
    int grow_cache(size_t bytes, bool later = false);
    void fill_now();
    void widen_stage();
    void narrow_stage();
    void resize_card_cache(int mib);   ///< resize_cache() and mind_vram() for this engine's card
    bool mind_card_vram();
    size_t trim_card_kv(int64_t end);
    int64_t kv_rows_for(int64_t end) const;   ///< the rows cover() maps for `end`
    void held_is_budget();
    void settle_swaps();

    Options opt_;
    std::unique_ptr<Shards> gg_;
    std::unique_ptr<Shards> dg_;      ///< Options::draft_model, while the draft block is read from it
    const Shards* block_ = nullptr;   ///< the file the draft block is in: gg_ or dg_
    std::shared_ptr<Workers> workers_;   ///< on several cards, one for all of them: they take turns
    int node_ = -1;
    // A layer split (open()).  The layers of the model are [lo_, hi_) here, and the engine of the next card has
    // the ones after.  A later card's engine reads its first layer's input from pass_, which the one before wrote:
    // a row of the residual a position, from pass_at_.
    int lo_ = 0, hi_ = 0;
    int end_ = -1;                        ///< what open() asked for hi_; -1: the model's end
    bool split_ = false;                  ///< one of several engines: a call sets its card first
    std::unique_ptr<Engine> next_;
    int64_t read_draft_calls_ = 0;               ///< read_draft_calls_for_test()
    const float* pass_ = nullptr;
    int64_t pass_at_ = 0;
    float* h_pass_ = nullptr;             ///< pinned; h_pass_rows_ tokens' rows
    int h_pass_rows_ = 0;
    mutable Stats all_stats_;             ///< stats() of every card's engine together
    double node_share_ = 0;
    std::thread mapper_;                  ///< reads the experts' pages once after load, so they are mapped
    std::atomic<bool> map_quit_{false};
    void* st_ = nullptr;   ///< the stream everything but the staging ahead runs on
    bool freeze_experts_ = false;            ///< STRATA_FREEZE_EXPERTS: the experts' split is frozen
                                             ///< for the engine's whole run (see Engine::load())
    bool freeze_experts_asked_ = false;      ///< the env was set: refreeze_experts() puts it back after the
                                             ///< serve VRAM resize lifted it (unfreeze_experts())
    void* blas_ = nullptr; ///< cuBLAS on st_
    void* ev_ = nullptr;   ///< marks the router's copy to the host: ev_row_'s first, or a window's row's own
    void* ev_row_[8] = {}; ///< one for each token a window can have
    std::vector<void*> dev_allocs_, host_allocs_;
    std::vector<void*> graphs_;   ///< every recorded graph, to destroy
    std::map<std::vector<int>, void*> passes_;   ///< a call's layers recorded as one (run()): by its tokens and
                                                 ///< every layer's head_key()
    void* tail_ = nullptr;        ///< one token's launches from the last layer to the logits, recorded
    // A recorded launch cannot be given the position as an argument: it reads it here when it runs.  eval() writes
    // the token's place in the sequence and its rotary position before it replays the layers.
    long long* at_ = nullptr;     ///< on the card: {pos(), turn(pos())}, and the same for a window's later tokens
    long long* h_at_ = nullptr;   ///< pinned
    bool recording_ = false;      ///< a layer's launches are being recorded: positions are read from at_
    bool recorded_ = false;       ///< took_record()

    // geometry
    int n_embd_ = 0, n_layer_ = 0, n_vocab_ = 0;
    int n_ff_ = 0;            ///< a dense FFN's width, in the architectures that have one
    int n_expert_ = 0, n_used_ = 0, n_ff_exp_ = 0, n_ff_shexp_ = 0, n_moe_ = 0;
    int rows_ = 1;            ///< tokens the scratch is sized for
    int read_rows_ = 1;       ///< the most tokens read() takes
    // read()'s calls between two layers: r_ holds two calls' rows, and with more calls than that the others'
    // wait in host memory, copied out and in beside the card's work on the call between
    struct Call {
        int at, m, mark;      ///< its first token, how many, and the mark it ends at or -1
    };
    std::vector<Call> calls_;
    float* h_res_ = nullptr;  ///< h_res_rows_ tokens' rows, pinned: as many as the longest read so far that needed it
    int h_res_rows_ = 0;
    void* rs_ = nullptr;      ///< the stream of those copies
    void* ev_in_[2] = {};     ///< marks a call's rows in each half of r_
    void* ev_out_[2] = {};    ///< marks the card done with a call in each half
    float rms_eps_ = 0;
    bool gelu_ = false;       ///< the routed experts' activation: GELU of the gate, else SiLU within the layer's limits
    std::vector<float> clamp_exp_, clamp_shexp_;   ///< a layer's SwiGLU limits: the routed experts', the shared one's
    /// a prompt batch's products go through guard_half_sums(): the model's activations can sum past a half's range
    /// in a block.  Off for the models whose sums stay in range: it cost LFM2.5 4% of a 30K prompt's read on a 5090.
    bool half_sum_guard_ = false;

    std::vector<Layer> layers_;   ///< the model's, and after them the draft block
    Dev out_norm_, output_;
    bool draft_ = false;
    bool block_layer_ = false;    ///< what guesses ahead is a block after the model's layers (layers_' last)
    bool block_guess_ = false;    ///< the model guesses a window's tokens itself, in a block (block_mask())
    bool block_pass_ = false;     ///< the call under way is that block: its tokens see each other, no state changes
    Likeliest* h_block_ = nullptr;   ///< the block's guesses and their probabilities, pinned
    int block_n_ = 0;                ///< how many it holds
    Dev d_proj_, d_enorm_, d_hnorm_, d_head_;   ///< the draft block's input projection and norms, and its output norm
    const uint8_t* tok_embd_ = nullptr;   ///< the token table in the mapped file, tok_row_ bytes a token
    size_t tok_row_ = 0;
    uint32_t tok_type_ = 0;

    uint8_t* dense_ = nullptr;   ///< the dense weights' arena
    size_t dense_bytes_ = 0, dense_at_ = 0;
    uint8_t* exp_ = nullptr;     ///< the routed experts' arena
    size_t exp_at_ = 0;
    Mapped arena_;               ///< its VRAM, when the card maps VRAM a piece at a time
    bool vmm_ = false;           ///< it does; without that every cache is as large as it can get from the start
    int device_ = 0;
    size_t grain_ = 0;           ///< what the card maps VRAM in
    std::vector<Kv> kv_;         ///< the attention caches that grow with the sequence
    int64_t kv_rows_ = 0;        ///< the rows they have VRAM for
    int64_t kv_reach_ = 0;       ///< the furthest row cover() was asked for since the last trim_kv()
    // The sequences (bind()).  The bound one's are the members above and below (pos_, kv_, state_ ...); the
    // others wait in seqs_, with what alloc_caches() gave them and the launches recorded with their buffers.
    struct Buf {
        void* p = nullptr;
        size_t bytes = 0;   ///< a kv_alloc() buffer's for every `per` rows; a seq_alloc() one's, all of it
        int per = 0;        ///< 0: seq_alloc()'s
    };
    struct Seq {
        int64_t pos = 0, kv_rows = 0, kv_reach = 0, ahead_pos = -1;
        int32_t ahead_tok = 0;
        float* state = nullptr;
        std::vector<Kv> kv;
        std::vector<Buf> bufs;
        std::vector<Image> images;
        std::vector<std::vector<void*>> heads;   ///< Layer::heads, by layer
        std::map<std::vector<int>, void*> passes;
        void* tail = nullptr;
        // A sequence's own fp16 twin of its int8 caches (Gemma 4's --kv int8), and the halves of it that have
        // VRAM behind them.  The twin is a sequence's: a batch gives every row its own sequence, and the twin is
        // where that sequence's prompt batch reads, so the twin belongs with the sequence's other members.
        Mapped twin_k, twin_v;
        size_t twin_mapped = 0;
    };
    std::vector<Seq> seqs_;      ///< the bound one's place holds nothing while it is bound
    int bound_ = 0;
    std::vector<Buf> bufs_;      ///< the bound sequence's: what kv_alloc() and seq_alloc() gave it, in that order
    size_t again_ = SIZE_MAX;    ///< while alloc_caches() points at the bound sequence's buffers again: the next one
    // The bound sequence's fp16 twin of its int8 caches (Gemma 4's --kv int8): the addresses for the whole and
    // the VRAM behind them.  A sequence's own, since a batch gives every row its own sequence.
    Mapped twin_k_, twin_v_;
    size_t twin_mapped_ = 0;
    void park_bound();
    void take_bound(int s);
    void hold(int s);   ///< bind() on this card alone
    size_t other_rows() const;
    size_t budget_ = 0;          ///< their VRAM and the expert cache's together, as the cache was filled
    int asked_reserve_mib_ = -1; ///< Options::vram_reserve_mib as load() got it
    int reserve_mib_ = 0;        ///< what the cache was sized to leave free
    bool ram_tight_ = false;     ///< the machine's RAM is within 5 GiB of the model's files (fill_expert_cache())
    int minds_ = 0;              ///< mind_vram()'s calls
    int full_cached_ = 0;        ///< the cache as load() filled it: its experts and their bytes
    size_t full_bytes_ = 0;
    float* scratch_ = nullptr;
    size_t scratch_at_ = 0;
    float* state_ = nullptr;     ///< every recurrent layer's state; with a draft block, hn_'s first row
    size_t state_n_ = 0, rec_n_ = 0;   ///< all of it, and the recurrent layers' part
    // The draft block reads the model's output for the token before each of its tokens.  Row 0 is the output of
    // the last token of the sequence, and the outputs of the tokens of the call under way follow it.
    float* hn_ = nullptr;
    Likeliest* h_guess_ = nullptr;   ///< a guess and its probability, pinned
    int32_t win_tok_[8] = {};           ///< the window's tokens, for keep()
    int64_t ahead_pos_ = -1;            ///< draft() has left the block's row of this position as its token leaves it,
    int32_t ahead_tok_ = 0;             ///< which is this one
    float* snap_ = nullptr;      ///< window(): the recurrent state after tokens of it (snap_tokens()), rec_n_ apart
    int snap_places_ = 0;        ///< how many states it holds
    float* snap_at_ = nullptr;   ///< where the call under way leaves the first of its states
    unsigned snap_bits_ = 0;     ///< the tokens of the call under way that leave their state, a bit a token
    bool windowed_ = false;      ///< the call under way is a window's
    int win_ = 0;                ///< the tokens of the window that keep() has not settled yet
    // A window with experts on the CPU goes through the layers whole, or in two halves one behind the other
    // (run()).  Which takes less time depends on the card, on the CPU and on how many of the experts the cache
    // has, so each way's time is kept for each window size (pick_way(), took_way()).
    struct Way {
        double ms = 0;           ///< what a window took this way, the later ones counting for more
        int n = 0;               ///< how many were timed
    };
    Way way_[2][9];              ///< [in halves][the window's tokens]
    int way_calls_[9] = {};      ///< windows of each size since the way behind had one
    int way_backoff_[9] = {};    ///< the way behind waits kWayAgain << this many windows of the size
    bool halves_ = false;        ///< the way of the window under way
    float *wlogits_ = nullptr, *h_wlogits_ = nullptr;   ///< a window's logits, token after token: VRAM and pinned
    int cached_ = 0;

    // device scratch: a layer's input streams, one stream, the normed input, the attention's and the FFN's output
    float *r_ = nullptr, *cur_ = nullptr, *xn_ = nullptr, *ao_ = nullptr, *fo_ = nullptr;
    float* wf_ = nullptr;      ///< the largest BF16 matrix as floats: lin() multiplies a prompt batch by cuBLAS
    size_t bf16_max_ = 0;      ///< that matrix's values
    float *g_ = nullptr, *u_ = nullptr, *h_ = nullptr, *d_ = nullptr, *rl_ = nullptr,
          *logits_ = nullptr;
    void *xq_ = nullptr, *q8_ = nullptr;

    // pinned host buffers
    float *h_r_ = nullptr, *h_rl_ = nullptr, *h_x_ = nullptr, *h_cpu_ = nullptr, *h_logits_ = nullptr;

    // A prompt batch's MoE (moe_rows).  A row is a token and one of the experts it routes to.
    std::unique_ptr<Mmq> mmq_;   ///< null when the card has no MMQ tile for the experts' types
    float *gu_ = nullptr, *hx_ = nullptr, *dx_ = nullptr;   ///< the rows' gate|up, hidden and down values
    float* hf_ = nullptr;   ///< the power of two a row was divided by for its half sums (guard_half_sums())
    void *mq_ = nullptr, *mq2_ = nullptr;                   ///< the rows quantized for the two products
    void* mqu_ = nullptr;   ///< the rows quantized for up's product, in a layer whose up is not gate's type
    void* mqx_ = nullptr;                                   ///< lin()'s tokens quantized for a product
    int32_t *tab_ = nullptr, *h_tab_ = nullptr;   ///< a layer's row tables, in VRAM and pinned
    int32_t* iota_ = nullptr;                     ///< 0, 1, 2, ...
    int32_t* d_ids_ = nullptr;                    ///< a batch's experts and weights when it routes on the card
    float* d_wts_ = nullptr;
    int32_t* h_ids_ = nullptr;                    ///< pinned: the experts, back from the card
    void** d_slots_ = nullptr;                    ///< every layer's L.slot on the card, n_expert_ a layer (moe_card())
    std::vector<void*> h_slots_;                  ///< what d_slots_ holds
    uint8_t* stage_ = nullptr;                    ///< experts that are not cached, on their way through VRAM
    uint8_t* h_stage_[2] = {};                    ///< pinned; filled in turn
    void* ev_stage_[2] = {};                      ///< marks the upload out of each
    void* up_ = nullptr;                          ///< the stream of the uploads that start before the routing
    void* ev_ahead_ = nullptr;                    ///< marks the last of them
    void* ev_free_ = nullptr;                     ///< marks a layer's last read of the staging buffer
    void* ev_hr_ = nullptr;                       ///< marks the card's last read of h_r_ by draft_input()
    int staged_il_ = -1;                          ///< the layer whose experts the buffer holds for its next call, or -1
    std::vector<int32_t> staged_;                 ///< those experts, in the buffer's order: one round of the layer's
    double ahead_ms_ = 0, route_ms_ = 0;          ///< what that layer's copy ahead took, and the card to its routing
    bool shared_ahead_ = false;                   ///< read() runs several calls: each copies ahead, whatever its size
    size_t stage_bytes_ = 0, chunk_bytes_ = 0;
    Mapped stage_map_;                            ///< stage_'s VRAM when it can grow for a read (stage_read_mib)
    size_t stage_narrow_ = 0, stage_wide_ = 0;    ///< stage_bytes_ outside a read and inside one
    bool stage_lent_ = false;                     ///< the expert cache has lent the buffer VRAM (widen_stage())
    size_t stage_cut_ = 0;                        ///< what the cache gave up for it, which it takes back
    int stage_turn_ = 0;
    std::vector<int32_t> ids_, count_, first_, group_, round_;
    std::vector<float> route_tmp_;   ///< route()'s scratch, a share for each host thread
    std::vector<float> bias_rl_;     ///< one token's scores with Options::cache_bias added (moe())
    struct Run {
        int c, j, m, most;   ///< the rows' class, its first cached expert and how many, the most rows one has
    };
    std::vector<Run> runs_;        ///< a layer's launches over cached experts whose slots lie one behind the other
    std::vector<int32_t> chan_;    ///< a layer's cached experts in the order of their slots
    std::vector<uint8_t> sent_;
    // A call that copies only the experts it routes to leaves some of those outside VRAM to the CPU (moe_rows()).
    std::vector<uint8_t> on_cpu_;  ///< per expert of the layer: the CPU's
    std::vector<int32_t> spare_;   ///< the layer's routed experts outside VRAM, the fewest tokens' first
    std::vector<CpuExpert> host_;  ///< their uses, by token
    double share_[5] = {0.25, 0.25, 0.25, 0.25, 0.25};   ///< the CPU's share of spare_, by the call's tokens:
                                                         ///< under 64, from 64, 128, 256 and 512
    std::unique_ptr<Workers> stagers_;   ///< fill the staging buffers meanwhile

    // The cache follows what is written (follow()).  It is filled for the average answer; while one is written,
    // the experts it keeps routing to and the cache lacks are copied over the cached ones it uses least, by a
    // thread and a stream of their own.
    struct Swap;
    struct Pick {
        int il = -1, e = 0;
        float recent = 0;
        int64_t prior = 0;
    };
    std::unique_ptr<Swap[]> swap_;   ///< the experts on their way, opt_.expert_swaps at most
    int n_swap_ = 0, n_trade_ = 0;
    size_t slot_max_ = 0;            ///< the largest expert of the cache
    void* swap_st_ = nullptr;        ///< the stream of their uploads
    std::thread follower_;
    static constexpr int kSwapPins = 2;
    uint8_t* swap_pin_[kSwapPins] = {};   ///< the follower's pinned buffers (follow_loop())
    void* swap_pin_ev_[kSwapPins] = {};   ///< each one's last upload
    struct Refill {
        int il, e;
        uint8_t* slot;
    };
    std::vector<Refill> refill_;   ///< empty slots of the cache the follower is to fill (grow_cache(later))
    /// the dense tensors' bytes in the mapped files, which leave the working set once they are on the card (load())
    std::vector<std::pair<const uint8_t*, size_t>> uploaded_;
    /// with Options::ram_experts_mib: the CPU's experts read through the mapping, whose pages stay in the working set
    /// (ram_keep(), at the load and after a long prompt); the others come from pool_
    std::vector<uint8_t> kept_;
    size_t outside_bytes_ = 0;   ///< the routed experts the card does not have, after load()
    void rekeep();
    struct FileView {
        const uint8_t* base;
        size_t bytes;
        std::string path;
    };
    std::vector<FileView> views_;   ///< the mapped files, for read_file()
    // With Options::ram_experts_mib, the experts outside kept_ that an answer gives the CPU: read from the files into
    // slots of this memory (gate | up | down, as a card's slot).  Through the mapping they stayed in the working set
    // until trim(), and a long answer brought nearly all back.  kept_ and the slots fill the budget from the load
    // on, so the working set stays where the load left it.
    std::unique_ptr<uint8_t[]> pool_;
    // The slots come in a size for each size of expert, side by side, each size with its share of the budget and a
    // clock of its own.  Every slot the size of the largest one, Qwen3.6's 6 GiB held 3263 of its experts where
    // their own sizes held 4625: one of its 41 layers has experts of 1.88 MiB, 37 of 1.30.
    struct PoolSize {
        size_t bytes;   ///< a slot's, in whole cache lines
        size_t at;      ///< where the first one starts in pool_
        size_t first;   ///< its number
        size_t n;
        size_t hand;    ///< pool_victim()'s, within these
        size_t nprot;   ///< of these, held (pool_prot_)
        size_t prot_max;
    };
    std::vector<PoolSize> pool_sizes_;
    std::vector<uint8_t> pool_size_of_;   ///< per layer: its experts' place in pool_sizes_
    size_t pool_bytes_ = 0;               ///< of Options::ram_experts_mib, what the slots take; kept_ has the rest
    uint8_t* pool_at(size_t v) const;
    size_t pool_slot_of(int il, const uint8_t* p) const;
    std::vector<uint8_t> pool_pin_;     ///< per slot: an upload to the card copies it (follow_loop())
    const uint8_t* pool_has(int il, int e) const;
    std::vector<int32_t> pool_of_;      ///< per (layer, expert): its slot, or -1
    std::vector<int64_t> pool_who_;     ///< per slot: layer * n_expert + expert, or -1
    std::vector<uint64_t> pool_tick_;   ///< per slot: the call that used it last
    std::vector<uint8_t> pool_ref_;     ///< per slot: used since pool_victim()'s hand came by
    // An expert read into a slot (missed, or read ahead) is on trial until a later call uses it, which holds it
    // (pool_prot_).  The clock takes the trial slots it comes to, and passes the held ones that were used since
    // it last came by; one that was not goes back on trial while more than prot_max of its size are held.  With every
    // slot alike, the reads ahead that went unused (a third of them) pushed out experts later tokens used.  One clock
    // whose reads ahead come in unmarked missed half as often with all the budget in slots, but the file cache kept a
    // second copy of what the slots held and pushed out what they did not: the laptop read 7.5 GiB from its disk in
    // an answer's 35 seconds, where kept_'s pages through the mapping are the file cache's own.
    std::vector<uint8_t> pool_prot_;
    size_t pool_victim(int il);
    uint64_t pool_now_ = 0;
    struct PoolRead {
        size_t e;
        int32_t s;
        bool ok;
    };
    std::vector<PoolRead> pool_reads_;   ///< pool_take()'s experts to read into their slots now
    void pool_take(const Layer& L, int il, CpuExpert* host, int n);
    void pool_read(const Layer& L, PoolRead& r);
    void pool_fill();
    int first_moe_layer() const;
    // The prompt's own uses of each expert since trim() (moe_rows()), and its rows: after a long one, the answer's
    // first call has the slots take the experts it used most (pool_reseed()).
    std::vector<float> prompt_use_;
    int64_t prompt_rows_ = 0;
    void pool_reseed();
    // Read ahead: moe_router() also scores the next MoE layer's router on this layer's input (on the card, copied to
    // h_ahead_ before ev_, a part of a window apiece, as ev_row_; a head recorded as a graph copies it too), and
    // moe() hands that layer's likeliest experts that the CPU would read from the files to pool_threads_ before
    // this layer's are computed.
    // A slot being read is pool_state_ 1 (2: the read failed); its layer's call waits for it.
    int prefetch_k_ = 0;   ///< the next MoE layer's guessed top k
    float* ahead_dev_ = nullptr;
    float* h_ahead_ = nullptr;
    std::unique_ptr<std::atomic<uint8_t>[]> pool_state_;
    struct PoolJob {   ///< a piece of an expert's read into its slot, or with s = kTouch / kDrop pages of the mapping
        uint8_t* dst;
        const uint8_t* src;
        size_t n;
        int32_t s;
    };
    static constexpr int32_t kTouch = -1, kDrop = -2;
    std::deque<PoolJob> pool_jobs_;
    /// kTouch / kDrop jobs, taken when pool_jobs_ is empty: in the first answer after the load the cache traded
    /// 2,700 experts, and their page work ahead of the reads held the reads a call waited for up
    std::deque<PoolJob> pool_pages_;
    std::unique_ptr<std::atomic<int>[]> pool_pending_;      ///< per slot: its pieces not read yet
    std::unique_ptr<std::atomic<uint8_t>[]> pool_failed_;   ///< per slot: a piece of it failed
    void pool_queue(const Layer& L, size_t e, int32_t s, bool first);
    std::mutex pool_m_;
    std::condition_variable pool_cv_;
    bool pool_quit_ = false;
    std::vector<std::thread> pool_threads_;
    void pool_ahead(int j, int n, size_t part, int k);
    void pool_loop();
    void pool_run(const PoolJob& job);
    bool pool_help();
    std::mutex swap_m_;
    std::condition_variable swap_cv_;
    bool swap_quit_ = false;
    std::string swap_error_;         ///< why the thread stopped; follow() throws it
    std::vector<float> recent_;      ///< per (layer, expert): its uses by written tokens, halved every kSwapHalf
    std::vector<uint8_t> coming_;    ///< per (layer, expert): on its way into the cache
    std::vector<Pick> in_, out_;     ///< per trade number: the experts to bring in, and the ones to give up
    std::vector<size_t> trade_at_;   ///< per trade number: how many of its pairs follow() has taken

    int64_t pos_ = 0;
    const int32_t* tok_ = nullptr;       ///< the tokens of the call under way, a row each as h_rl_ has them (route())
    std::vector<Image> images_;          ///< the sequence's pictures, by row
    uint64_t images_set_ = 0;            ///< counts images(): what was derived from an earlier list is stale
    std::vector<int64_t> use_, prior_;   ///< routed count per (layer, expert): this run, and the profile's
    Stats stats_;
};

}  // namespace strata::poly
