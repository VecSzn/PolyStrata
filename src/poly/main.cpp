// src/poly/main.cpp - strata-poly: GLM 5.3 Flash from its GGUF.
//
//   strata-poly --model M.gguf --tokens 1,2,3 [--max-new N]   one prompt; prints the generated ids
//   strata-poly --model M.gguf --serve                        the line protocol serve/server.py speaks
//
// The protocol is the one `strata --serve` speaks (src/program/generate.cpp): GEN, GENI for a prompt with pictures,
// STOP and QUIT in, and RESUME, PP, T, DONE and ERR out.
#include "conversations.hpp"
#include "engine.hpp"
#include "guesses.hpp"
#include "strata/core/conversation_memory.hpp"
#include "strata/spec/draft_policy.hpp"
#include "strata/spec/suffix_drafter.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <condition_variable>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <deque>
#include <iostream>
#include <mutex>
#include <random>
#include <stdexcept>
#include <string>
#include <thread>
#include <unordered_map>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#ifndef WIN32_LEAN_AND_MEAN
#define WIN32_LEAN_AND_MEAN
#endif
#include <windows.h>
#undef far   // a word of this file's
#endif

#ifndef STRATA_VERSION
#define STRATA_VERSION "dev"
#endif

namespace {

using strata::poly::CacheOptions;
using strata::poly::Conversations;
using strata::poly::Engine;
using strata::poly::Guesses;
using strata::poly::Options;
using Clock = std::chrono::steady_clock;

double ms_since(Clock::time_point t) { return std::chrono::duration<double, std::milli>(Clock::now() - t).count(); }

// serve() saves the experts' use after its first answer and then after an answer when the last save is this long
// ago, and not only when it is told to quit: a server that is ended on Windows takes its engine with it at once.
// A save is a file of some 40 KB and a rename.
constexpr double kProfileSaveMs = 10000.0;

// The most tokens of a window whose guesses are the block's: the token the model has and three guesses, each
// from the one before.  A repeat of the conversation may fill a longer window (Options::window).
constexpr int kBlockWindow = 4;

// A window longer than that is for a copy: a repeat of the conversation whose match is this many tokens or more.
// A repeat of a few tokens is in any text, and what followed it there is seldom what follows here for long.
constexpr int kCopyMatch = 8;

struct Sampling {
    float temperature = 0.f;   // 0: the most likely token
    float top_p = 1.f, min_p = 0.f;
    int top_k = 20;            // 0: no limit
    int penalty_last_n = 0;
    float penalty_repeat = 1.f, penalty_freq = 0.f, penalty_present = 0.f;
    unsigned long long seed = 0;
};

// llama.cpp's sampler chain in its default order: penalties, top-k, top-p, min-p, temperature, then the draw.
class Sampler {
public:
    int pick(const float* logits, int n_vocab, const Sampling& s, const std::vector<int32_t>& history,
             std::mt19937_64& rng) {
        if (penalizes(s)) {
            pen_.assign(logits, logits + n_vocab);
            penalize(s, history);
            logits = pen_.data();
        }
        if (s.temperature <= 0.f) return strata::poly::most_likely(logits, n_vocab);

        // The k best, the best first and of equals the lower id.  A token is a candidate only when it beats the
        // worst of the k so far, which is the heap's first; most of a vocabulary never does.
        auto by_logit = [](const Cand& a, const Cand& b) { return a.logit != b.logit ? a.logit > b.logit : a.id < b.id; };
        const size_t k = s.top_k > 0 ? std::min((size_t) s.top_k, (size_t) n_vocab) : (size_t) n_vocab;
        cand_.clear();
        for (int i = 0; i < n_vocab; ++i) {
            if (cand_.size() == k) {
                if (!(logits[i] > cand_.front().logit)) continue;   // an equal with a higher id is no better
                std::pop_heap(cand_.begin(), cand_.end(), by_logit);
                cand_.back() = {logits[i], i};
            } else {
                cand_.push_back({logits[i], i});
            }
            std::push_heap(cand_.begin(), cand_.end(), by_logit);
        }
        std::sort_heap(cand_.begin(), cand_.end(), by_logit);

        if (s.top_p < 1.f) {
            softmax();
            double cum = 0.0;
            size_t keep = cand_.size();
            for (size_t i = 0; i < cand_.size(); ++i) {
                cum += p_[i];
                if (cum >= s.top_p) {
                    keep = i + 1;
                    break;
                }
            }
            cand_.resize(keep);
        }
        if (s.min_p > 0.f) {
            const float floor = cand_[0].logit + std::log(s.min_p);
            size_t keep = 1;
            while (keep < cand_.size() && cand_[keep].logit >= floor) ++keep;
            cand_.resize(keep);
        }
        for (Cand& c : cand_) c.logit /= s.temperature;
        softmax();
        return cand_[std::discrete_distribution<size_t>(p_.begin(), p_.end())(rng)].id;
    }

private:
    struct Cand {
        float logit;
        int id;
    };

    static bool penalizes(const Sampling& s) {
        return s.penalty_last_n != 0 &&
               (s.penalty_repeat != 1.f || s.penalty_freq != 0.f || s.penalty_present != 0.f);
    }

    // on pen_, the logits' copy
    void penalize(const Sampling& s, const std::vector<int32_t>& history) {
        const size_t n = s.penalty_last_n < 0 ? history.size() : std::min(history.size(), (size_t) s.penalty_last_n);
        seen_.clear();
        for (size_t i = history.size() - n; i < history.size(); ++i) ++seen_[history[i]];
        for (const auto& [id, count] : seen_) {
            float& l = pen_[(size_t) id];
            l = l <= 0.f ? l * s.penalty_repeat : l / s.penalty_repeat;
            l -= (float) count * s.penalty_freq + s.penalty_present;
        }
    }

    void softmax() {
        p_.resize(cand_.size());
        float mx = cand_[0].logit;
        for (const Cand& c : cand_) mx = std::max(mx, c.logit);
        double sum = 0.0;
        for (size_t i = 0; i < cand_.size(); ++i) sum += p_[i] = std::exp((double) (cand_[i].logit - mx));
        for (double& p : p_) p /= sum;
    }

    std::vector<Cand> cand_;
    std::vector<float> pen_;
    std::vector<double> p_;
    std::unordered_map<int32_t, int> seen_;
};

bool parse_ids(const char* s, std::vector<int32_t>& out) {
    out.clear();
    while (*s == ' ') ++s;
    while (*s != '\0') {
        char* end = nullptr;
        const long long v = std::strtoll(s, &end, 10);
        if (end == s || v < 0 || v > INT32_MAX) return false;
        out.push_back((int32_t) v);
        s = end;
        while (*s == ' ') ++s;
        if (*s == ',') ++s;
        else if (*s != '\0') return false;
    }
    return !out.empty();
}

uint64_t fnv1a(const void* p, size_t n, uint64_t h = 1469598103934665603ull) {
    const uint8_t* b = (const uint8_t*) p;
    for (size_t i = 0; i < n; ++i) h = (h ^ b[i]) * 1099511628211ull;
    return h;
}

// A request's pictures: strata-vision's records (tools/vision/strata_vision.cpp), one after the other in the
// order of the prompt.
struct Pictures {
    std::vector<float> rows;
    std::vector<strata::poly::Image> at;
};

// Reads a request's pictures and finds their places: each takes the next run of `pad` in ids, a token a row.
// Those ids then become numbers below zero made from the picture, so that the conversation cache tells a picture
// from another one of the same size.  Returns what is wrong, or nothing.
std::string take_pictures(const std::string& path, int32_t pad, int n_embd, std::vector<int32_t>& ids, Pictures& out) {
    struct Record {
        int n, nx;
        size_t off;
    };
    std::vector<Record> records;
    out = Pictures{};
    std::FILE* f = std::fopen(path.c_str(), "rb");
    if (!f) return "cannot open " + path;
    std::string err;
    for (;;) {
        int32_t hdr[5];   // 'SVE1', rows, rows of a line, lines, a row's floats
        const size_t got = std::fread(hdr, sizeof(int32_t), 5, f);
        if (got == 0) break;
        if (got != 5 || hdr[0] != 0x31455653 || hdr[1] < 1 || hdr[1] > 1 << 20 || hdr[2] < 1 || hdr[3] < 1 ||
            (int64_t) hdr[2] * hdr[3] != hdr[1] || hdr[4] != n_embd) {
            err = "bad embeddings file (expected strata-vision records of width " + std::to_string(n_embd) + ")";
            break;
        }
        const size_t off = out.rows.size(), count = (size_t) hdr[1] * (size_t) n_embd;
        out.rows.resize(off + count);
        if (std::fread(out.rows.data() + off, sizeof(float), count, f) != count) {
            err = "short embeddings file";
            break;
        }
        records.push_back({hdr[1], hdr[2], off});
    }
    std::fclose(f);
    if (!err.empty()) return err;

    size_t i = 0;
    for (size_t k = 0; k < records.size(); ++k) {
        const Record& r = records[k];
        while (i < ids.size() && ids[i] != pad) ++i;
        for (int j = 0; j < r.n; ++j)
            if (i + (size_t) j >= ids.size() || ids[i + (size_t) j] != pad)
                return "picture " + std::to_string(k + 1) + " has " + std::to_string(r.n) +
                       " rows and the prompt fewer places for it";
        const float* rows = out.rows.data() + r.off;
        const int32_t grid[2] = {r.n, r.nx};
        const uint64_t h = fnv1a(rows, (size_t) r.n * (size_t) n_embd * sizeof(float), fnv1a(grid, sizeof grid));
        for (int j = 0; j < r.n; ++j)
            ids[i + (size_t) j] = -1 - (int32_t) ((h + (uint64_t) j * 0x9E3779B97F4A7C15ull) >> 33);
        out.at.push_back({(int64_t) i, r.n, r.nx, rows});
        i += (size_t) r.n;
    }
    if (std::find(ids.begin() + (std::ptrdiff_t) i, ids.end(), pad) != ids.end())
        return "the prompt has more pictures than the embeddings file";
    if (!ids.empty() && ids.back() < 0) return "a prompt cannot end in a picture";
    return {};
}

// stdin on its own thread, so a STOP is seen while a request runs
class Lines {
public:
    Lines() : th_([this] { read(); }) { th_.detach(); }

    std::string pop() {
        std::unique_lock<std::mutex> l(m_);
        cv_.wait(l, [this] { return !q_.empty(); });
        std::string s = std::move(q_.front());
        q_.pop_front();
        return s;
    }

    std::atomic<bool> stop{false};

private:
    void read() {
        std::string l;
        while (std::getline(std::cin, l)) {
            if (!l.empty() && l.back() == '\r') l.pop_back();
            if (l == "STOP") stop.store(true);
            else push(l);
        }
        push("QUIT");
    }
    void push(std::string s) {
        { std::lock_guard<std::mutex> l(m_); q_.push_back(std::move(s)); }
        cv_.notify_one();
    }

    std::mutex m_;
    std::condition_variable cv_;
    std::deque<std::string> q_;
    std::thread th_;
};

// What the conversation cache is told on the command line.
struct CacheFlags {
    CacheOptions o;
    long long min_free_mib = 2560;   // parking leaves this much physical RAM free
};

struct Session {
    Engine& engine;
    std::vector<int32_t> eos;
    CacheFlags flags;
    Conversations<Engine> kept;   // the conversation in the engine, and the parked ones
    Sampler sampler;
    std::mt19937_64 rng{std::random_device{}()};
    const std::vector<int32_t>* follow = nullptr;   // --follow: the answer's tokens, taken in place of the picks
    const float* picked_from = nullptr;             // the logits of the last token picked
    long long asked = 0, reused = 0;   // prompt tokens of every request, and those of them not read again
    // Guesses at the tokens to come, checked a window at a time (Engine::window).  They come from the block some
    // files carry for it (Engine::draft), and from the conversation: when the last tokens repeat an earlier stretch
    // of it, what followed there is the guess.  The policy learns from the windows so far, their hits and their
    // time, which of the two pays.  It and the repeats are the other engine's (strata/spec).
    int most;                             // the tokens of a window at most; 1: no guesses
    bool look;                            // guess from repeats
    strata::spec::SuffixDrafter repeats;
    strata::spec::DraftPolicy policy;
    Guesses guesses;                      // whether the block is asked in a round, and for how many (guesses.hpp)

    Session(Engine& e, std::vector<int32_t> eos_ids, const CacheFlags& f, int lookup)
        : engine(e), eos(std::move(eos_ids)), flags(f), kept(e, with_floor(f)),
          most(lookup > 0 || e.drafts() ? std::min<int>(e.max_window(), strata::spec::DraftPolicy::kMaxT) : 1),
          look(lookup > 0 && most > 1),
          repeats(std::max(lookup, 3), 64, (size_t) e.max_context() + 4096), policy(most) {}

    static CacheOptions with_floor(const CacheFlags& f) {
        CacheOptions o = f.o;
        o.log = stderr;
        o.admit = [floor = (uint64_t) f.min_free_mib << 20](size_t bytes) {
            return strata::core::conversation_memory_admit(strata::core::available_host_bytes(), bytes, floor);
        };
        return o;
    }

    bool is_eos(int32_t t) const { return std::find(eos.begin(), eos.end(), t) != eos.end(); }

    struct Result {
        int64_t produced = 0, resume = 0, read = 0;
        int64_t guessed = 0, guessed_right = 0;
        double prompt_ms = 0, decode_ms = 0;
        const char* finish = "length";
    };

    // Reads the prompt and generates.  on_token gets every token, the end-of-sequence one included.  The rows of
    // the prompt's pictures have ids below zero (take_pictures()); `pad` is the token they stand in for.
    template<class OnToken, class OnProgress>
    Result generate(const std::vector<int32_t>& ids, const Pictures& pictures, int32_t pad, int64_t turn_at,
                    int64_t max_new, const Sampling& s, const std::atomic<bool>& stop, OnToken on_token,
                    OnProgress on_progress) {
        Result r;
        const int64_t n = (int64_t) ids.size();
        Clock::time_point t0 = Clock::now(), shown = t0;   // bringing a parked conversation back is prompt time
        engine.images(pictures.at.data(), (int) pictures.at.size());
        r.resume = kept.resume(ids);
        if (const size_t freed = engine.trim_kv(n)) {
            const int had = engine.cached_experts();
            engine.mind_vram();
            std::fprintf(stderr, "strata-poly: the attention caches gave back %.2f GiB that this request and the one "
                                 "before do not reach; the expert cache %d -> %d experts\n",
                         double(freed) / double(1ull << 30), had, engine.cached_experts());
        }
        kept.turn_at(turn_at);
        asked += n;
        reused += r.resume;
        on_progress(r.resume, n, 0.0, true);
        if (s.seed != 0) rng.seed(s.seed);

        // The last token goes in alone, for the logits that come with it.  The tokens before it go in as many at
        // a time as the engine takes.  A prompt without pictures goes in several calls' worth at once, which the
        // engine reads faster than the calls (read()), and the checkpoints due inside are the engine's to write.
        // One with pictures goes in a call at a time up to the next checkpoint, a picture whole.
        const bool text = pictures.at.empty();
        std::vector<std::pair<int64_t, uint8_t*>> due;
        std::vector<Engine::Mark> marks;
        for (int64_t i = r.resume; i < n;) {
            if (stop.load()) {
                r.finish = "cancel";
                break;
            }
            const bool last = i == n - 1;
            const int64_t room = std::min<int64_t>(n - 1 - i, text ? engine.max_read() : engine.max_rows());
            const int64_t m = last ? 1 : text ? room : engine.whole(i, kept.until_checkpoint(i, room));
            kept.before(i, last);
            if (text && !last) {
                kept.inside(i, m, due);
                marks.clear();
                for (const auto& d : due) marks.push_back({(int) d.first, d.second});
                // A long prompt is one read.  Between its layers the read says how far it is, and a request that
                // was stopped gives it up there: the conversation is then at its last checkpoint.
                const std::function<bool(int, int)> turn = [&](int done, int of) {
                    if (stop.load()) return false;
                    if (ms_since(shown) >= 500.0) {
                        on_progress(i + m * done / of, n, ms_since(t0), false);
                        shown = Clock::now();
                    }
                    return true;
                };
                if (!engine.read(ids.data() + i, (int) m, marks.data(), (int) marks.size(), &turn)) {
                    kept.rewind();
                    r.finish = "cancel";
                    break;
                }
            } else {
                engine.eval(ids.data() + i, (int) m, last);
            }
            kept.read(ids.data() + i, m);
            i += m;
            r.read += m;
            if (ms_since(shown) >= 500.0 || last) {
                on_progress(i, n, ms_since(t0), false);
                shown = Clock::now();
            }
        }
        kept.prompt_was(n);
        r.prompt_ms = ms_since(t0);
        if (std::strcmp(r.finish, "cancel") == 0) return r;

        t0 = Clock::now();
        std::vector<int32_t> history = kept.seq();
        std::replace_if(history.begin(), history.end(), [](int32_t t) { return t < 0; }, pad);
        if (look) {
            repeats.reset();
            repeats.append(history.data(), history.size());
        }
        const auto pick = [&](const float* logits) {
            picked_from = logits;
            if (follow && r.produced < (int64_t) follow->size()) return (*follow)[(size_t) r.produced];
            return (int32_t) sampler.pick(logits, engine.n_vocab(), s, history, rng);
        };
        // Shows a token.  False when the answer ends with it.
        const auto show = [&](int32_t t) {
            on_token(t);
            ++r.produced;
            history.push_back(t);
            if (look) repeats.append(t);
            if (is_eos(t)) r.finish = "stop";
            else if (stop.load()) r.finish = "cancel";
            else return r.produced < max_new;
            return false;
        };
        if (max_new > 0) {
            int32_t w[strata::spec::DraftPolicy::kMaxT], d[strata::spec::DraftPolicy::kMaxT];
            int32_t t = pick(engine.logits());   // picked, and not shown yet
            while (show(t)) {
                // t goes into the model now, and with it guesses at what follows when there are any that pay
                const Clock::time_point round = Clock::now();
                const bool room = engine.pos() + most <= engine.max_context();
                const int offered = look && room ? repeats.propose(most - 1, w + 1) : 0;
                const int match = repeats.last_match();
                const int widest = match >= kCopyMatch ? most : std::min(most, kBlockWindow);   // this round's window
                const int k = std::min(offered, widest - 1);
                // A repeat that fills the window, from a match of a length that has been right nine times in ten,
                // is taken as it is: the block would mostly say the same, and asking it takes time.  Otherwise the
                // block guesses, each guess from the one before, while a guess more pays (Guesses).
                const bool blocks = engine.drafts() && room && !(k == widest - 1 && policy.lookup_rate(match) >= 0.9);
                const bool ask = guesses.begin(blocks, policy);
                // the round's time, as the policy counts it: a window of `in` tokens wrote n of them
                const auto count = [&](int in, int n) {
                    return guesses.end(in, n, ms_since(round), engine.took_record());
                };
                strata::spec::DraftPolicy::Pick use = policy.choose(1, k, match);
                if (ask) {
                    int n_d = 0;
                    for (int32_t from = t; n_d < std::min(most, kBlockWindow) - 1 && guesses.wants(n_d); from = d[n_d++]) {
                        float p = 0.f;
                        const Clock::time_point asked_at = Clock::now();
                        d[n_d] = engine.draft(from, n_d, &p);
                        if (!guesses.takes(n_d, p, ms_since(asked_at))) break;
                    }
                    use = policy.choose(1 + n_d, k, match);
                }
                if (use.t == 1) {
                    engine.eval(&t, 1, true);
                    kept.read(&t, 1);
                    t = pick(engine.logits());
                    policy.observe(false, 1, 0, 0, count(1, 1));
                    continue;
                }
                w[0] = t;
                if (!use.lookup) std::copy(d, d + use.t - 1, w + 1);
                engine.window(w, use.t);
                // logits_of(i) is what follows w[i].  A guess that is what was picked has been through the model
                // already; the token an answer ends with is shown and never fed, as without guesses.
                int right = 0;
                bool more = true;
                t = pick(engine.logits_of(0));
                while (right < use.t - 1 && t == w[right + 1] && (more = show(t))) t = pick(engine.logits_of(++right));
                engine.keep(right + 1);
                kept.read(w, right + 1);
                r.guessed += use.t - 1;
                r.guessed_right += right + (more ? 0 : 1);
                policy.observe(use.lookup, use.t, right, match, count(use.t, right + 1));
                if (!more) break;
            }
        }
        r.decode_ms = ms_since(t0);
        return r;
    }
};

// GEN <max_new> [key=value ...] <id,id,...>, or GENI with the file of the prompt's pictures ahead of the ids and
// the key image_pad, the token whose runs in the ids are the pictures' places.  turn_at is the place in the ids
// where the prompt's last turn begins (the server knows the template's turn tokens): a checkpoint goes there.
bool parse_gen(const std::string& line, long long& max_new, Sampling& s, int32_t& pad, int64_t& turn_at,
               std::string& pictures, std::vector<int32_t>& ids) {
    const bool geni = line.rfind("GENI ", 0) == 0;
    char* p = nullptr;
    max_new = std::strtoll(line.c_str() + (geni ? 5 : 4), &p, 10);
    if (max_new < 1) return false;
    pad = -1;
    turn_at = -1;
    for (;;) {   // key=value tokens, then the file and the ids
        while (*p == ' ') ++p;
        const char* start = p;
        while (*p != '\0' && *p != ' ') ++p;
        const std::string tok(start, (size_t) (p - start));
        const size_t eq = tok.find('=');
        if (eq == std::string::npos) {
            p = const_cast<char*>(start);
            break;
        }
        const std::string key = tok.substr(0, eq);
        const char* v = tok.c_str() + eq + 1;
        if (key == "temperature") s.temperature = std::strtof(v, nullptr);
        else if (key == "top_p") s.top_p = std::strtof(v, nullptr);
        else if (key == "top_k") s.top_k = std::atoi(v);
        else if (key == "min_p") s.min_p = std::strtof(v, nullptr);
        else if (key == "penalty_last_n") s.penalty_last_n = std::atoi(v);
        else if (key == "penalty_repeat") s.penalty_repeat = std::strtof(v, nullptr);
        else if (key == "penalty_freq") s.penalty_freq = std::strtof(v, nullptr);
        else if (key == "penalty_present") s.penalty_present = std::strtof(v, nullptr);
        else if (key == "seed") s.seed = std::strtoull(v, nullptr, 10);
        else if (key == "image_pad") pad = (int32_t) std::atoi(v);
        else if (key == "turn_at") turn_at = std::strtoll(v, nullptr, 10);
        // other keys belong to the other engine and are skipped
    }
    pictures.clear();
    if (geni) {
        const char* end = std::strchr(p, ' ');
        if (!end || end == p || pad < 0) return false;
        pictures.assign(p, (size_t) (end - p));
        p = const_cast<char*>(end);
    }
    return parse_ids(p, ids);
}

int serve(Session& ss) {
    Engine& engine = ss.engine;
    std::printf("INFO context=%d expert_slots=%d expert_cache_mib=%lld pool_workers=%d conversation_cache_mib=%lld "
                "conversation_cache_slots=%d conversation_cache_min_free_mib=%lld vram_free_mib=%lld vram_elastic=1 "
                "engine=" STRATA_VERSION "\n",
                engine.max_context(), engine.cached_experts(), (long long) (engine.cached_gib() * 1024.0),
                engine.threads(), (long long) ss.flags.o.park_mib, ss.flags.o.park_slots, ss.flags.min_free_mib,
                (long long) engine.vram_free_mib());
    std::printf("READY %d stop\n", engine.max_context());
    std::fflush(stdout);

    static Lines lines;   // its thread may still be in a read when this function returns
    Pictures pictures;    // the engine reads the last request's until the next one's are set
    // the experts' use (--expert-profile): as if it was last saved long ago, so that the first answer saves it
    Clock::time_point saved = Clock::now() - std::chrono::milliseconds((long long) kProfileSaveMs);
    for (;;) {
        const std::string line = lines.pop();
        if (line == "QUIT") break;
        if (line.empty()) continue;
        lines.stop.store(false);   // a STOP that came between requests is stale
        long long max_new = 0;
        Sampling s;
        std::vector<int32_t> ids;
        int32_t pad = -1;
        int64_t turn_at = -1;
        std::string file;
        // VRAM [reserve_mib], the server's POST /v1/vram: the expert cache sized again between requests, to leave
        // that much VRAM to other programs; alone, back to what the engine started with
        if (line.rfind("VRAM", 0) == 0 && (line.size() == 4 || line[4] == ' ')) {
            char* end = nullptr;
            const bool bare = line.find_first_not_of(' ', 4) == std::string::npos;
            const long long reserve = bare ? -1 : std::strtoll(line.c_str() + 4, &end, 10);
            if (!bare && (end == line.c_str() + 4 || reserve < 0 || reserve > (1LL << 24))) {
                std::printf("ERR expected: VRAM [reserve_mib]\n");
                std::fflush(stdout);
                continue;
            }
            try {
                const Clock::time_point t0 = Clock::now();
                const int before = engine.cached_experts();
                engine.resize_cache((int) reserve);
                const long long free_mib = (long long) engine.vram_free_mib();
                std::fprintf(stderr, "strata-poly: VRAM %d MiB kept free: the expert cache %d -> %d of %d experts "
                                     "(%.2f of %.2f GiB), %lld MiB free now, in %.0f ms\n",
                             engine.reserve_mib(), before, engine.cached_experts(), engine.full_experts(),
                             engine.cached_gib(), engine.full_gib(), free_mib, ms_since(t0));
                std::printf("VRAM reserve_mib=%d expert_slots=%d expert_slots_full=%d expert_cache_mib=%lld "
                            "expert_cache_full_mib=%lld vram_free_mib=%lld prompt_chunk=%d\n",
                            engine.reserve_mib(), engine.cached_experts(), engine.full_experts(),
                            (long long) (engine.cached_gib() * 1024.0), (long long) (engine.full_gib() * 1024.0),
                            free_mib, engine.max_rows());
            } catch (const std::exception& e) {
                std::printf("ERR VRAM: %s\n", e.what());
            }
            std::fflush(stdout);
            continue;
        }
        if (line.rfind("GEN ", 0) != 0 && line.rfind("GENI ", 0) != 0) {
            std::printf("ERR expected: GEN <max_new> <id,id,...> or GENI <max_new> image_pad=<id> <file> <id,id,...>\n");
            std::fflush(stdout);
            continue;
        }
        if (!parse_gen(line, max_new, s, pad, turn_at, file, ids)) {
            std::printf("ERR bad request\n");
            std::fflush(stdout);
            continue;
        }
        const long long n = (long long) ids.size();
        if (n + max_new > engine.max_context()) {
            std::printf("ERR prompt (%lld tokens) + max_new (%lld) exceeds the context (%d)\n", n, max_new,
                        engine.max_context());
            std::fflush(stdout);
            continue;
        }
        if (std::any_of(ids.begin(), ids.end(), [&](int32_t t) { return t >= engine.n_vocab(); })) {
            std::printf("ERR a token id is outside the vocabulary\n");
            std::fflush(stdout);
            continue;
        }
        std::string wrong;
        if (file.empty()) pictures = Pictures{};
        else if (!engine.sees()) wrong = "this model does not read pictures";
        else wrong = take_pictures(file, pad, engine.n_embd(), ids, pictures);
        if (!wrong.empty()) {
            std::printf("ERR %s\n", wrong.c_str());
            std::fflush(stdout);
            continue;
        }

        const strata::poly::Stats before = engine.stats();
        int64_t resume = 0;
        Session::Result r;
        try {
            const int had = engine.cached_experts();
            if (engine.mind_vram())
                std::fprintf(stderr, "strata-poly: the expert cache %d -> %d experts (%.2f GiB) for the VRAM free "
                                     "now, %lld MiB left\n",
                             had, engine.cached_experts(), engine.cached_gib(), (long long) engine.vram_free_mib());
            r = ss.generate(
                ids, pictures, pad, turn_at, max_new, s, lines.stop,
                [](int32_t t) {
                    std::printf("T %d\n", (int) t);
                    std::fflush(stdout);
                },
                [&](int64_t at, int64_t total, double ms, bool first) {
                    if (first) {
                        resume = at;
                        std::printf("RESUME %lld\n", (long long) at);
                    } else {
                        std::printf("PP %lld %lld %.0f %.1f\n", (long long) at, (long long) total, ms,
                                    ms > 0.0 ? 1000.0 * (double) (at - resume) / ms : 0.0);
                    }
                    std::fflush(stdout);
                });
        } catch (const std::exception& e) {
            // the engine's state is unknown now: the conversation it was reading is lost
            ss.kept.drop();
            try { engine.reset(); } catch (const std::exception&) {}
            std::printf("ERR %s\n", e.what());
            std::fflush(stdout);
            continue;
        }
        // DONE <generated> <prompt> <prompt ms> <decode ms> <finish> <drafts accepted> <drafts offered> <reused>
        //      <experts from VRAM> <experts> <RAM blobs> <file blobs> <file MB> <prompt tokens read> <offloaded>
        const strata::poly::Stats& now = engine.stats();
        // the cache's share of the experts looked up while answering: a prompt batch's go through VRAM either way
        const long long hits = now.gpu_experts - before.gpu_experts;
        const long long misses = now.cpu_experts - before.cpu_experts;
        std::printf("DONE %lld %lld %.1f %.1f %s %lld %lld %lld %lld %lld 0 0 0.0 %lld 0\n", (long long) r.produced, n,
                    r.prompt_ms, r.decode_ms, r.finish, (long long) r.guessed_right, (long long) r.guessed,
                    (long long) r.resume, hits, hits + misses, (long long) r.read);
        std::fflush(stdout);
        if (ms_since(saved) >= kProfileSaveMs) {
            try {
                engine.save_profile();
            } catch (const std::exception& e) {   // the next start's cache is the worse for it, this run is not
                std::fprintf(stderr, "strata-poly: %s\n", e.what());
            }
            saved = Clock::now();
        }
    }
    std::fprintf(stderr, "strata-poly: %lld of %lld prompt tokens were not read again (%.1f%%); %zu conversations "
                         "parked in %zu MiB, %zu pushed out; %lld MiB of VRAM free at the end\n",
                 ss.reused, ss.asked, ss.asked > 0 ? 100.0 * (double) ss.reused / (double) ss.asked : 0.0,
                 ss.kept.parked(), ss.kept.parked_bytes() >> 20, ss.kept.evictions(), (long long) engine.vram_free_mib());
    return 0;
}

// The n most likely next tokens with their probabilities, to hold against another engine's on the same prompt.
void print_top(const float* logits, int n_vocab, int n) {
    std::vector<int> order((size_t) n_vocab);
    for (int i = 0; i < n_vocab; ++i) order[(size_t) i] = i;
    n = std::min(n, n_vocab);
    std::partial_sort(order.begin(), order.begin() + n, order.end(),
                      [logits](int a, int b) { return logits[a] > logits[b]; });
    const float top = logits[order[0]];
    double sum = 0.0;
    for (int i = 0; i < n_vocab; ++i) sum += std::exp((double) (logits[i] - top));
    std::fprintf(stderr, "top");
    for (int i = 0; i < n; ++i)
        std::fprintf(stderr, " %d:%.6g", order[(size_t) i], std::exp((double) (logits[order[(size_t) i]] - top)) / sum);
    std::fprintf(stderr, "\n");
}

int once(Session& ss, const std::vector<int32_t>& ids, const Pictures& pictures, int32_t pad, long long max_new,
         const Sampling& s, int top_probs) {
    std::atomic<bool> stop{false};
    std::string out;
    const Session::Result r = ss.generate(
        ids, pictures, pad, -1, max_new, s, stop,
        [&](int32_t t) {
            if (top_probs > 0) print_top(ss.picked_from, ss.engine.n_vocab(), top_probs);
            out += (out.empty() ? "" : ",") + std::to_string(t);
        },
        [](int64_t, int64_t, double, bool) {});
    std::printf("%s\n", out.c_str());
    const strata::poly::Stats& st = ss.engine.stats();
    const double experts = (double) std::max<int64_t>(st.gpu_experts + st.cpu_experts, 1);
    std::fprintf(stderr,
                 "prompt %lld tokens %.1f tok/s, generated %lld tokens %.2f tok/s (%s); experts from VRAM %.1f%%, "
                 "CPU experts %.1f ms/token, router wait %.1f ms/token, %lld experts swapped into VRAM, "
                 "%lld left out, %lld of %lld guessed tokens right\n",
                 (long long) r.read, r.prompt_ms > 0 ? 1000.0 * (double) r.read / r.prompt_ms : 0.0,
                 (long long) r.produced, r.decode_ms > 0 ? 1000.0 * (double) (r.produced - 1) / r.decode_ms : 0.0,
                 r.finish, 100.0 * (double) st.gpu_experts / experts, st.cpu_ms / (double) std::max<int64_t>(st.tokens, 1),
                 st.route_wait_ms / (double) std::max<int64_t>(st.tokens, 1), (long long) st.swapped_experts,
                 (long long) st.tail_experts, (long long) r.guessed_right, (long long) r.guessed);
    return 0;
}

void usage() {
    std::fprintf(stderr,
                 "usage: strata-poly --model FILE.gguf (--serve | --tokens ID,ID,... [--max-new N])\n"
                 "  --max-context N        tokens the attention caches hold (default 8192)\n"
                 "  --threads N            CPU threads for the experts that are not in VRAM (default: all usable)\n"
                 "  --vram-reserve-mib N   VRAM left free beside the expert cache, 256 at least (default: a\n"
                 "                         thirty-second of the card, 256 to 768, or a sixteenth, 512 to 1024,\n"
                 "                         when the PC's free RAM is within 4 GiB of the model's files, and then\n"
                 "                         kept whole; else half of it after the first request)\n"
                 "  --layer-split S        the model's layers across the CUDA devices this process sees, in their\n"
                 "                         order: S is auto (by each card's free VRAM) or the first layer of each\n"
                 "                         later card, as 20 or 20,30 (default: the first card alone)\n"
                 "  --expert-profile FILE  expert use counts: read to choose the cached experts, written back on\n"
                 "                         exit and by --serve after an answer, every 10 s at most\n"
                 "  --no-gpu-experts       keep every routed expert on the CPU\n"
                 "  --any-node             on a PC with several NUMA nodes, let the CPU threads run on any of them\n"
                 "                         (default: on the node whose memory holds most of the experts)\n"
                 "  --window N             the most tokens that go through the model at once while an answer is\n"
                 "                         written: the one that is known and guesses at the ones after it, which\n"
                 "                         count when they turn out right (1 = no guesses; default 8, of which\n"
                 "                         the block guesses 3, and 4 for GLM 5.3 Flash)\n"
                 "  --lookup N             guess what followed an earlier repeat of the last N or more tokens\n"
                 "                         (default 3, 0 = never)\n"
                 "  --no-draft             do not run the block that guesses ahead\n"
                 "  --draft-model FILE     a file of the same model to take that block from, when --model has none;\n"
                 "                         for Gemma 4, its draft model (a gemma4-assistant file)\n"
                 "  --draft-vocab N        the block guesses among the vocabulary's first N tokens (default:\n"
                 "                         131072 for Qwen3.6, Ornith and Tiel, all of it for the others; 0 = all)\n"
                 "  --no-graphs            launch a token's kernels one by one (default: a layer's are recorded once\n"
                 "                         as a CUDA graph and replayed)\n"
                 "  --kv-whole             give the attention caches VRAM for the whole context at load (default: as\n"
                 "                         the conversation grows, the expert cache having the rest until then)\n"
                 "  --expert-swaps N       while an answer is written, up to N experts it keeps using are on their way\n"
                 "                         into VRAM, in place of cached ones it uses less (default 64, 0 = the cache\n"
                 "                         stays as it was loaded)\n"
                 "  --prompt-batch N       prompt tokens read at a time (default 1024); about 1.4 MiB of VRAM each\n"
                 "  --stage-mib N          VRAM the experts outside the cache pass through in a prompt batch (default\n"
                 "                         2048, or what a layer's experts take when that is less); with less, more\n"
                 "                         of them are copied after the layer has routed\n"
                 "  --stage-ahead N        from N tokens, a batch copies those experts before it routes (default 1024,\n"
                 "                         0 = never)\n"
                 "  --cpu-share F          under that, the share of those experts the CPU computes meanwhile, the\n"
                 "                         ones the fewest tokens chose (default: as the times say; 0 = none)\n"
                 "  --prompt-cache N       checkpoints kept of a conversation, about 150 MB of RAM each (default 6;\n"
                 "                         0 = a prompt is read from its start unless it goes on from the last token)\n"
                 "  --prompt-cache-every N a checkpoint every N prompt tokens, besides the one at a prompt's end\n"
                 "                         (default 2048, 0 = none)\n"
                 "  --conversation-cache-mib N   RAM for parked conversations, so that several can take turns\n"
                 "                         without being read again (default 0 = off)\n"
                 "  --conversation-cache-slots N        parked conversations at most (default 4)\n"
                 "  --conversation-cache-min-tokens N   a conversation shorter than this is not parked (default 1024)\n"
                 "  --conversation-cache-min-free-mib N park only while this much physical RAM stays free (default\n"
                 "                         2560)\n"
                 "  --eos-ids ID,ID,...    tokens that end a reply (default: the GGUF's eos, eot and eom)\n"
                 "  --temperature F --top-k N --top-p F --min-p F --seed N   sampling for --tokens (default greedy)\n"
                 "  --images FILE          with --tokens: the prompt's pictures, as strata-vision writes them\n"
                 "  --image-pad ID         the token whose runs in --tokens are the pictures' places, a token a row\n"
                 "  --dump-layers          print a checksum of every layer's stages to stderr\n"
                 "  --top-probs N          with --tokens: print the N most likely ids of every step to stderr\n"
                 "  --follow ID,ID,...     with --tokens: write these tokens in place of the ones picked, for\n"
                 "                         --top-probs of the same answer under other settings\n"
                 "  --tail-skip N          an expert outside VRAM that every token of a window ranks at place N+1\n"
                 "                         or later among its experts is left out: faster, and the answers move a\n"
                 "                         little (default 0 = none is; setup gives Qwen3.6 6)\n"
                 "environment: STRATA_ROUTE_TAIL_SKIP=N  in place of --tail-skip (0 turns it off)\n");
}

}  // namespace

// Windows holds the threads of a process it takes for background work to the efficiency cores (power throttling):
// on a laptop with both kinds an answer after a long prompt came at 39 tokens/s where 58.  serve/winjob.py takes the
// engines the server starts out of it; this does the same for an engine anything else starts.
void no_power_throttling() {
#ifdef _WIN32
    for (const ULONG speed_and_timer : {0x1ul | 0x4ul, 0x1ul}) {   // Windows 10 knows the first bit only
        PROCESS_POWER_THROTTLING_STATE off{PROCESS_POWER_THROTTLING_CURRENT_VERSION, speed_and_timer, 0};
        if (SetProcessInformation(GetCurrentProcess(), ProcessPowerThrottling, &off, sizeof(off))) return;
    }
#endif
}

int main(int argc, char** argv) {
    no_power_throttling();
    Options o;
    CacheFlags cache;
    Sampling s;
    bool want_serve = false;
    long long max_new = 32;
    int top_probs = 0, lookup = 3, image_pad = -1;
    std::string images;
    std::vector<int32_t> ids, eos, follow;
    for (int i = 1; i < argc; ++i) {
        const std::string a = argv[i];
        auto val = [&]() -> const char* {
            if (i + 1 >= argc) {
                std::fprintf(stderr, "strata-poly: %s needs a value\n", a.c_str());
                std::exit(2);
            }
            return argv[++i];
        };
        if (a == "--model") o.model = val();
        else if (a == "--serve") want_serve = true;
        else if (a == "--max-context") o.max_context = std::atoi(val());
        else if (a == "--threads") o.threads = std::atoi(val());
        else if (a == "--vram-reserve-mib") o.vram_reserve_mib = std::atoi(val());
        else if (a == "--expert-profile") o.profile = val();
        else if (a == "--no-gpu-experts") o.gpu_experts = false;
        else if (a == "--any-node") o.stay_near = false;
        else if (a == "--no-graphs") o.graphs = false;
        else if (a == "--kv-whole") o.kv_grow = false;
        else if (a == "--window") o.window = std::atoi(val());
        else if (a == "--lookup") lookup = std::atoi(val());
        else if (a == "--no-draft") o.draft = false;
        else if (a == "--draft-model") o.draft_model = val();
        else if (a == "--expert-swaps") o.expert_swaps = std::atoi(val());
        else if (a == "--tail-skip") o.tail_skip = std::atoi(val());
        else if (a == "--layer-split") o.layer_split = val();
        else if (a == "--draft-vocab") o.draft_vocab = std::atoi(val());
        else if (a == "--prompt-batch") o.prompt_batch = std::atoi(val());
        else if (a == "--stage-mib") o.stage_mib = std::atoi(val());
        else if (a == "--stage-ahead") o.stage_ahead = std::atoi(val());
        else if (a == "--cpu-share") o.cpu_share = (float) std::atof(val());
        else if (a == "--prompt-cache") cache.o.checkpoints = std::atoi(val());
        else if (a == "--prompt-cache-every") cache.o.every = std::atoll(val());
        else if (a == "--conversation-cache-mib") cache.o.park_mib = std::atoll(val());
        else if (a == "--conversation-cache-slots") cache.o.park_slots = std::atoi(val());
        else if (a == "--conversation-cache-min-tokens") cache.o.park_min = std::atoll(val());
        else if (a == "--conversation-cache-min-free-mib") cache.min_free_mib = std::atoll(val());
        else if (a == "--dump-layers") o.dump = true;
        else if (a == "--images") images = val();
        else if (a == "--image-pad") image_pad = std::atoi(val());
        else if (a == "--top-probs") top_probs = std::atoi(val());
        else if (a == "--max-new") max_new = std::atoll(val());
        else if (a == "--temperature") s.temperature = std::strtof(val(), nullptr);
        else if (a == "--top-k") s.top_k = std::atoi(val());
        else if (a == "--top-p") s.top_p = std::strtof(val(), nullptr);
        else if (a == "--min-p") s.min_p = std::strtof(val(), nullptr);
        else if (a == "--seed") s.seed = std::strtoull(val(), nullptr, 10);
        else if (a == "--tokens" || a == "--eos-ids" || a == "--follow") {
            if (!parse_ids(val(), a == "--tokens" ? ids : a == "--eos-ids" ? eos : follow)) {
                std::fprintf(stderr, "strata-poly: %s takes token ids separated by commas\n", a.c_str());
                return 2;
            }
        } else {
            if (a != "--help" && a != "-h") std::fprintf(stderr, "strata-poly: unknown option %s\n", a.c_str());
            usage();
            return 2;
        }
    }
    if (o.model.empty() || (!want_serve && ids.empty()) || max_new < 1) {
        usage();
        return 2;
    }
    // over --tail-skip, as Strata's engine reads it: =0 turns a model's default off
    if (const char* v = std::getenv("STRATA_ROUTE_TAIL_SKIP")) o.tail_skip = std::atoi(v);
    const long long most = 1LL << 24;   // MiB: 16 TiB
    if (cache.o.checkpoints < 0 || cache.o.every < 0 || cache.o.park_mib < 0 || cache.o.park_mib > most ||
        cache.o.park_slots < 0 || cache.o.park_slots > 1024 || cache.o.park_min < 0 || cache.min_free_mib < 0 ||
        cache.min_free_mib > most) {
        std::fprintf(stderr, "strata-poly: a cache option is negative or too large\n");
        return 2;
    }

    std::unique_ptr<Engine> loaded;
    try {
        const Clock::time_point t0 = Clock::now();
        loaded = Engine::open(o);
        Engine& engine = *loaded;
        std::fprintf(stderr, "strata-poly: loaded in %.1f s; %d of %d routed experts in VRAM (%.1f GiB), %d CPU threads\n",
                     ms_since(t0) / 1000.0, engine.cached_experts(), engine.total_experts(), engine.cached_gib(),
                     engine.threads());
        if (o.tail_skip > 0)
            std::fprintf(stderr, "strata-poly: tail skip %d: an expert outside VRAM that every token of a window ranks "
                                 "at place %d or later among its experts is left out; STRATA_ROUTE_TAIL_SKIP=0 turns "
                                 "it off\n",
                         o.tail_skip, o.tail_skip + 1);
        // Pages on another node than the threads are read at about two thirds of the speed.  The file cache puts a
        // page on the node of whatever read it first, so a file read back by something else lands on any.
        const bool far = engine.numa_share() > 0.0 && engine.numa_share() < 0.95;
        if (engine.numa_node() >= 0)
            std::fprintf(stderr, "strata-poly: the CPU threads stay on NUMA node %d, which has %.0f%% of the experts' "
                                 "pages%s\n", engine.numa_node(), 100.0 * engine.numa_share(),
                         far ? "; the rest is read from another node, more slowly" : "");
        else if (far)
            std::fprintf(stderr, "strata-poly: the experts' pages are on several NUMA nodes, %.0f%% on the fullest; "
                                 "the CPU reads the ones far from it more slowly\n", 100.0 * engine.numa_share());
        if (eos.empty())
            for (const char* key : {"tokenizer.ggml.eos_token_id", "tokenizer.ggml.eot_token_id",
                                    "tokenizer.ggml.eom_token_id"}) {
                const int64_t id = engine.meta_u(key, -1);
                if (id >= 0 && std::find(eos.begin(), eos.end(), (int32_t) id) == eos.end()) eos.push_back((int32_t) id);
            }
        Session ss(engine, eos, cache, lookup);
        int rc = 0;
        if (want_serve) {
            rc = serve(ss);
        } else if ((long long) ids.size() + max_new > engine.max_context()) {
            throw std::runtime_error("the prompt and --max-new together exceed --max-context");
        } else {
            Pictures pictures;
            if (!images.empty()) {
                if (!engine.sees()) throw std::runtime_error("this model does not read pictures");
                if (image_pad < 0) throw std::runtime_error("--images needs --image-pad");
                const std::string wrong = take_pictures(images, image_pad, engine.n_embd(), ids, pictures);
                if (!wrong.empty()) throw std::runtime_error(wrong);
            }
            if (!follow.empty()) ss.follow = &follow;
            rc = once(ss, ids, pictures, image_pad, max_new, s, top_probs);
        }
        engine.save_profile();
        return rc;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "strata-poly: %s\n", e.what());
        if (want_serve) {
            std::printf("ERR %s\n", e.what());
            std::fflush(stdout);
        }
        return 1;
    }
}
