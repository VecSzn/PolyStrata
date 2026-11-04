// src/poly/conversations_test.cpp - Conversations against an engine whose state can be checked: after every
// request the engine must hold exactly what reading the prompt from its first token gives.
//
//   g++ -std=c++20 src/poly/conversations_test.cpp src/poly/spill.cpp -o conversations_test && ./conversations_test
#include "conversations.hpp"

#include <chrono>
#include <cstdlib>
#include <cstring>
#include <random>
#include <thread>

using strata::poly::CacheOptions;
using strata::poly::Conversations;

namespace {

int checks = 0;
void check(bool ok, const char* what) {
    ++checks;
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        std::exit(1);
    }
}

uint64_t mix(uint64_t h, int32_t t) { return (h ^ (uint64_t) (uint32_t) t) * 0x9E3779B97F4A7C15ull + 0x7F4A7C15ull; }

// The recurrent state is a hash of the tokens read.  A token leaves two rows, as the real engine's layers do: the
// hash after it, and the token.
struct Fake {
    static constexpr int64_t ctx = 1 << 16;
    uint64_t h = 1;
    int64_t at = 0, read = 0;
    size_t state = 8;   // bytes a checkpoint takes; the hash is the first 8
    std::vector<uint64_t> a = std::vector<uint64_t>(ctx), b = std::vector<uint64_t>(ctx);

    void reset() { h = 1, at = 0; }
    void eval(const int32_t* t, int64_t n) {
        for (int64_t i = 0; i < n; ++i, ++at, ++read) {
            a[(size_t) at] = h = mix(h, t[i]);
            b[(size_t) at] = (uint64_t) t[i];
        }
    }
    size_t state_bytes() const { return state; }
    void save_state(uint8_t* dst) const { std::memcpy(dst, &h, 8); }
    void load_state(const uint8_t* src, int64_t pos) { std::memcpy(&h, src, 8), at = pos; }
    size_t row_bytes() const { return 16; }
    void save_rows(int64_t first, int64_t n, uint8_t* dst) const {
        std::memcpy(dst, &a[(size_t) first], (size_t) n * 8);
        std::memcpy(dst + n * 8, &b[(size_t) first], (size_t) n * 8);
    }
    void load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
        std::memcpy(&a[(size_t) first], src + (first - from) * 8, (size_t) n * 8);
        std::memcpy(&b[(size_t) first], src + count * 8 + (first - from) * 8, (size_t) n * 8);
    }

    bool holds(const std::vector<int32_t>& ids) const {
        uint64_t x = 1;
        for (size_t i = 0; i < ids.size(); ++i)
            if (a[i] != (x = mix(x, ids[i])) || b[i] != (uint64_t) ids[i]) return false;
        return at == (int64_t) ids.size() && h == x;
    }
};

// Above zero: the prompts are read this many tokens at a time, past the checkpoints inside (inside()), as an
// engine does that reads several calls together.  Zero: a call at a time, up to the next checkpoint.
int64_t far = 0;
// Above zero: one such read in so many is given up part way (rewind()), and the request is made again.
unsigned quits = 0;

struct Chat {
    Fake e;
    Conversations<Fake> kept;
    std::mt19937 rng{7};
    std::vector<std::pair<int64_t, uint8_t*>> due;
    explicit Chat(const CacheOptions& o) : kept(e, o) {}

    // A request as the server makes it: the prompt, where its last turn begins, then `reply` generated tokens.
    // Returns where the prompt was taken up, and leaves the tokens the engine then holds in `ids`.
    int64_t ask(std::vector<int32_t>& ids, int reply = 3, int64_t turn = -1) {
        const int64_t n = (int64_t) ids.size(), from = kept.resume(ids);
        kept.turn_at(turn);
        check(from >= 0 && from < n && e.at == from, "a prompt is taken up before its last token");
        for (int64_t i = from; i < n;) {
            const bool last = i == n - 1;
            const int64_t room = std::min<int64_t>(n - 1 - i, far > 0 ? far : 100);
            const int64_t m = last ? 1 : far > 0 ? room : kept.until_checkpoint(i, room);
            kept.before(i, last);
            if (far > 0 && !last && quits > 0 && rng() % quits == 0) {
                // the engine got part of the way, a state among the ones due written or not, and stopped
                kept.inside(i, m, due);
                e.eval(ids.data() + i, m / 2);
                if (!due.empty() && due[0].second && due[0].first <= m / 2) e.save_state(due[0].second);
                e.h ^= 0x5EEDull;
                const int64_t back = kept.rewind();
                check(back <= i && e.at == back && e.holds(kept.seq()), "a read given up leaves the last checkpoint");
                return ask(ids, reply, turn);
            }
            if (far > 0 && !last) {
                kept.inside(i, m, due);
                int64_t done = 0;
                for (const auto& d : due) {   // the engine writes a state when it has read up to it
                    e.eval(ids.data() + i + done, d.first - done);
                    done = d.first;
                    if (d.second) e.save_state(d.second);
                }
                e.eval(ids.data() + i + done, m - done);
            } else {
                e.eval(ids.data() + i, m);
            }
            kept.read(ids.data() + i, m);
            i += m;
        }
        kept.prompt_was(n);
        for (int k = 0; k < reply; ++k) {
            const int32_t t = 1000000 + (int32_t) (rng() % 1000);
            e.eval(&t, 1);
            kept.read(&t, 1);
            ids.push_back(t);
        }
        check(e.holds(ids) && kept.seq() == ids, "the engine holds the prompt and the reply");
        kept.release();   // between requests, as the server does: what comes back is read from the file cache
        return from;
    }
};

std::vector<int32_t> text(int seed, int n) {
    std::vector<int32_t> v((size_t) n);
    std::mt19937 r((unsigned) seed);
    for (int32_t& t : v) t = (int32_t) (r() % 50000);
    return v;
}

std::vector<int32_t> operator+(std::vector<int32_t> a, const std::vector<int32_t>& b) {
    a.insert(a.end(), b.begin(), b.end());
    return a;
}

CacheOptions options(int64_t park_mib, int slots = 4) {
    CacheOptions o;
    o.every = 2048;
    o.park_mib = park_mib;
    o.park_slots = slots;
    return o;
}

void one_conversation() {
    Chat c(options(0));
    std::vector<int32_t> a = text(1, 5000);
    check(c.ask(a) == 0, "a first prompt is read from its start");
    const std::vector<int32_t> turn1 = a;
    a = a + text(2, 300);
    check(c.ask(a) == (int64_t) turn1.size(), "a prompt that goes on from the reply goes on from the engine");

    // the client sends the reply back in other tokens: back to the checkpoint before the first prompt's last token
    std::vector<int32_t> b(turn1.begin(), turn1.begin() + 5000);
    b = b + text(3, 400);
    check(c.ask(b) == 4999, "a rewritten reply costs the reply");

    // an edit far back: the last checkpoint before it
    std::vector<int32_t> d(b.begin(), b.begin() + 3000);
    d = d + text(4, 100);
    check(c.ask(d) == 2048, "an edit goes back to the checkpoint before it");
    check(c.kept.parked() == 0, "nothing is parked without a budget");

    std::vector<int32_t> other = text(5, 4000);
    check(c.ask(other) == 0, "another conversation starts from nothing");
    check(c.ask(d) == 0, "and without a budget the first one is gone");
}

void two_conversations() {
    Chat c(options(64));
    std::vector<int32_t> a = text(1, 9000), b = text(2, 3000);
    c.ask(a);
    c.ask(b);
    check(c.kept.parked() == 1, "the first conversation is parked");
    a = a + text(3, 200);
    const int64_t before = c.e.read;
    check(c.ask(a) == 8999, "it comes back at the end of its prompt");
    check(c.e.read - before == 3 + 200 + 1 + 3, "and only the reply and the new turn are read");
    b = b + text(4, 100);
    check(c.ask(b) == 2999, "so does the second");

    // a side request too short to be worth a slot
    std::vector<int32_t> title = text(6, 300);
    c.ask(title);
    a = a + text(7, 50);
    check(c.ask(a) > 9000, "a short request between two turns costs nothing");
    check(c.kept.parked() == 1, "and is not kept");
}

void shared_start() {
    Chat c(options(64));
    const std::vector<int32_t> sys = text(1, 5000);
    std::vector<int32_t> a = sys + text(2, 6000), sub = sys + text(3, 2500);
    c.ask(a);
    check(c.ask(sub) == 4096, "a prompt with the same start takes it from the checkpoint inside");
    check(c.kept.parked() == 1, "and the first conversation is kept whole");
    sub = sub + text(4, 100);
    check(c.ask(sub) == 7503, "the second goes on from where the engine is");
    a = a + text(5, 100);
    check(c.ask(a) == 10999, "and so does the first");
    std::vector<int32_t> sub2 = sys + text(6, 2500);
    check(c.ask(sub2) == 4096, "a third shares the start again");
    check(c.kept.parked() == 2, "with both others parked");
}

// The server says where a prompt's last turn begins.  The chat's next request has the reply in other tokens than
// the ones the prompt ended with after the turn's header (a template drops the reply's thinking): it goes on from
// the turn, and not from a checkpoint thousands of tokens back.
void last_turn() {
    const std::vector<int32_t> sys = text(1, 5000), head = text(2, 3), think = text(3, 2);
    Chat c(options(64));
    std::vector<int32_t> a = sys + head + think;
    check(c.ask(a, 3, 5000) == 0, "a first prompt is read from its start");
    std::vector<int32_t> b = sys + head + text(4, 300) + head + think;
    check(c.ask(b, 3, 5303) == 5000, "the next turn goes on from where the last one began");
    std::vector<int32_t> d = sys + head + text(4, 300) + head + text(5, 200) + head + think;
    check(c.ask(d, 3, 5506) == 5303, "and so does the one after it");

    Chat e(options(64));
    a = sys + head + think;
    e.ask(a);
    b = sys + head + text(4, 300) + head + think;
    check(e.ask(b) == 4096, "without the place, the checkpoint before the last token is past where they differ");
}

void limits() {
    // two slots: a third conversation pushes out the one not used for longest
    Chat c(options(64, 2));
    std::vector<int32_t> v[4] = {text(1, 3000), text(2, 3000), text(3, 3000), text(4, 3000)};
    for (auto& x : v) c.ask(x);
    check(c.kept.parked() == 2 && c.kept.evictions() == 1, "the slots bound what is parked");
    v[0] = v[0] + text(5, 10);
    check(c.ask(v[0]) == 0, "the pushed-out conversation is read again");
    v[2] = v[2] + text(6, 10);
    check(c.ask(v[2]) == 2999, "a kept one is not");

    // a budget of 1 MiB with checkpoints of 300 KiB: a parked conversation holds one checkpoint and its rows
    Chat d(options(1, 8));
    d.e.state = 300 << 10;
    std::vector<int32_t> w[4] = {text(1, 1500), text(2, 1500), text(3, 1500), text(4, 1500)};
    for (auto& x : w) d.ask(x);
    check(d.kept.parked_bytes() <= (1u << 20) && d.kept.parked() == 3, "the budget bounds what is parked");
    for (auto& x : w) d.ask(x = x + text(9, 5));

    // the host says no
    CacheOptions o = options(64);
    o.admit = [](size_t) { return false; };
    Chat e(o);
    std::vector<int32_t> p = text(1, 3000), q = text(2, 3000);
    e.ask(p);
    e.ask(q);
    check(e.kept.parked() == 0, "nothing is parked when the host has no memory to spare");
}

// Any order of prompts over conversations that share starts, under any options: the engine always ends up
// holding the prompt.
void any_order() {
    std::mt19937 rng(11);
    const int64_t every[] = {0, 64, 256, 1000}, budget[] = {0, 1, 64};
    for (int round = 0; round < 60; ++round) {
        CacheOptions o;
        o.checkpoints = (int) (rng() % 5);
        o.every = every[rng() % 4];
        o.park_mib = budget[rng() % 3];
        o.park_slots = (int) (rng() % 4);
        o.park_min = rng() % 2 ? 200 : 0;
        Chat c(o);
        c.e.state = rng() % 2 ? 100 << 10 : 8;
        c.rng.seed((unsigned) round);
        std::vector<std::vector<int32_t>> chats;
        for (int step = 0; step < 40; ++step) {
            std::vector<int32_t> ids;
            const unsigned what = rng() % 10;
            if (chats.empty() || what == 0) {
                ids = text((int) rng(), 1 + (int) (rng() % 1500));
            } else {
                ids = chats[rng() % chats.size()];
                if (what < 4) ids.resize(1 + rng() % ids.size());   // goes on from somewhere inside
                ids = ids + text((int) rng(), (int) (rng() % 400));
            }
            const int64_t turn = rng() % 2 ? (int64_t) (rng() % ids.size()) : -1;
            c.ask(ids, (int) (rng() % 5), turn);
            chats.push_back(ids);
            check(c.kept.parked() <= (size_t) std::max(o.park_slots, 0), "no more parked than slots");
            check(c.kept.parked_bytes() <= ((size_t) o.park_mib << 20), "no more parked than the budget");
        }
    }
}

}  // namespace

// A long prompt read in one go has more checkpoints due than a conversation keeps.  Only the ones it keeps get a
// state, and they are the ones that reading up to each checkpoint in turn leaves: where a prompt that goes on
// from each place is taken up says which are there.
void long_read() {
    std::vector<int64_t> left[2];
    const std::vector<int32_t> a = text(1, 31000);
    for (int way = 0; way < 2; ++way) {
        far = way ? 40000 : 0;
        for (int64_t place = 2048; place < 31000; place += 2048) {
            Chat c(options(0));
            c.e.state = 1 << 20;
            std::vector<int32_t> first = a;
            check(c.ask(first) == 0, "a first prompt is read from its start");
            std::vector<int32_t> b(a.begin(), a.begin() + place + 1);
            b = b + text(3, 5);
            left[way].push_back(c.ask(b));
        }
    }
    far = 0;
    check(left[0] == left[1], "a read past its checkpoints keeps the ones a read to each of them keeps");
    const std::vector<int64_t> six = {0,     0,     0,     8192,  8192,  8192,  8192, 16384,
                                      16384, 16384, 22528, 22528, 22528, 28672, 30720};
    check(left[1] == six, "which are five of the fifteen inside, beside the one before the prompt's last token");
}

// What is kept leaves the heap for a temporary file after a release, and reads back the same, before and after its
// pages leave the process again.  The writer is a thread: the heap goes once it is done and a release follows.
void spill() {
    const size_t n = (size_t(3) << 20) + 123;   // not a whole number of pages
    auto s = std::make_shared<const strata::poly::Spill>(n);
    uint8_t* heap = s->data();
    check(heap != nullptr && s->bytes() == n, "a spill has its bytes");
    for (size_t i = 0; i < n; ++i) heap[i] = (uint8_t) (i * 131 + (i >> 9));
    std::vector<uint8_t> want(heap, heap + n);
    s->release();   // hands it to the writer
    const uint8_t* p = heap;
    for (int k = 0; k < 500 && p == heap; ++k) {
        std::this_thread::sleep_for(std::chrono::milliseconds(10));
        s->release();   // frees the heap once it is written
        p = s->data();
    }
    check(p != heap, "the heap is let go once the file holds it");
    check(std::memcmp(p, want.data(), n) == 0, "the file holds the same bytes");
    s->release();   // the pages the read brought back leave the process
    check(std::memcmp(s->data(), want.data(), n) == 0, "and reads back the same after its pages left");
}

int main() {
    spill();
    for (const int64_t f : {0, 6000, 700}) {   // 700: reads that start and end between checkpoints
        far = f;
        one_conversation();
        two_conversations();
        shared_start();
        last_turn();
        limits();
        any_order();
    }
    quits = 3;   // with reads given up, the places a prompt is taken up from are others: any order, any options
    for (const int64_t f : {6000, 700}) {
        far = f;
        any_order();
    }
    quits = 0;
    long_read();
    std::printf("conversations_test: %d checks passed\n", checks);
    return 0;
}
