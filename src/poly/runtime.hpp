// src/poly/runtime.hpp - what the engine's source files share and its users do not see.
#pragma once

#include "engine.hpp"

#include "strata/artifact/gguf_reader.hpp"

#include <cuda_runtime.h>

#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <memory>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

namespace strata::poly {

constexpr uint32_t kTypeF32 = 0, kTypeBF16 = 30;
constexpr int kMaxUsed = 32;
constexpr int kSumRows = 32;        // routed experts one launch adds into their tokens' rows (ScaledRows)
constexpr int kSlots = 8;           // cached experts one launch multiplies (kernels::MmvqSlots)
constexpr int kCpuRows = 8;         // tokens whose CPU experts one call computes; a window has no more
// From this many tokens, a call runs its routed experts on the GPU alone and copies the ones that are not cached
// through VRAM.  Below it, that copy costs more than the CPU takes to compute the few experts.
constexpr int kStageRows = 32;
// Engine::read() takes as many tokens as leave their rows of the residual in this much host memory.  With experts
// outside the cache a layer's are copied to the card once for all of a read's tokens, so more is faster: GLM 5.3
// Flash reads 7,788 tokens at 606 to 616 tokens/s in one read and at 517 to 531 in reads of 2,048, and on an
// 8 GB card Qwen3.6 reads 30,836 at 1,386 to 1,399 in one read and at 1,101 to 1,233 in reads of 8,192.  A read
// of more than two calls keeps those rows in pinned host memory, 64 KiB a token for GLM and 8 for Qwen3.6: this
// much is 8,192 tokens of GLM's and 65,536 of Qwen3.6's.
constexpr size_t kReadBytes = size_t(512) << 20;
constexpr size_t kChunkBytes = size_t(512) << 20;   // one upload, and each of the two pinned buffers
// Zero bytes behind every matrix the batched kernels may multiply (Mmq::fits): they read up to a step of 256
// columns past a row that ends inside one, which is at most 7 blocks of 34 bytes.
constexpr size_t kTail = 256;
// follow(): an expert's recent use is the count of its uses, halved every kSwapHalf tokens.  An expert takes a
// cached one's slot when it was used twice lately and more than twice as much as that one.  Counts this small are
// noisy, and an upload gets in the way of the work: a slot goes to an expert that will clearly be used more.
constexpr float kSwapMin = 2.f;
constexpr int kSwapHalf = 128;
// Engine::pick_way(): each way of a window is timed this many times before the faster one is taken.  What a way
// takes moves, most in an answer's first windows, while the cache comes to hold the answer's experts, so the way
// behind is timed again after kWayAgain windows of that size, then after twice as many each time it is still
// behind, up to kWayAgainMost.  A time counts kWayWeight against the ones before, and one of the way behind
// kWayBehind, since it has fewer and older ones.
constexpr int kWayFirst = 3;
constexpr int kWayAgain = 4, kWayAgainMost = 64;
constexpr double kWayWeight = 0.125, kWayBehind = 0.5;

// window(): the recurrent states a window leaves when its layers can leave them after any of its tokens
// (Engine::snaps_spread()), 62 MiB each for Qwen3.6.  A window of four has one after each token but the last,
// so a longer one takes no more VRAM than that, and keep() puts up to two right tokens through the layers again.
constexpr int kSnapPlaces = 3;

// The attention caches grow with the sequence (Engine::cover()): a call that would write past the rows they have
// gives them VRAM for rows kKvGrow at a time, kKvAhead past the call, and the expert cache gives that up in pieces
// of kArenaPiece.
constexpr int64_t kKvGrow = 4096, kKvAhead = 1024;
constexpr size_t kArenaPiece = size_t(64) << 20;

using Clock = std::chrono::steady_clock;

inline double ms_since(Clock::time_point t) {
    return std::chrono::duration<double, std::milli>(Clock::now() - t).count();
}

// A failed call also leaves its error for cudaGetLastError(), where the check after the next kernel launch would
// find it and fail a later request with it.  The exception reports it, so it is taken back.
inline void ck(cudaError_t e, const char* what) {
    if (e == cudaSuccess) return;
    cudaGetLastError();
    throw std::runtime_error(std::string(what) + ": " + cudaGetErrorString(e));
}

inline size_t pad(size_t n) { return (n + 255) & ~size_t(255); }

inline std::string blk(int il, const char* name) { return "blk." + std::to_string(il) + "." + name; }

inline const float* fp(const Dev& d) { return (const float*) d.p; }
inline cudaStream_t cs(void* s) { return (cudaStream_t) s; }

inline std::string shape_of(const TensorInfo& t) {
    std::string s = "[";
    for (size_t d = 0; d < t.shape.size(); ++d) s += (d ? ", " : "") + std::to_string(t.shape[d]);
    return s + "] " + t.type_name();
}

// table's entry for `name`, or null
template <size_t N> const Names* names_of(const Names (&table)[N], const char* name) {
    for (const Names& n : table)
        if (std::strcmp(n.name, name) == 0) return &n;
    return nullptr;
}

// A SwiGLU clamp limit per layer, stored as one value or one per layer.  Absent or zero means no clamp.
inline std::vector<float> clamp_limits(const MetaValue* v, int n_layer) {
    std::vector<float> out((size_t) n_layer, INFINITY);
    if (!v) return out;
    for (int il = 0; il < n_layer; ++il) {
        double x = v->f;
        if (v->type == MetaType::ARRAY) {
            if ((size_t) il >= v->items.size())
                throw std::runtime_error("the GGUF's SwiGLU clamp has no value for layer " + std::to_string(il));
            x = v->items[(size_t) il].f;
        }
        if (x > 1e-6) out[(size_t) il] = (float) x;
    }
    return out;
}

// The model's GGUF: one file, or the numbered files llama.cpp splits a large one into.  The first has the
// metadata, and a tensor is in the file that lists it.
class Shards {
public:
    explicit Shards(const std::string& path) {
        files_.push_back(std::make_unique<GgufFile>(path));
        paths_.push_back(path);
        const MetaValue* no = files_[0]->get("split.no");
        const MetaValue* count = files_[0]->get("split.count");
        if (no && no->is_num() && no->u != 0)
            throw std::runtime_error(path + " is not the first file of its model: give the one whose name ends in "
                                            "-00001-of-...");
        const int n = count && count->is_num() ? (int) count->u : 1;
        if (n > 1) {
            char tail[40];
            std::snprintf(tail, sizeof(tail), "-00001-of-%05d.gguf", n);
            const size_t len = std::strlen(tail);
            if (path.size() < len || path.compare(path.size() - len, len, tail) != 0)
                throw std::runtime_error(path + " is the first of " + std::to_string(n) +
                                         " files, and the others are found by its name: it must end in " + tail);
            const std::string base = path.substr(0, path.size() - len);
            for (int i = 2; i <= n; ++i) {
                std::snprintf(tail, sizeof(tail), "-%05d-of-%05d.gguf", i, n);
                files_.push_back(std::make_unique<GgufFile>(base + tail));
                paths_.push_back(base + tail);
            }
        }
        for (const auto& f : files_)
            for (const TensorInfo& t : f->tensors())
                if (!where_.emplace(t.name, Where{&t, f.get()}).second)
                    throw std::runtime_error("tensor " + t.name + " is in two of the model's files");
    }

    const MetaValue* get(const std::string& key) const { return files_[0]->get(key); }
    /// A vocabulary-sized string array is kept by the reader only as its first 64 items; a lookup past
    /// the sample walks the file's header again (see GgufFile::find_in_array).
    int64_t find_in_array(const char* key, const char* text) const {
        return files_[0]->find_in_array(key, text);
    }
    const std::vector<std::string>& paths() const { return paths_; }
    const TensorInfo* find(const std::string& name) const {
        const auto it = where_.find(name);
        return it == where_.end() ? nullptr : it->second.t;
    }
    const uint8_t* data(const TensorInfo& t) const { return where_.at(t.name).file->tensor_data(t); }
    /// The files' bytes together.
    uint64_t bytes() const {
        uint64_t n = 0;
        for (const auto& f : files_) n += f->file_size();
        return n;
    }
    /// f(tensor) for every tensor of the model, in no order.
    template <class F>
    void each(F&& f) const {
        for (const auto& w : where_) f(*w.second.t);
    }
    /// A tensor's byte count, after checking that its file holds that much.
    size_t payload(const TensorInfo& t) const {
        const GgufFile& g = *where_.at(t.name).file;
        const uint64_t bytes = tensor_payload_bytes(t), room = g.file_size() - g.data_start();
        if (bytes == 0 || t.offset > room || bytes > room - t.offset)
            throw std::runtime_error("tensor " + t.name + " " + shape_of(t) + " does not fit its file");
        return (size_t) bytes;
    }

private:
    struct Where {
        const TensorInfo* t;
        const GgufFile* file;
    };
    std::vector<std::unique_ptr<GgufFile>> files_;
    std::vector<std::string> paths_;
    std::unordered_map<std::string, Where> where_;
};

}  // namespace strata::poly
