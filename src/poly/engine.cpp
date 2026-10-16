// src/poly/engine.cpp - see the header.  What every architecture shares: the files, the weights' way to VRAM, the
// routed experts (the cache, the CPU, a prompt batch), the sequence's state, and the calls that feed tokens.
#include "engine.hpp"

#include "deepseek4.hpp"
#include "gemma4.hpp"
#include "glm5next.hpp"
#include "lfm2moe.hpp"
#include "mmq.hpp"
#include "qwen35.hpp"
#include "runtime.hpp"
#include "strata/poly/kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"

#include "ggml.h"

#include <cuda.h>
#include <cuda_runtime.h>

#include <algorithm>
#include <atomic>
#include <bit>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <exception>
#include <filesystem>
#include <new>
#include <numeric>
#include <stdexcept>
#include <thread>
#include <unordered_map>

#ifdef __linux__
#include <sched.h>
#include <unistd.h>
#endif

namespace strata::poly {
namespace {

// A launch's tiles are as wide as the most rows one of its matrices has, up to 128, and every matrix in it
// computes whole tiles.  So a prompt batch's cached experts go by how many rows they have (moe_rows()): a launch
// for the experts of up to 32 rows, one for those of up to 128, and one for the experts with more.  Narrower
// widths saved less work than their launches cost.
constexpr int kTile[] = {32, 128};
constexpr int kClasses = (int) (sizeof(kTile) / sizeof(kTile[0])) + 1;

int tile_class(int rows) {
    int c = 0;
    while (c < kClasses - 1 && rows > kTile[c]) ++c;
    return c;
}

// A container's CPU quota is not in the affinity mask: a 25-CPU pod on a 208-thread host shows 208.
int usable_cpus() {
    long long n = (long long) std::thread::hardware_concurrency();
#ifdef __linux__
    cpu_set_t set;
    if (sched_getaffinity(0, sizeof(set), &set) == 0) n = CPU_COUNT(&set);
    long long quota = -1, period = 0;
    if (FILE* f = std::fopen("/sys/fs/cgroup/cpu.max", "r")) {   // "max 100000" when there is no quota
        if (std::fscanf(f, "%lld %lld", &quota, &period) != 2) quota = -1;
        std::fclose(f);
    } else if (FILE* q = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_quota_us", "r")) {
        if (std::fscanf(q, "%lld", &quota) != 1) quota = -1;
        std::fclose(q);
        if (FILE* p = std::fopen("/sys/fs/cgroup/cpu/cpu.cfs_period_us", "r")) {
            if (std::fscanf(p, "%lld", &period) != 1) period = 0;
            std::fclose(p);
        }
    }
    if (quota > 0 && period > 0) n = std::min(n, std::max(1LL, quota / period));
#endif
    return (int) std::max(n, 1LL);
}

// On a machine with several NUMA nodes the file cache's pages are on the node whose CPU read them first, and the
// CPU reads the experts at the speed of memory: from the other node that is slower (29.4 against 22.5 ms of CPU
// experts a token on a two-socket Xeon).  `spans` are the experts in the mapped files.  When most of their pages
// are on one node, and the process may use `threads` CPUs of it, this thread is kept on that node's CPUs, and so
// are the threads it starts from then on.  Returns the node, or -1 when nothing was changed.
int stay_near(const std::vector<std::pair<const uint8_t*, size_t>>& spans, const std::vector<std::string>& files,
              int threads, double& share) {
#ifdef __linux__
    if (access("/sys/devices/system/node/node1", F_OK) != 0) return -1;   // one node
    // A page shows in numa_maps once it is mapped: touch a few thousand, spread over the experts.
    size_t total = 0;
    for (const auto& s : spans) total += s.second;
    const size_t step = std::max<size_t>(total / 4096, 4096);
    volatile uint8_t seen = 0;
    for (const auto& s : spans)
        for (size_t at = 0; at < s.second; at += step) seen = (uint8_t) (seen + s.first[at]);

    std::vector<std::string> real;
    for (const std::string& f : files)
        if (char* r = realpath(f.c_str(), nullptr)) {
            real.emplace_back(r);
            std::free(r);
        }
    std::vector<long long> pages;   // per node, of the model's files
    if (FILE* f = std::fopen("/proc/self/numa_maps", "r")) {
        char* line = nullptr;
        size_t cap = 0;
        while (getline(&line, &cap, f) > 0) {
            const char* file = std::strstr(line, " file=");
            if (!file) continue;
            file += 6;
            const size_t len = std::strcspn(file, " \n");
            bool ours = false;
            for (const std::string& r : real) ours = ours || (r.size() == len && r.compare(0, len, file, len) == 0);
            for (const char* p = file + len; ours && (p = std::strstr(p, " N")) != nullptr; ++p) {
                int node = 0;
                long long n = 0;
                if (std::sscanf(p, " N%d=%lld", &node, &n) != 2 || node < 0 || node > 1023) continue;
                if ((size_t) node >= pages.size()) pages.resize((size_t) node + 1, 0);
                pages[(size_t) node] += n;
            }
        }
        std::free(line);
        std::fclose(f);
    }
    long long all = 0;
    for (long long n : pages) all += n;
    if (all == 0) return -1;
    const int best = (int) (std::max_element(pages.begin(), pages.end()) - pages.begin());
    share = (double) pages[(size_t) best] / (double) all;
    if (share < 0.6) return -1;   // split between the nodes: one is as good as another

    cpu_set_t may, node;
    CPU_ZERO(&node);
    const std::string list = "/sys/devices/system/node/node" + std::to_string(best) + "/cpulist";
    if (FILE* f = std::fopen(list.c_str(), "r")) {   // "52-103,156-207"
        int a = 0, b = 0;
        while (std::fscanf(f, "%d", &a) == 1) {
            b = a;
            int c = std::fgetc(f);
            if (c == '-' && std::fscanf(f, "%d", &b) == 1) c = std::fgetc(f);
            for (int i = a; i <= b && i < CPU_SETSIZE; ++i) CPU_SET(i, &node);
            if (c != ',') break;
        }
        std::fclose(f);
    }
    if (sched_getaffinity(0, sizeof(may), &may) != 0) return -1;
    CPU_AND(&node, &node, &may);
    if (CPU_COUNT(&node) < threads || sched_setaffinity(0, sizeof(node), &node) != 0) return -1;
    return best;
#else
    (void) spans, (void) files, (void) threads, (void) share;
    return -1;
#endif
}

// The mapped file's pages come into this process's page table as they are first read, a fault every few pages,
// and an expert read through those faults arrives at half the speed: 21.0 against 11.0 ms a token for GLM 5.3
// Flash's CPU experts on the two-socket Xeon, while every expert a token meets is new to the process.  So one
// thread reads a byte of every page of the experts, once, beside whatever the engine is doing: the 100 GiB of a
// model that the file cache holds take it 4 s.  What the cache does not hold is read from the disk here, not
// while a token waits.
void map_pages(const std::vector<std::pair<const uint8_t*, size_t>>& spans, const std::atomic<bool>& quit) {
    constexpr size_t kPage = 4096, kLook = size_t(1) << 24;   // `quit` is looked at every 16 MiB
    volatile uint8_t seen = 0;
    for (const auto& s : spans) {
        for (size_t at = 0; at < s.second; at += kPage) {
            if (at % kLook == 0 && quit.load(std::memory_order_relaxed)) return;
            seen = (uint8_t) (seen + s.first[at]);
        }
        if (s.second > 0) seen = (uint8_t) (seen + s.first[s.second - 1]);   // its last page, when it starts in one
    }
}

// Lines of "layer expert count".  A missing file is not an error: the first run writes it.
void read_profile(const std::string& path, int n_layer, int n_expert, std::vector<int64_t>& out) {
    FILE* f = std::fopen(path.c_str(), "r");
    if (!f) return;
    out.assign((size_t) n_layer * n_expert, 0);
    int il = 0, e = 0;
    long long n = 0;
    while (std::fscanf(f, "%d %d %lld", &il, &e, &n) == 3)
        if (il >= 0 && il < n_layer && e >= 0 && e < n_expert && n > 0) out[(size_t) il * n_expert + e] = n;
    std::fclose(f);
}

void cu_ck(CUresult r, const char* what) {
    if (r == CUDA_SUCCESS) return;
    const char* s = nullptr;
    cuGetErrorString(r, &s);
    throw std::runtime_error(std::string(what) + ": " + (s ? s : "a CUDA driver error"));
}

size_t up_to(size_t n, size_t grain) { return (n + grain - 1) / grain * grain; }

// The largest of n values, as eight maxima side by side: one best-so-far is a chain every compare waits on, and
// a vocabulary is 150,000 values.
constexpr int kLanes = 8;
float largest(const float* x, int n) {
    float mx[kLanes];
    std::fill(mx, mx + kLanes, -INFINITY);
    int i = 0;
    for (; i + kLanes <= n; i += kLanes)
        for (int j = 0; j < kLanes; ++j) mx[j] = std::max(mx[j], x[i + j]);
    float best = -INFINITY;
    for (int j = 0; j < kLanes; ++j) best = std::max(best, mx[j]);
    for (; i < n; ++i) best = std::max(best, x[i]);
    return best;
}

// The CUDA calls of a card's engine go to its card: a split's engines share the caller's thread (but in read()),
// and the card is the thread's.  Back to the caller's card at the end.  One engine alone changes nothing.
class OnCard {
public:
    OnCard(int device, bool split) : split_(split) {
        if (!split_) return;
        cudaGetDevice(&was_);
        cudaSetDevice(device);   // a failure shows in the calls that follow
    }
    ~OnCard() {
        if (split_) cudaSetDevice(was_);
    }
    OnCard(const OnCard&) = delete;
    OnCard& operator=(const OnCard&) = delete;

private:
    bool split_;
    int was_ = 0;
};

// Work on a thread of its own, whose exception wait() throws.  Going out of scope it waits and drops the exception.
class Behind {
public:
    Behind() = default;
    Behind(const Behind&) = delete;
    Behind& operator=(const Behind&) = delete;
    ~Behind() {
        if (th_.joinable()) th_.join();
    }
    void go(std::function<void()> work) {
        th_ = std::thread([this, work = std::move(work)] {
            try {
                work();
            } catch (...) {
                err_ = std::current_exception();
            }
        });
    }
    void wait() {
        if (th_.joinable()) th_.join();
        if (err_) std::rethrow_exception(std::exchange(err_, nullptr));
    }

private:
    std::thread th_;
    std::exception_ptr err_;
};

// The first layer of each card's share of n_layer layers, for Options::layer_split on `cards` cards; empty: no
// split.  "auto" shares the layers' bytes out by each card's free VRAM, less what every card's engine takes beside
// its layers and, on the last card, the output and the draft block.  Then every card caches about the same share
// of its layers' experts, which is where a byte of VRAM on one is worth as much as on another.
std::vector<int> layer_starts(const std::string& split, const Shards& file, int n_layer, int cards) {
    if (split.empty() || cards < 2) return {};
    std::vector<int> starts{0};
    if (split != "auto") {
        for (size_t at = 0; at < split.size();) {
            const size_t comma = std::min(split.find(',', at), split.size());
            const std::string s = split.substr(at, comma - at);
            char* end = nullptr;
            const long v = std::strtol(s.c_str(), &end, 10);
            // the last card keeps two layers at least: Gemma 4's draft model reads the last two layers' caches
            if (s.empty() || *end || v <= starts.back() || v > n_layer - 2)
                throw std::runtime_error("--layer-split " + split + ": the first layer of each later card, rising, "
                                         "between 1 and " + std::to_string(n_layer - 2) + " (\"auto\" shares them "
                                         "out by each card's free VRAM)");
            starts.push_back((int) v);
            at = comma + 1;
        }
        if ((int) starts.size() > cards)
            throw std::runtime_error("--layer-split " + split + " is for " + std::to_string(starts.size()) +
                                     " cards; this process sees " + std::to_string(cards));
        return starts;
    }
    if (n_layer < 2 * cards) throw std::runtime_error("the model has too few layers to split them on " + std::to_string(cards) + " cards");
    std::vector<double> before((size_t) n_layer + 1, 0.0);   // the bytes of the layers before each one
    double head = 0;
    file.each([&](const TensorInfo& t) {
        int il = -1;
        if (std::sscanf(t.name.c_str(), "blk.%d.", &il) == 1 && il >= 0 && il < n_layer)
            before[(size_t) il + 1] += (double) file.payload(t);
        else if (il >= n_layer || t.name == "output.weight")   // the draft block, the output
            head += (double) file.payload(t);
    });
    if (!file.find("output.weight") && file.find("token_embd.weight"))   // the output is the token table
        head += (double) file.payload(*file.find("token_embd.weight"));
    for (int il = 0; il < n_layer; ++il) before[(size_t) il + 1] += before[(size_t) il];
    constexpr double kBeside = 1.5 * double(1ull << 30);   // scratch, the staging buffer and the reserve, about
    std::vector<double> room((size_t) cards);
    double all_room = 0;
    for (int d = 0; d < cards; ++d) {
        size_t free_b = 0, total_b = 0;
        ck(cudaSetDevice(d), "cudaSetDevice");
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        room[(size_t) d] = std::max((double) free_b - kBeside - (d == cards - 1 ? head : 0.0), double(1ull << 28));
        all_room += room[(size_t) d];
    }
    double share = 0;
    for (int d = 1; d < cards; ++d) {
        share += room[(size_t) d - 1] / all_room;
        int il = starts.back() + 1;   // card d starts where the layers before it pass the cards before it's share
        while (il < n_layer - 2 * (cards - d) && before[(size_t) il] < share * before[(size_t) n_layer]) ++il;
        starts.push_back(il);
    }
    return starts;
}

// The reserve when none is asked for: a thirty-second of the card, 256 to 768 MiB.  When the machine's RAM is
// tight for the model, a sixteenth, 512 to 1,024: on an 8 GB laptop with 32 GB, Ornith's APEX-I-Quality (23 GB)
// read a 24,000-token prompt at 1,392 and 1,138 tokens/s with 256 MiB kept free and at 1,532 and 1,451 with 512,
// in turns with the build that kept 512 (1,496 and 1,311); the smaller files read as fast either way.  Under
// Windows what the card is short of moves to the system's RAM, which the file then has less of.
int default_reserve_mib(size_t total_b, bool ram_tight) {
    if (ram_tight) return (int) std::clamp<size_t>(total_b >> 24, 512, 1024);
    return (int) std::clamp<size_t>(total_b >> 25, 256, 768);
}

// The RAM the machine has free for the model's file, or 0 when it cannot tell.
size_t ram_available() {
#ifdef _WIN32
    MEMORYSTATUSEX st{};
    st.dwLength = sizeof(st);
    return GlobalMemoryStatusEx(&st) ? (size_t) st.ullAvailPhys : 0;
#else
    size_t kb = 0;
    if (FILE* f = std::fopen("/proc/meminfo", "r")) {
        char line[128];
        while (std::fgets(line, sizeof(line), f))
            if (std::sscanf(line, "MemAvailable: %zu kB", &kb) == 1) break;
        std::fclose(f);
    }
    return kb << 10;
#endif
}

}  // namespace

// One expert on its way into the cache.  The engine's thread asks, the follower copies and sends, and the engine's
// thread gives it the slot once the upload has arrived.
struct Engine::Swap {
    enum { kFree, kAsked, kSent };
    std::atomic<int> at{kFree};
    int il = 0, e = 0;
    uint8_t* slot = nullptr;       // where it goes in VRAM: the slot of the expert it replaces
    uint8_t* pinned = nullptr;
    cudaEvent_t done = nullptr;    // marks its upload
};

Engine::Engine() = default;

Engine::~Engine() {
    next_.reset();
    OnCard here(device_, split_);
    map_quit_.store(true);
    if (mapper_.joinable()) mapper_.join();
    if (follower_.joinable()) {
        {
            std::lock_guard<std::mutex> l(swap_m_);
            swap_quit_ = true;
        }
        swap_cv_.notify_all();
        follower_.join();
    }
    if (swap_st_) cudaStreamSynchronize(cs(swap_st_));
    if (up_) cudaStreamSynchronize(cs(up_));
    if (rs_) cudaStreamSynchronize(cs(rs_));
    if (st_) cudaStreamSynchronize(cs(st_));
    workers_.reset();
    mmq_.reset();
    for (void* p : dev_allocs_) cudaFree(p);
    for (Kv& k : kv_) vmm_free(k);
    vmm_free(arena_);
    for (void* p : host_allocs_) cudaFreeHost(p);
    for (void* e : {ev_stage_[0], ev_stage_[1], ev_ahead_, ev_free_, ev_in_[0], ev_in_[1], ev_out_[0], ev_out_[1], ev_hr_})
        if (e) cudaEventDestroy((cudaEvent_t) e);
    for (void* e : ev_row_)
        if (e) cudaEventDestroy((cudaEvent_t) e);
    for (int j = 0; j < n_swap_; ++j)
        if (swap_[(size_t) j].done) cudaEventDestroy(swap_[(size_t) j].done);
    for (void* g : graphs_) cudaGraphExecDestroy((cudaGraphExec_t) g);
    blas_destroy(blas_);
    if (swap_st_) cudaStreamDestroy(cs(swap_st_));
    if (up_) cudaStreamDestroy(cs(up_));
    if (rs_) cudaStreamDestroy(cs(rs_));
    if (st_) cudaStreamDestroy(cs(st_));
}

void* Engine::dmalloc(size_t bytes, const char* what) {
    void* p = nullptr;
    const cudaError_t e = cudaMalloc(&p, std::max<size_t>(bytes, 256));
    if (e != cudaSuccess) {
        cudaGetLastError();   // as ck() does
        throw std::runtime_error("no VRAM for " + std::string(what) + " (" + std::to_string(bytes >> 20) +
                                 " MiB): " + cudaGetErrorString(e));
    }
    dev_allocs_.push_back(p);
    return p;
}

void* Engine::hmalloc(size_t bytes, const char* what) {
    void* p = nullptr;
    const std::string why = std::string(what) + " (" + std::to_string(bytes) + " bytes)";
    // A container at its memory limit refused pinned memory with "invalid argument" five times in five minutes
    // while a download filled its page cache; a moment later the kernel has reclaimed some.
    cudaError_t e = cudaSuccess;
    for (int tries = 0; tries < 4; ++tries) {
        e = cudaMallocHost(&p, std::max<size_t>(bytes, 256));
        if (e == cudaSuccess) break;
        cudaGetLastError();
        std::fprintf(stderr, "strata-poly: %s: %s, trying again\n", why.c_str(), cudaGetErrorString(e));
        std::this_thread::sleep_for(std::chrono::milliseconds(250));
    }
    ck(e, why.c_str());
    host_allocs_.push_back(p);
    return p;
}

// A container whose memory is not much more than the model's files.  The file cache cannot hold them beside
// the engine, so an answer reads experts from the disk: on a pod with 80 GiB and two RTX 3080s, GLM REAP50's
// 79 GB file wrote at 6 tokens/s where 20.  And the pod ended the engine with SIGKILL in half of its long reads,
// within 2% of the limit, with no OOM kill counted by the kernel.  Said once at load; the remedy is RAM.
void Engine::mind_memory() {
#ifdef __linux__
    if (split_ && lo_ > 0) return;   // the first card's engine says it for all of them
    static const char* const files[] = {"/sys/fs/cgroup/memory.max", "/sys/fs/cgroup/memory/memory.limit_in_bytes"};
    unsigned long long limit = 0;
    for (const char* path : files) {
        FILE* f = std::fopen(path, "r");
        if (!f) continue;
        if (std::fscanf(f, "%llu", &limit) != 1) limit = 0;   // "max" when there is none
        std::fclose(f);
        break;
    }
    if (limit == 0 || limit >= (1ULL << 50)) return;   // none, or one beyond any machine's RAM
    const uint64_t file = gg_->bytes() + (dg_ ? dg_->bytes() : 0);
    if (file + (8ULL << 30) <= limit) return;   // the engine and the rest of the container take a few GiB
    std::fprintf(stderr, "strata-poly: the model's files are %.1f GiB and the container has %.1f GiB: the file "
                         "cache cannot hold them beside the engine, so answers will read experts from the disk, "
                         "and the container may end the engine\n",
                 (double) file / (1 << 30), (double) limit / (1 << 30));
#endif
}

void Engine::vmm_reserve(Mapped& m, size_t bytes) {
    CUdeviceptr p = 0;
    m.reserved = up_to(std::max<size_t>(bytes, 1), grain_);
    cu_ck(cuMemAddressReserve(&p, m.reserved, 0, 0, 0), "reserving addresses for VRAM");
    m.base = p;
}

// VRAM for the first `bytes` of m, in pieces of at most `piece` bytes.  Throws std::runtime_error.
void Engine::vmm_map(Mapped& m, size_t bytes, size_t piece, const char* what) {
    bytes = std::min(up_to(bytes, grain_), m.reserved);
    CUmemAllocationProp prop = {};
    prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
    prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
    prop.location.id = device_;
    CUmemAccessDesc access = {};
    access.location = prop.location;
    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
    piece = std::max(grain_, piece / grain_ * grain_);
    while (m.mapped < bytes) {
        const size_t n = std::min(bytes - m.mapped, piece);
        CUmemGenericAllocationHandle h = 0;
        if (cuMemCreate(&h, n, &prop, 0) != CUDA_SUCCESS)
            throw std::runtime_error("no VRAM for " + std::string(what) + " (" + std::to_string(n >> 20) +
                                     " MiB more)");
        const CUdeviceptr at = m.base + m.mapped;
        const CUresult mapped = cuMemMap(at, n, 0, h, 0);
        if (mapped != CUDA_SUCCESS) cuMemRelease(h);
        cu_ck(mapped, "mapping VRAM");
        m.pieces.emplace_back(h, n);
        m.mapped += n;
        const CUresult acc = cuMemSetAccess(at, n, &access, 1);
        if (acc != CUDA_SUCCESS) {   // HIP hands out the physical pages here, not at create: the card may not have
            cuMemUnmap(at, n);       // them even though the addresses do.  Leave m as it was and say no.
            cuMemRelease(h);
            m.pieces.pop_back();
            m.mapped -= n;
            (void) cudaGetLastError();   // HIP's failed SetAccess leaves a sticky error: do not let it escape with the throw
            throw std::runtime_error("no VRAM for " + std::string(what) + " (" + std::to_string(n >> 20) +
                                     " MiB more)");
        }
    }
}

// Gives back the pieces of m that lie past its first `bytes`.  The card must be done with them.
void Engine::vmm_unmap(Mapped& m, size_t bytes) {
    while (!m.pieces.empty() && m.mapped - m.pieces.back().second >= bytes) {
        const auto [h, n] = m.pieces.back();
        m.mapped -= n;
        m.pieces.pop_back();
        cu_ck(cuMemUnmap(m.base + m.mapped, n), "unmapping VRAM");
        cu_ck(cuMemRelease(h), "unmapping VRAM");
    }
}

// For the destructor: what is left of m, and its addresses, without throwing.
void Engine::vmm_free(Mapped& m) {
    for (; !m.pieces.empty(); m.pieces.pop_back()) {
        m.mapped -= m.pieces.back().second;
        cuMemUnmap(m.base + m.mapped, m.pieces.back().second);
        cuMemRelease(m.pieces.back().first);
    }
    if (m.base) cuMemAddressFree(m.base, m.reserved);
    m.base = 0;
}

void* Engine::kv_alloc(size_t bytes, int per) {
    const size_t all = (size_t) ((opt_.max_context + per - 1) / per) * bytes;
    if (!vmm_) return dmalloc(all, "the attention cache");
    Kv k;
    k.bytes = bytes;
    k.per = per;
    vmm_reserve(k, all);
    kv_.push_back(std::move(k));
    return (void*) kv_.back().base;
}

// The attention caches' VRAM for the rows below `end`: rows kKvGrow at a time, kKvAhead more for the answer to go
// on.  The expert cache gives it up (shrink_cache()), as if the rows had been there when it was filled: the
// reserve stays for what is recorded later.  A quarter of the reserve stays free whatever else took VRAM since
// (mind_vram()).
void Engine::cover(int64_t end) {
    kv_reach_ = std::max(kv_reach_, end);
    if (kv_.empty() || end <= kv_rows_) return;
    const int64_t rows = kv_rows_for(end);
    const auto size = [rows](const Kv& k) { return (size_t) ((rows + k.per - 1) / k.per) * k.bytes; };
    size_t more = 0, held = arena_.mapped;
    for (const Kv& k : kv_) {
        const size_t want = std::min(up_to(size(k), grain_), k.reserved);
        if (want > k.mapped) more += want - k.mapped;   // trim_card_kv() leaves a piece it cuts through
        held += k.mapped;
    }
    if (exp_) {
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        const size_t quarter = (size_t(std::max(reserve_mib_, 0)) << 20) / 4;
        size_t cut = held + more > budget_ ? held + more - budget_ : 0;
        if (free_b < more + quarter) cut = std::max(cut, more + quarter - free_b);
        if (cut) shrink_cache(cut);
    }
    // in pieces of kKvGrow rows, so that trim_card_kv() can give back what a long read mapped in one go
    for (Kv& k : kv_)
        vmm_map(k, size(k), (size_t) ((kKvGrow + k.per - 1) / k.per) * k.bytes, "the attention cache");
    kv_rows_ = rows;
}

int64_t Engine::kv_rows_for(int64_t end) const {
    return std::min<int64_t>(opt_.max_context, (end + kKvAhead + kKvGrow - 1) / kKvGrow * kKvGrow);
}

size_t Engine::trim_kv(int64_t end) {
    size_t freed = 0;
    for (Engine* e = this; e; e = e->next_.get()) freed += e->trim_card_kv(end);
    return freed;
}

// A card that holds every expert keeps the whole context's rows (load()): nothing would take what they give back.
size_t Engine::trim_card_kv(int64_t end) {
    const int64_t before = kv_reach_;
    kv_reach_ = end;
    if (kv_.empty() || !exp_ || all_cached()) return 0;
    const int64_t rows = kv_rows_for(std::max(end, before));
    if (rows >= kv_rows_) return 0;
    OnCard here(device_, split_);
    ck(cudaDeviceSynchronize(), "the attention cache");
    size_t freed = 0;
    for (Kv& k : kv_) {
        const size_t had = k.mapped;
        vmm_unmap(k, (size_t) ((rows + k.per - 1) / k.per) * k.bytes);
        freed += had - k.mapped;
    }
    kv_rows_ = rows;
    return freed;
}

// Every expert on its way into the cache arrives and takes its slot.
void Engine::settle_swaps() {
    if (n_swap_ == 0) return;
    const auto failed = [&] {
        std::lock_guard<std::mutex> l(swap_m_);
        return !swap_error_.empty();
    };
    for (int j = 0; j < n_swap_; ++j)
        while (swap_[(size_t) j].at.load() == Swap::kAsked && !failed()) std::this_thread::yield();
    if (failed()) throw std::runtime_error("expert upload: " + swap_error_);
    ck(cudaStreamSynchronize(cs(swap_st_)), "expert upload");
    follow(0);
}

// The expert cache gives up `bytes` or a little more from its end, and loads nothing again: the experts whose
// slots lie there and that were used more than cached ones before the cut take those ones' slots, as follow()
// ranks them, and the others leave the cache.
void Engine::shrink_cache(size_t bytes) {
    size_t cut = arena_.mapped;
    for (size_t i = arena_.pieces.size(); i > 0 && arena_.mapped - cut < bytes; --i) cut -= arena_.pieces[i - 1].second;
    settle_swaps();
    ck(cudaStreamSynchronize(cs(st_)), "the expert cache");
    const size_t ne = (size_t) n_expert_, end = cut > kTail ? cut - kTail : 0;   // kTail stays past the last slot
    struct Held {
        float recent;
        int64_t prior;
        int il, e;
    };
    const auto held = [&](int il, int e) {
        const size_t i = (size_t) il * ne + (size_t) e;
        return Held{recent_.empty() ? 0.f : recent_[i], (prior_.empty() ? 0 : prior_[i]) + use_[i], il, e};
    };
    const auto more = [](const Held& a, const Held& b) {
        return a.recent != b.recent ? a.recent > b.recent : a.prior > b.prior;
    };
    std::vector<Held> past;
    std::vector<std::vector<Held>> low((size_t) n_trade_);   // the cached ones before the cut, by slot size
    for (int il = 0; il < (int) layers_.size(); ++il) {
        const Layer& L = layers_[(size_t) il];
        if (L.trade < 0) continue;
        for (int e = 0; e < n_expert_; ++e) {
            const uint8_t* s = L.slot[(size_t) e];
            if (!s) continue;
            if ((size_t) (s - exp_) + L.slot_bytes > end) past.push_back(held(il, e));
            else low[(size_t) L.trade].push_back(held(il, e));
        }
    }
    std::sort(past.begin(), past.end(), more);
    for (std::vector<Held>& l : low) std::sort(l.begin(), l.end(), [&](const Held& a, const Held& b) { return more(b, a); });
    std::vector<size_t> next((size_t) n_trade_, 0);
    for (const Held& x : past) {
        const Layer& L = layers_[(size_t) x.il];
        uint8_t*& from = layers_[(size_t) x.il].slot[(size_t) x.e];
        const std::vector<Held>& l = low[(size_t) L.trade];
        size_t& k = next[(size_t) L.trade];
        if (k < l.size() && more(x, l[k])) {
            uint8_t*& to = layers_[(size_t) l[k].il].slot[(size_t) l[k].e];
            ck(cudaMemcpyAsync(to, from, L.slot_bytes, cudaMemcpyDeviceToDevice, cs(st_)), "the expert cache");
            from = to;
            to = nullptr;
            ++k;
        } else {
            from = nullptr;
        }
        --cached_;
    }
    ck(cudaStreamSynchronize(cs(st_)), "the expert cache");
    vmm_unmap(arena_, cut);
    if (cut > kTail) ck(cudaMemset(exp_ + end, 0, kTail), "the expert cache");
    exp_at_ = std::min(exp_at_, end);
    staged_il_ = -1;
}

float* Engine::fbuf(size_t n) {
    float* p = scratch_ ? scratch_ + scratch_at_ : nullptr;
    scratch_at_ += (n + 63) & ~size_t(63);
    return p;
}

int64_t Engine::meta_u(const char* key, int64_t fallback) const {
    const MetaValue* v = gg_ ? gg_->get(key) : nullptr;
    return v && v->is_num() ? (int64_t) v->u : fallback;
}

// A layer's tensor, under any of its names.  The draft block's are in the file it comes from.
const TensorInfo* Engine::tensor(int il, const char* name) const {
    const Shards& g = block_ && il >= n_layer_ ? *block_ : *gg_;
    if (const TensorInfo* t = g.find(blk(il, name))) return t;
    if (const Names* n = tensor_alias(name))
        for (const char* other : n->other)
            if (const TensorInfo* t = other ? g.find(blk(il, other)) : nullptr) return t;
    return nullptr;
}

// A value of the metadata of the model that `g` is a file of, under any of the names converters give it.
const MetaValue* Engine::meta_in(const Shards& g, const char* key) const {
    const MetaValue* arch = g.get("general.architecture");
    if (!arch) return nullptr;
    const std::string pre = arch->s + ".";
    if (const MetaValue* v = g.get(pre + key)) return v;
    if (const Names* n = meta_alias(key))
        for (const char* other : n->other)
            if (const MetaValue* v = other ? g.get(pre + other) : nullptr) return v;
    return nullptr;
}

int Engine::shape_i(const char* key) const {
    const MetaValue* v = meta_in(*gg_, key);
    if (!v) throw std::runtime_error("the GGUF has no " + gg_->get("general.architecture")->s + "." + key);
    return (int) v->u;
}

float Engine::shape_f(const char* key) const {
    const MetaValue* v = meta_in(*gg_, key);
    if (!v) throw std::runtime_error("the GGUF has no " + gg_->get("general.architecture")->s + "." + key);
    return (float) v->f;
}

// What every architecture's file says of the model: its layers, its width, its vocabulary and its experts.
//
// The block that guesses a second token after the model's own is one more block after the model's.  It is the
// model's and not a file's: a file that leaves it out can take it from another file of the model (draft_model).
// One converter gives the block a token table or an output matrix of its own, which this engine does not run.
void Engine::load_shape() {
    const MetaValue* nextn = meta_in(*gg_, "nextn_predict_layers");
    n_layer_ = shape_i("block_count") - (nextn ? (int) nextn->u : 0);
    if (opt_.draft && opt_.window != 1) {
        const std::string probe = blk(n_layer_, "nextn.eh_proj.weight");
        if (nextn && nextn->u == 1 && gg_->find(probe)) block_ = gg_.get();
        else if (dg_ && dg_->find(probe)) block_ = dg_.get();
        else if (dg_)
            throw std::runtime_error(opt_.draft_model + " has no block that guesses ahead for this model: no tensor " +
                                     probe);
    }
    draft_ = block_ && !tensor(n_layer_, "nextn.embed_tokens.weight") &&
             !tensor(n_layer_, "nextn.shared_head_head.weight");
    if (!draft_) block_ = nullptr;
    block_layer_ = draft_;
    n_embd_ = shape_i("embedding_length");
    // a file without the key has as many tokens as its token table
    const MetaValue* vocab = meta_in(*gg_, "vocab_size");
    const TensorInfo* te = gg_->find("token_embd.weight");
    n_vocab_ = vocab ? (int) vocab->u : te && te->shape.size() == 2 ? (int) te->shape[1] : 0;
    rms_eps_ = shape_f("attention.layer_norm_rms_epsilon");
    n_expert_ = shape_i("expert_count");
    n_used_ = shape_i("expert_used_count");
    n_ff_exp_ = shape_i("expert_feed_forward_length");
    const MetaValue* shexp = meta_in(*gg_, "expert_shared_feed_forward_length");
    n_ff_shexp_ = shexp ? (int) shexp->u : n_ff_exp_;

    auto need = [](bool ok, const char* what) {
        if (!ok) throw std::runtime_error(std::string("unsupported model: ") + what);
    };
    need(n_layer_ > 0 && n_embd_ > 0 && n_embd_ % 32 == 0 && n_vocab_ > 0, "bad block_count or embedding_length");
    need(n_used_ >= 1 && n_used_ <= kMaxUsed && n_used_ <= n_expert_, "bad expert_used_count");

    // a block from another file must be this model's: a pruned model's has fewer experts to choose among
    if (block_ && block_ != gg_.get()) {
        const TensorInfo* gate = tensor(n_layer_, "ffn_gate_inp.weight");
        if (!gate || gate->shape.size() != 2 || (int64_t) gate->shape[0] != n_embd_ ||
            (int64_t) gate->shape[1] != n_expert_)
            throw std::runtime_error(opt_.draft_model + " is not a file of this model: its block's router is " +
                                     (gate ? shape_of(*gate) : std::string("missing")) + ", the model's layers have [" +
                                     std::to_string(n_embd_) + ", " + std::to_string(n_expert_) + "]");
    }
}

// A MoE layer: its router and shared expert to VRAM, and its routed experts' places in the file, where they
// stay; fill_expert_cache() copies the ones VRAM has room for.  A file may keep an expert's gate and up in one
// tensor, the gate's rows first.
void Engine::load_moe(int il, bool shared) {
    Layer& L = layers_[(size_t) il];
    auto mat = [&](const char* n, int64_t a, int64_t b) { return upload(tensor(il, n), blk(il, n), a, b, kAnyMat); };
    L.moe = true;
    L.gate_inp = mat("ffn_gate_inp.weight", n_embd_, n_expert_);
    if (shared) {
        L.sh_gate = mat("ffn_gate_shexp.weight", n_embd_, n_ff_shexp_);
        L.sh_up = mat("ffn_up_shexp.weight", n_embd_, n_ff_shexp_);
        L.sh_down = mat("ffn_down_shexp.weight", n_ff_shexp_, n_embd_);
    }

    auto exps = [&](const char* n, int64_t n_in, int64_t rows) -> const TensorInfo& {
        const TensorInfo* t = tensor(il, n);
        if (!t) throw std::runtime_error("the GGUF has no tensor " + blk(il, n));
        if (t->shape.size() != 3 || (int64_t) t->shape[0] != n_in || (int64_t) t->shape[1] != rows ||
            (int64_t) t->shape[2] != n_expert_)
            throw std::runtime_error(t->name + " is " + shape_of(*t) + ", expected [" + std::to_string(n_in) + ", " +
                                     std::to_string(rows) + ", " + std::to_string(n_expert_) + "]");
        return *t;
    };
    const TensorInfo& td = exps("ffn_down_exps.weight", n_ff_exp_, n_embd_);
    L.d_type = td.type;
    L.d_bytes = home(td).payload(td) / (size_t) n_expert_;
    L.exp_down = home(td).data(td);
    if (L.d_type == GGML_TYPE_MXFP4) mxfp4_to_q5_0(il);
    if (tensor(il, "ffn_gate_up_exps.weight")) {
        const TensorInfo& gu = exps("ffn_gate_up_exps.weight", n_embd_, 2 * n_ff_exp_);
        L.gu_type = gu.type;
        L.gu_stride = home(gu).payload(gu) / (size_t) n_expert_;
        L.gu_bytes = L.gu_stride / 2;
        L.exp_gate = home(gu).data(gu);
        L.exp_up = L.exp_gate + L.gu_bytes;
    } else {
        const TensorInfo& tg = exps("ffn_gate_exps.weight", n_embd_, n_ff_exp_);
        const TensorInfo& tu = exps("ffn_up_exps.weight", n_embd_, n_ff_exp_);
        if (tg.type != tu.type)
            throw std::runtime_error(tg.name + " and " + tu.name +
                                     " have different types, which this engine does not run");
        L.gu_type = tg.type;
        L.gu_stride = L.gu_bytes = home(tg).payload(tg) / (size_t) n_expert_;
        home(tu).payload(tu);
        L.exp_gate = home(tg).data(tg);
        L.exp_up = home(tu).data(tu);
    }
    std::string err;
    if (!cpu_expert_types_ok((int) L.gu_type, (int) L.d_type, n_embd_, n_ff_exp_, err))
        throw std::runtime_error(blk(il, "ffn_*_exps.weight: ") + err);
    L.slot.assign((size_t) n_expert_, nullptr);
    ++n_moe_;
}

// No kernel here multiplies MXFP4 on the card, by matvec or MMQ, so a layer whose down experts are MXFP4 ran them on
// the CPU, a prompt batch's too: DeepSeek V4 Flash UD-IQ2_XXS has two such layers, and they took 10.7 s of a
// 10,398-token read whose other 41 layers' experts took 4.9.  An MXFP4 value is 0, 1, 2, 3, 4, 6, 8 or 12, with a
// sign, times the block's power of two halved, which Q5_0 holds exactly while that power fits a half.  So the experts
// are converted once, into RAM of the engine's own, 22 bytes a block of 32 instead of 17.  On the CPU both types meet
// the activations as Q8_0, and the dot products differ only in the order of their sums.  A layer with a power out of
// range stays MXFP4.
void Engine::mxfp4_to_q5_0(int il) {
    Layer& L = layers_[(size_t) il];
    constexpr size_t kIn = 17, kOut = 22;   // block_mxfp4, block_q5_0
    const size_t blocks = L.d_bytes * (size_t) n_expert_ / kIn;
    std::unique_ptr<uint8_t[]> out;
    try {
        out.reset(new uint8_t[blocks * kOut]);
    } catch (const std::bad_alloc&) {
        return;
    }
    std::atomic<bool> bad{false};
    auto convert = [&](size_t lo, size_t hi) {
        for (size_t b = lo; b < hi && !bad.load(std::memory_order_relaxed); ++b)
            if (!mxfp4_block_to_q5_0(L.exp_down + b * kIn, out.get() + b * kOut)) bad.store(true);
    };
    const size_t n = (size_t) std::clamp(usable_cpus(), 1, 32), step = (blocks + n - 1) / n;
    std::vector<std::thread> th;
    for (size_t i = 1; i < n; ++i) th.emplace_back(convert, std::min(blocks, i * step), std::min(blocks, (i + 1) * step));
    convert(0, std::min(blocks, step));
    for (std::thread& t : th) t.join();
    if (bad) return;
    L.exp_down = out.get();
    L.d_type = GGML_TYPE_Q5_0;
    L.d_bytes = blocks / (size_t) n_expert_ * kOut;
    converted_.push_back(std::move(out));
    std::fprintf(stderr, "strata-poly: layer %d's down experts converted from MXFP4 to Q5_0, %.2f GiB of RAM\n", il,
                 (double) (blocks * kOut) / (1 << 30));
}

// The engine for a model file: its general.architecture says which.
std::unique_ptr<Engine> Engine::open(const Options& o) {
    const Shards file(o.model);
    const MetaValue* arch = file.get("general.architecture");
    const std::string name = arch ? arch->s : std::string();
    const auto make = [&]() -> std::unique_ptr<Engine> {
        if (name == "glm5-next" || name == "glm5next") return std::make_unique<Glm5Next>();   // Unsloth: no hyphen
        if (name == "qwen35moe") return std::make_unique<Qwen35>();
        if (name == "gemma4") return std::make_unique<Gemma4>();
        if (name == "deepseek4") return std::make_unique<DeepSeek4>();
        if (name == "lfm2moe") return std::make_unique<Lfm2Moe>();
        throw std::runtime_error(o.model + " is a '" + name + "' model, which this engine does not run");
    };
    int cards = 1;
    if (!o.layer_split.empty()) ck(cudaGetDeviceCount(&cards), "cudaGetDeviceCount");
    std::vector<int> starts;
    if (cards > 1) {
        const MetaValue* blocks = file.get(name + ".block_count");
        const MetaValue* nextn = file.get(name + ".nextn_predict_layers");
        if (!blocks) throw std::runtime_error("the GGUF has no " + name + ".block_count");
        int was = 0;
        ck(cudaGetDevice(&was), "cudaGetDevice");
        starts = layer_starts(o.layer_split, file, (int) blocks->u - (nextn ? (int) nextn->u : 0), cards);
        ck(cudaSetDevice(was), "cudaSetDevice");
    }
    if (starts.empty()) {
        std::unique_ptr<Engine> engine = make();
        engine->load(o);
        return engine;
    }

    // a card after another, each with the next run of layers; the last one has the output and the draft block
    std::unique_ptr<Engine> first;
    Engine* prev = nullptr;
    for (size_t k = 0; k < starts.size(); ++k) {
        std::unique_ptr<Engine> e = make();
        const bool last = k + 1 == starts.size();
        Options so = o;
        if (!last) {
            so.draft = false;
            so.draft_model.clear();
        }
        if (first) {
            so.window = first->opt_.window;   // every card's windows are the first one's
            e->workers_ = first->workers_;
        }
        e->lo_ = starts[k];
        e->end_ = last ? -1 : starts[k + 1];
        e->split_ = true;
        ck(cudaSetDevice((int) k), "cudaSetDevice");
        try {
            e->load(so);
        } catch (const std::runtime_error& err) {
            throw std::runtime_error("card " + std::to_string(k) + ": " + err.what());
        }
        Engine* at = e.get();
        if (prev) prev->next_ = std::move(e);
        else first = std::move(e);
        prev = at;
    }
    ck(cudaSetDevice(0), "cudaSetDevice");
    int k = 0;
    for (const Engine* e = first.get(); e; e = e->next_.get(), ++k)
        std::fprintf(stderr, "strata-poly: layers %d to %d on card %d: %d of %d routed experts in VRAM (%.1f GiB)\n",
                     e->lo_, e->hi_ - 1, k, e->cached_, e->n_moe_ * e->n_expert_, double(e->exp_at_) / double(1ull << 30));
    return first;
}

// The file a tensor is in.
const Shards& Engine::home(const TensorInfo& t) const {
    return block_ && gg_->find(t.name) != &t ? *block_ : *gg_;
}

// A tensor to VRAM.  It goes as the file has it when the kernels compute with that type.  Otherwise it is
// converted on the way: to F32 where a kernel reads floats, to BF16 for a matrix.  `fix` is applied to every
// value, which also takes the float path.  While dense_ is null the call only adds up the arena (load() runs the
// model's tensors through twice).
Dev Engine::upload(const TensorInfo* t, const std::string& name, int64_t n_in, int64_t n_out, Want want,
                   float (*fix)(float)) {
    if (!t) throw std::runtime_error("the GGUF has no tensor " + name);
    const uint64_t rest = t->elements() / std::max<uint64_t>(t->shape[0], 1);
    if ((int64_t) t->shape[0] != n_in || (int64_t) rest != n_out)
        throw std::runtime_error(t->name + " is " + shape_of(*t) + ", expected [" + std::to_string(n_in) + ", " +
                                 std::to_string(n_out) + "]");
    const size_t bytes = home(*t).payload(*t), count = (size_t) t->elements();
    if (want == kGuessMat && !fix && n_in % ggml_blck_size(GGML_TYPE_Q4_0) == 0 &&
        ggml_get_type_traits((ggml_type) t->type)->to_float) {
        const size_t row = ggml_row_size(GGML_TYPE_Q4_0, n_in), out = row * (size_t) n_out;
        const Dev d{dense_ ? dense_ + dense_at_ : nullptr, (uint32_t) GGML_TYPE_Q4_0, (int) n_in, (int) n_out};
        dense_at_ += pad(out) + kTail;
        if (!dense_) return d;
        if (dense_at_ > dense_bytes_) throw std::logic_error("the dense arena is too small for " + t->name);
        // a few thousand rows at a time, to floats and to Q4_0, on threads of their own: the engine's are not
        // started yet
        const size_t in_row = bytes / (size_t) n_out, rows = 4096;
        std::vector<uint8_t> q(out);
        std::vector<float> f(rows * (size_t) n_in);
        const ggml_to_float_t to_float = ggml_get_type_traits((ggml_type) t->type)->to_float;
        const uint8_t* src = (const uint8_t*) home(*t).data(*t);
        Workers pool(std::max(1u, std::thread::hardware_concurrency() / 2));
        for (size_t r0 = 0; r0 < (size_t) n_out; r0 += rows) {
            const size_t m = std::min(rows, (size_t) n_out - r0);
            pool.run((int) ((m + 255) / 256), [&](int k) {
                const size_t a = (size_t) k * 256, b = std::min(m, a + 256);
                to_float(src + (r0 + a) * in_row, f.data() + a * n_in, (int64_t) ((b - a) * n_in));
                ggml_quantize_chunk(GGML_TYPE_Q4_0, f.data() + a * n_in, q.data() + (r0 + a) * row, 0,
                                    (int64_t) (b - a), n_in, nullptr);
            });
        }
        ck(cudaMemcpy(d.p, q.data(), out, cudaMemcpyHostToDevice), "weight upload");
        return d;
    }
    const bool as_is = !fix && (t->type == kTypeF32 || (want != kF32 && t->type == kTypeBF16) ||
                                (want == kAnyMat && kernels::native_mmvq_supported((int) t->type)));
    const uint32_t type = as_is ? t->type : want == kF32 ? kTypeF32 : kTypeBF16;
    const size_t out = as_is ? bytes : count * (type == kTypeF32 ? sizeof(float) : sizeof(uint16_t));
    if (type == kTypeBF16) bf16_max_ = std::max(bf16_max_, count);
    const Dev d{dense_ ? dense_ + dense_at_ : nullptr, type, (int) n_in, (int) n_out};
    dense_at_ += pad(out) + kTail;
    if (!dense_) return d;
    if (dense_at_ > dense_bytes_) throw std::logic_error("the dense arena is too small for " + t->name);
    if (as_is) {
        ck(cudaMemcpy(d.p, home(*t).data(*t), bytes, cudaMemcpyHostToDevice), "weight upload");
        return d;
    }

    std::vector<float> f(count);
    const ggml_type_traits* traits = ggml_get_type_traits((ggml_type) t->type);
    if (t->type == kTypeF32) std::memcpy(f.data(), home(*t).data(*t), bytes);
    else if (traits && traits->to_float) traits->to_float(home(*t).data(*t), f.data(), (int64_t) count);
    else throw std::runtime_error(t->name + " is " + shape_of(*t) + ", a type this engine cannot read");
    if (fix)
        for (float& v : f) v = fix(v);
    if (type == kTypeF32) {
        ck(cudaMemcpy(d.p, f.data(), out, cudaMemcpyHostToDevice), "weight upload");
        return d;
    }
    std::vector<ggml_bf16_t> h(count);
    ggml_fp32_to_bf16_row(f.data(), h.data(), (int64_t) count);
    ck(cudaMemcpy(d.p, h.data(), out, cudaMemcpyHostToDevice), "weight upload");
    return d;
}

int64_t Engine::vram_free_mib() const {
    OnCard here(device_, split_);
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    const int64_t mib = (int64_t) (free_b >> 20);
    return next_ ? std::min(mib, next_->vram_free_mib()) : mib;
}

// Experts go to VRAM by use per byte when there are counts, the profile's and this run's, and otherwise the same
// number from every layer.
//
// The cache takes what is free but for a reserve.  What the engine allocates after it comes out of the reserve:
// the recorded launches and a library's work space took 180 MiB of an 8 GB card in the first request.  Unless the
// caller names it, the reserve is a thirty-second of the card, between 256 and 768 MiB (mind_vram() keeps what is
// free near half of it later), or a sixteenth when the machine's RAM is within 4 GiB of the model's files
// (default_reserve_mib()).  Under Windows a card that is asked for more than it has does not refuse: memory
// moves to the system's, and an answer came at 13 tokens/s where it comes at 30.
void Engine::fill_expert_cache() {
    if (!opt_.gpu_experts || n_moe_ == 0) return;
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    const size_t ram = ram_available(), files = gg_->bytes() + (dg_ ? dg_->bytes() : 0);
    ram_tight_ = ram > 0 && files + (4ULL << 30) > ram;
    // 256 MiB at least, whatever was asked: the launches recorded later take theirs out of the reserve (mind_vram())
    reserve_mib_ = opt_.vram_reserve_mib >= 0 ? std::max(opt_.vram_reserve_mib, 256)
                                             : default_reserve_mib(total_b, ram_tight_);
    const size_t reserve = size_t(reserve_mib_) << 20;
    if (free_b <= reserve) return;

    struct Cand {
        int il, e;
        size_t bytes;
        double value;
    };
    std::vector<Cand> cands;
    std::vector<size_t> sizes;   // the slot sizes there are; a layer's place among them is its trade number
    size_t all = 0;
    for (int il = 0; il < (int) layers_.size(); ++il) {
        Layer& L = layers_[(size_t) il];
        if (!L.moe || !kernels::native_mmvq_supported((int) L.gu_type) ||
            !kernels::native_mmvq_supported((int) L.d_type))
            continue;
        // A slot is a whole number of blocks of both of the experts' types, so the slots of a layer that lie one
        // behind the other are the matrices of one launch (moe_rows()), and of 8 bytes, which keeps every block
        // where its type may be read.
        const size_t unit = std::lcm(std::lcm(ggml_type_size((ggml_type) L.gu_type),
                                              ggml_type_size((ggml_type) L.d_type)),
                                     size_t(8));
        const size_t bytes = (2 * L.gu_bytes + L.d_bytes + kTail + unit - 1) / unit * unit;
        L.slot_bytes = bytes;
        if (std::find(sizes.begin(), sizes.end(), bytes) == sizes.end()) sizes.push_back(bytes);
        L.trade = (int) (std::find(sizes.begin(), sizes.end(), bytes) - sizes.begin());
        slot_max_ = std::max(slot_max_, bytes);
        for (int e = 0; e < n_expert_; ++e) {
            const size_t i = (size_t) il * n_expert_ + e;
            const double used = (double) ((prior_.empty() ? 0 : prior_[i]) + use_[i]);
            cands.push_back({il, e, bytes, used / (double) bytes});
            all += bytes;
        }
    }
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Cand& a, const Cand& b) { return a.value != b.value ? a.value > b.value : a.e < b.e; });

    n_trade_ = (int) sizes.size();
    // sized again (resize_cache()), it is no larger than load() made it, or than its addresses with VMM
    const size_t most = vmm_ ? std::min(all, total_b) : full_bytes_ > 0 ? full_bytes_ : all;
    size_t arena = std::min({all, free_b - reserve, arena_.base ? arena_.reserved - 256 : most});
    if (arena == 0) return;
    if (vmm_) {   // in pieces, so that it can give some up to the attention caches (shrink_cache()) and grow
        if (!arena_.base) vmm_reserve(arena_, most + 256);   // addresses only: grow_cache() maps more of them
        try {
            vmm_map(arena_, arena + 256, kArenaPiece, "the expert cache");
        } catch (const std::runtime_error&) {   // the card may not have it all (grow_cache() is just as tolerant)
            arena = arena_.mapped > 256 ? arena_.mapped - 256 : 0;
            if (arena == 0) return;
        }
        exp_ = (uint8_t*) arena_.base;
        budget_ = free_b - reserve;   // when it holds every expert, what is left over goes to the rows first
        for (const Kv& k : kv_) budget_ += k.mapped;
    } else {
        exp_ = (uint8_t*) dmalloc(arena + 256, "the expert cache");
    }
    ck(cudaMemset(exp_, 0, arena + 256), "the expert cache");   // the slots' tails stay zero (kTail)
    // Which experts is by value; where they go is by layer, a layer's side by side in the experts' order.
    std::vector<Cand> taken;
    size_t room = arena;
    for (const Cand& c : cands) {
        if (c.bytes > room) continue;   // a smaller expert of another layer may still fit
        room -= c.bytes;
        taken.push_back(c);
    }
    std::sort(taken.begin(), taken.end(),
              [](const Cand& a, const Cand& b) { return a.il != b.il ? a.il < b.il : a.e < b.e; });
    for (const Cand& c : taken) {
        Layer& L = layers_[(size_t) c.il];
        uint8_t* s = exp_ + exp_at_;
        exp_at_ += c.bytes;
        const size_t e = (size_t) c.e;
        ck(cudaMemcpy(s, L.exp_gate + e * L.gu_stride, L.gu_bytes, cudaMemcpyHostToDevice), "expert upload");
        ck(cudaMemcpy(s + L.gu_bytes, L.exp_up + e * L.gu_stride, L.gu_bytes, cudaMemcpyHostToDevice),
           "expert upload");
        ck(cudaMemcpy(s + 2 * L.gu_bytes, L.exp_down + e * L.d_bytes, L.d_bytes, cudaMemcpyHostToDevice),
           "expert upload");
        L.slot[e] = s;
        ++cached_;
    }
    if (full_bytes_ == 0) {
        full_bytes_ = arena;
        full_cached_ = cached_;
    }
}

// The thread and the stream that bring experts into the cache while an answer is written (follow()).
void Engine::start_swaps() {
    if (swap_ || cached_ == 0 || opt_.expert_swaps == 0) return;
    swap_ = std::make_unique<Swap[]>((size_t) opt_.expert_swaps);
    cudaStream_t up = nullptr;
    ck(cudaStreamCreateWithFlags(&up, cudaStreamNonBlocking), "cudaStreamCreate");
    swap_st_ = up;
    for (; n_swap_ < opt_.expert_swaps; ++n_swap_) {
        Swap& s = swap_[(size_t) n_swap_];
        s.pinned = (uint8_t*) hmalloc(slot_max_, "pinned memory");
        ck(cudaEventCreateWithFlags(&s.done, cudaEventDisableTiming), "cudaEventCreate");
    }
    recent_.assign(use_.size(), 0.f);
    coming_.assign(use_.size(), 0);
    follower_ = std::thread([this] { follow_loop(); });
}

// The cache gives up what it has and is filled again for another reserve.  The experts on their way into it
// arrive first: their slots are about to go.
void Engine::resize_cache(int mib) {
    for (Engine* e = this; e; e = e->next_.get()) e->resize_card_cache(mib);
}

void Engine::resize_card_cache(int mib) {
    if (!opt_.gpu_experts || n_moe_ == 0) return;
    OnCard here(device_, split_);
    ck(cudaStreamSynchronize(cs(st_)), "resize");
    const auto failed = [&] {
        std::lock_guard<std::mutex> l(swap_m_);
        return !swap_error_.empty();
    };
    for (int j = 0; j < n_swap_; ++j)   // one that is asked for is the follower's, now or in a moment
        while (swap_[(size_t) j].at.load() == Swap::kAsked && !failed()) std::this_thread::yield();
    if (failed()) throw std::runtime_error("expert upload: " + swap_error_);
    if (swap_st_) ck(cudaStreamSynchronize(cs(swap_st_)), "resize");
    for (int j = 0; j < n_swap_; ++j) swap_[(size_t) j].at.store(Swap::kFree);
    std::fill(coming_.begin(), coming_.end(), (uint8_t) 0);
    for (Layer& L : layers_) std::fill(L.slot.begin(), L.slot.end(), nullptr);
    staged_il_ = -1;
    cached_ = 0;
    exp_at_ = 0;
    if (exp_ && vmm_) {
        vmm_unmap(arena_, 0);
        exp_ = nullptr;
    } else if (exp_) {
        dev_allocs_.erase(std::find(dev_allocs_.begin(), dev_allocs_.end(), (void*) exp_));
        ck(cudaFree(exp_), "resize");
        exp_ = nullptr;
    }
    opt_.vram_reserve_mib = mib < 0 ? asked_reserve_mib_ : mib;
    fill_expert_cache();
    start_swaps();
}

// The launches the engine records as it runs take VRAM after the cache was sized: 180 to 200 MiB of an 8 GB card,
// most of it in the first request.  That is what the reserve is for.  From the second request on the cache keeps
// `keep` free: the reserve the caller named, else half the one it was filled for and 128 MiB at least, or the
// whole of it when the PC's RAM is tight for the model (default_reserve_mib(): with the reserve named, the laptop
// read as the release did; with half of it kept, 7% to 14% slower).  It gives back what another program took, its
// least used experts, and takes VRAM that came free again for the most used experts it lacks.  Filled again
// whole it went down with every program that held VRAM a moment, and never came back up.
bool Engine::mind_vram() {
    bool changed = false;
    for (Engine* e = this; e; e = e->next_.get()) changed = e->mind_card_vram() || changed;
    return changed;
}

bool Engine::mind_card_vram() {
    if (!exp_ || reserve_mib_ <= 0) return false;
    OnCard here(device_, split_);
    size_t free_b = 0, total_b = 0;
    ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
    const int64_t free = (int64_t) (free_b >> 20);   // this card's
    if (!vmm_) {   // a cache in one allocation is filled again
        if (free >= reserve_mib_ / 4) return false;
        resize_card_cache(opt_.vram_reserve_mib);
        return true;
    }
    if (minds_++ == 0) return false;   // nothing is recorded before the first request
    const int64_t keep = opt_.vram_reserve_mib >= 0 || ram_tight_ ? reserve_mib_ : std::max(reserve_mib_ / 2, 128);
    if (free < keep / 2) {
        shrink_cache(size_t(keep - free) << 20);
        held_is_budget();
        return true;
    }
    return free >= keep + (int64_t) (kArenaPiece >> 20) && grow_cache(size_t(free - keep) << 20) > 0;
}

// The cache's VRAM and the attention caches' as they are now become the budget cover() keeps them to.
void Engine::held_is_budget() {
    budget_ = arena_.mapped;
    for (const Kv& k : kv_) budget_ += k.mapped;
}

// `bytes` more VRAM at the cache's end, for the experts it lacks that are used most (use per byte, as
// fill_expert_cache() chooses), side by side by layer.  VRAM the card does not give is not an error: the cache
// keeps what was mapped.  Returns the experts it took.
int Engine::grow_cache(size_t bytes) {
    if (!vmm_ || !exp_) return 0;
    settle_swaps();
    ck(cudaStreamSynchronize(cs(st_)), "the expert cache");
    if (swap_st_) ck(cudaStreamSynchronize(cs(swap_st_)), "the expert cache");
    struct Cand {
        int il, e;
        double value;
    };
    std::vector<Cand> cands;
    for (int il = 0; il < (int) layers_.size(); ++il) {
        const Layer& L = layers_[(size_t) il];
        if (L.trade < 0) continue;
        for (int e = 0; e < n_expert_; ++e) {
            const size_t i = (size_t) il * n_expert_ + e;
            if (L.slot[(size_t) e] || (!coming_.empty() && coming_[i])) continue;
            const double used = (double) ((prior_.empty() ? 0 : prior_[i]) + use_[i]);
            cands.push_back({il, e, used / (double) L.slot_bytes});
        }
    }
    if (cands.empty()) return 0;
    try {
        vmm_map(arena_, std::min(arena_.mapped + bytes, arena_.reserved), kArenaPiece, "the expert cache");
    } catch (const std::runtime_error&) {
    }
    if (arena_.mapped < exp_at_ + kTail) return 0;
    ck(cudaMemset(exp_ + exp_at_, 0, arena_.mapped - exp_at_), "the expert cache");   // the new slots' tails
    std::stable_sort(cands.begin(), cands.end(),
                     [](const Cand& a, const Cand& b) { return a.value != b.value ? a.value > b.value : a.e < b.e; });
    std::vector<Cand> taken;
    size_t room = arena_.mapped - kTail - exp_at_;
    for (const Cand& c : cands) {
        const size_t n = layers_[(size_t) c.il].slot_bytes;
        if (n > room) continue;
        room -= n;
        taken.push_back(c);
    }
    std::sort(taken.begin(), taken.end(),
              [](const Cand& a, const Cand& b) { return a.il != b.il ? a.il < b.il : a.e < b.e; });
    for (const Cand& c : taken) {
        Layer& L = layers_[(size_t) c.il];
        uint8_t* s = exp_ + exp_at_;
        exp_at_ += L.slot_bytes;
        const size_t e = (size_t) c.e;
        ck(cudaMemcpy(s, L.exp_gate + e * L.gu_stride, L.gu_bytes, cudaMemcpyHostToDevice), "expert upload");
        ck(cudaMemcpy(s + L.gu_bytes, L.exp_up + e * L.gu_stride, L.gu_bytes, cudaMemcpyHostToDevice),
           "expert upload");
        ck(cudaMemcpy(s + 2 * L.gu_bytes, L.exp_down + e * L.d_bytes, L.d_bytes, cudaMemcpyHostToDevice),
           "expert upload");
        L.slot[e] = s;
        ++cached_;
    }
    staged_il_ = -1;
    held_is_budget();
    return (int) taken.size();
}

// What a layer copies to the staging buffer ahead of its routing: the experts outside the cache, the most used
// first, as many as the buffer holds.  Worked out when the layer runs: the cache changes (follow()).
void Engine::plan_ahead(Layer& L, int il) {
    L.ahead.clear();
    for (int e = 0; e < n_expert_; ++e)
        if (!L.slot[(size_t) e]) L.ahead.push_back(e);
    if (!prior_.empty()) {
        const int64_t* used = prior_.data() + (size_t) il * n_expert_;
        std::stable_sort(L.ahead.begin(), L.ahead.end(), [used](int32_t a, int32_t b) { return used[a] > used[b]; });
    }
    L.ahead.resize(std::min(L.ahead.size(), stage_bytes_ / (2 * L.gu_bytes + L.d_bytes)));
}

// The follower: an asked-for expert goes from the mapped file to its pinned buffer, and from there to its slot.
// The file may have to be read for it, which is why this is not the engine's thread.
void Engine::follow_loop() {
    cudaSetDevice(device_);   // a thread starts on card 0
    std::unique_lock<std::mutex> lk(swap_m_);
    for (;;) {
        Swap* s = nullptr;
        swap_cv_.wait(lk, [&] {
            for (int j = 0; j < n_swap_ && !s; ++j)
                if (swap_[(size_t) j].at.load() == Swap::kAsked) s = &swap_[(size_t) j];
            return swap_quit_ || s;
        });
        if (swap_quit_) return;
        lk.unlock();
        const Layer& L = layers_[(size_t) s->il];
        const size_t e = (size_t) s->e, gu = L.gu_bytes;
        std::memcpy(s->pinned, L.exp_gate + e * L.gu_stride, gu);
        std::memcpy(s->pinned + gu, L.exp_up + e * L.gu_stride, gu);
        std::memcpy(s->pinned + 2 * gu, L.exp_down + e * L.d_bytes, L.d_bytes);
        cudaError_t err = cudaMemcpyAsync(s->slot, s->pinned, 2 * gu + L.d_bytes, cudaMemcpyHostToDevice, cs(swap_st_));
        if (err == cudaSuccess) err = cudaEventRecord(s->done, cs(swap_st_));
        lk.lock();
        if (err != cudaSuccess) {
            swap_error_ = cudaGetErrorString(err);
            return;
        }
        s->at.store(Swap::kSent);
    }
}

// The cache follows what is written.  Called at the start of an eval, where nothing queued reads the cache, with
// the number of tokens the eval writes (0 for a prompt batch).  The experts whose upload has arrived get their
// slots.  Then, for written tokens, the experts used lately that the cache lacks start on their way, each to the
// slot of the cached expert of its size that was used least (among equals, the one the profile ranks last).
//
// A prompt batch neither counts nor moves anything.  What a prompt routes to says no more about the answer than
// the profile does, and moving the cache while the prompt was read cost 3% of the reading speed (measured).
void Engine::follow(int written) {
    {
        std::lock_guard<std::mutex> l(swap_m_);
        if (!swap_error_.empty()) throw std::runtime_error("expert upload: " + swap_error_);
    }
    const size_t ne = (size_t) n_expert_;
    int n_free = 0;
    for (int j = 0; j < n_swap_; ++j) {
        Swap& s = swap_[(size_t) j];
        if (s.at.load() == Swap::kSent) {
            const cudaError_t arrived = cudaEventQuery(s.done);
            if (arrived == cudaErrorNotReady) continue;
            ck(arrived, "expert upload");
            layers_[(size_t) s.il].slot[(size_t) s.e] = s.slot;
            coming_[(size_t) s.il * ne + (size_t) s.e] = 0;
            s.at.store(Swap::kFree);
            ++stats_.swapped_experts;
            staged_il_ = -1;   // the staging buffer holds what was not cached
        }
        n_free += s.at.load() == Swap::kFree;
    }
    if (written <= 0) return;
    const float keep = std::exp2(-(float) written / (float) kSwapHalf);
    for (float& r : recent_) r *= keep;
    if (n_free == 0) return;

    // One pass over the experts lists, for each slot size, the ones that may come in, the most used first, and
    // the cached ones that may go, the least used first: as many of each as uploads are free, which is all the
    // pairs below can take.  A pass for every upload was 0.6 ms of a window of GLM 5.3 Flash's.
    const size_t most = (size_t) n_free;
    in_.assign((size_t) n_trade_ * most, Pick{});
    out_.assign((size_t) n_trade_ * most, Pick{});
    // p to its place in a list, whose last one drops out; equals stay in the order they came
    const auto put = [most](Pick* list, const Pick& p, bool (*before)(const Pick&, const Pick&)) {
        if (list[most - 1].il >= 0 && !before(p, list[most - 1])) return;
        size_t at = most - 1;
        for (; at > 0 && (list[at - 1].il < 0 || before(p, list[at - 1])); --at) list[at] = list[at - 1];
        list[at] = p;
    };
    const auto more = [](const Pick& a, const Pick& b) {
        return a.recent != b.recent ? a.recent > b.recent : a.prior > b.prior;
    };
    const auto less = [](const Pick& a, const Pick& b) {
        return a.recent != b.recent ? a.recent < b.recent : a.prior < b.prior;
    };
    for (int il = 0; il < (int) layers_.size(); ++il) {
        const Layer& L = layers_[(size_t) il];
        if (L.trade < 0) continue;
        Pick* in = in_.data() + (size_t) L.trade * most;
        Pick* out = out_.data() + (size_t) L.trade * most;
        for (int e = 0; e < n_expert_; ++e) {
            const size_t i = (size_t) il * ne + (size_t) e;
            const float r = recent_[i];
            const int64_t p = prior_.empty() ? 0 : prior_[i];
            if (L.slot[(size_t) e]) put(out, {il, e, r, p}, less);
            else if (r >= kSwapMin && !coming_[i]) put(in, {il, e, r, p}, more);
        }
    }

    // An upload goes to the size whose next pair gains most.  A cached expert that has given its slot is no
    // candidate to come in: it was used no more than the cached ones left.
    trade_at_.assign((size_t) n_trade_, 0);
    bool asked = false;
    for (; n_free > 0; --n_free) {
        int best = -1;
        float gain = 0.f;
        for (int c = 0; c < n_trade_; ++c) {
            const size_t k = (size_t) c * most + trade_at_[(size_t) c];
            const Pick& in = in_[k];
            const Pick& out = out_[k];
            if (in.il < 0 || out.il < 0 || in.recent < 2.f * out.recent + kSwapMin) continue;
            if (best < 0 || in.recent - out.recent > gain) {
                best = c;
                gain = in.recent - out.recent;
            }
        }
        if (best < 0) break;

        const size_t k = (size_t) best * most + trade_at_[(size_t) best]++;
        const Pick in = in_[k], out = out_[k];
        Swap* s = nullptr;
        for (int j = 0; j < n_swap_ && !s; ++j)
            if (swap_[(size_t) j].at.load() == Swap::kFree) s = &swap_[(size_t) j];
        uint8_t*& given = layers_[(size_t) out.il].slot[(size_t) out.e];
        s->il = in.il;
        s->e = in.e;
        s->slot = given;
        given = nullptr;
        staged_il_ = -1;
        coming_[(size_t) in.il * ne + (size_t) in.e] = 1;
        s->at.store(Swap::kAsked);
        asked = true;
    }
    if (asked) {
        { std::lock_guard<std::mutex> l(swap_m_); }
        swap_cv_.notify_one();
    }
}

void Engine::load(const Options& o) {
    opt_ = o;
    asked_reserve_mib_ = o.vram_reserve_mib;
    if (opt_.max_context < 16) throw std::runtime_error("the context must be at least 16 tokens");
    if (opt_.prompt_batch < 1 || opt_.prompt_batch > 4096)
        throw std::runtime_error("a prompt batch is 1 to 4096 tokens");
    if (opt_.stage_mib < 1) throw std::runtime_error("the staging buffer must be at least 1 MiB");
    if (opt_.stage_ahead < 0) throw std::runtime_error("the batch size to stage ahead from cannot be negative");
    if (opt_.expert_swaps < 0 || opt_.expert_swaps > 64)
        throw std::runtime_error("0 to 64 experts can be on their way into the cache at a time");
    if (opt_.window < 0 || opt_.window > 8)
        throw std::runtime_error("a window is 1 to 8 tokens, or 0 for the model's own");
    if (opt_.tail_skip < 0) throw std::runtime_error("--tail-skip (STRATA_ROUTE_TAIL_SKIP) cannot be negative");
    if (opt_.dump || std::min(opt_.prompt_batch, opt_.max_context) == 1) opt_.window = 1;
    gg_ = std::make_unique<Shards>(opt_.model);
    if (!opt_.draft_model.empty() && opt_.draft && opt_.window != 1) dg_ = std::make_unique<Shards>(opt_.draft_model);
    load_geometry();
    if (block_ != dg_.get()) dg_.reset();
    hi_ = end_ < 0 ? n_layer_ : end_;

    cudaStream_t st = nullptr;
    ck(cudaStreamCreateWithFlags(&st, cudaStreamNonBlocking), "cudaStreamCreate");
    st_ = st;
    for (void*& row : ev_row_) {
        cudaEvent_t ev = nullptr;
        ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        row = ev;
    }
    ev_ = ev_row_[0];
    blas_ = blas_create(st_);
    // The card maps VRAM onto reserved addresses a piece at a time: the attention caches grow with the sequence,
    // and the expert cache has what a short one leaves (cover()).
    ck(cudaGetDevice(&device_), "cudaGetDevice");
    CUdevice dev = 0;
    int vmm = 0;
    if (opt_.kv_grow && cuDeviceGet(&dev, device_) == CUDA_SUCCESS &&
        cuDeviceGetAttribute(&vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, dev) == CUDA_SUCCESS && vmm) {
        CUmemAllocationProp prop = {};
        prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
        prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
        prop.location.id = device_;
        vmm_ = cuMemGetAllocationGranularity(&grain_, &prop, CU_MEM_ALLOC_GRANULARITY_MINIMUM) == CUDA_SUCCESS &&
               grain_ > 0;
    }

    // the token table stays in the file, a row a token, and embed() reads a row in whatever type it has
    const TensorInfo* te = gg_->find("token_embd.weight");
    const ggml_type_traits* tt = te ? ggml_get_type_traits((ggml_type) te->type) : nullptr;
    if (!te || te->shape.size() != 2 || (int64_t) te->shape[0] != n_embd_ || (int64_t) te->shape[1] != n_vocab_ ||
        !tt || tt->blck_size <= 0 || n_embd_ % tt->blck_size != 0 || (te->type != kTypeF32 && !tt->to_float))
        throw std::runtime_error("token_embd.weight must be [" + std::to_string(n_embd_) + ", " +
                                 std::to_string(n_vocab_) + "] in a type ggml converts to floats, it is " +
                                 (te ? shape_of(*te) : "missing"));
    tok_row_ = gg_->payload(*te) / (size_t) n_vocab_;
    tok_type_ = te->type;
    tok_embd_ = gg_->data(*te);

    // twice: the first pass adds up the arena, the second fills it
    for (int pass = 0; pass < 2; ++pass) {
        const int n_all = n_layer_ + (block_layer_ ? 1 : 0);
        layers_.assign((size_t) n_all, Layer{});
        n_moe_ = 0;
        dense_at_ = 0;
        begin_pass();
        for (int il = 0; il < n_all; ++il)
            if (runs(il)) load_layer(il);
        if (hi_ == n_layer_) {
            const TensorInfo* norm = gg_->find("output_norm.weight");
            if (const Names* a = norm ? nullptr : tensor_alias("output_norm.weight"))
                for (const char* other : a->other)
                    if (other && !norm) norm = gg_->find(other);
            out_norm_ = upload(norm, "output_norm.weight", n_embd_, 1, kF32);
            // a file without an output matrix ties the output to the token table
            const TensorInfo* head = gg_->find("output.weight");
            output_ = upload(head ? head : gg_->find("token_embd.weight"), "output.weight", n_embd_, n_vocab_,
                             kAnyMat);
        }
        if (draft_) load_drafter();
        if (pass == 0) {
            dense_bytes_ = dense_at_;
            dense_ = (uint8_t*) dmalloc(dense_bytes_ + 256, "the dense weights");
            ck(cudaMemset(dense_, 0, dense_bytes_ + 256), "the dense weights");   // between the tensors: kTail
        }
    }

    // A guess needs the recurrent layers' state as it was before it.  Where that is a copy in VRAM for every
    // guess of a window, a window is four tokens: the copies come out of what the experts would have.  A model
    // without such layers pays nothing for a longer one, and one whose layers leave the state after a few of a
    // window's tokens (snaps_spread()) no more than for four.  A repeat of the conversation can fill a window
    // of eight.  The layers are counted by now.
    if (opt_.window == 0) opt_.window = recurrent_floats() == 0 || snaps_spread() ? 8 : 4;
    opt_.window = std::min({opt_.window, opt_.prompt_batch, opt_.max_context});
    alloc();
    if (!opt_.profile.empty()) read_profile(opt_.profile, (int) layers_.size(), n_expert_, prior_);
    fill_expert_cache();
    // A card of a split that has less free beside its layers than the reserve it keeps by default runs out of VRAM
    // in a later request, where one card alone can do nothing else.  It may be that another program held VRAM when
    // the split was made.  A larger reserve asked for (--vram-reserve-mib) only leaves the card without experts.
    if (split_ && !exp_) {
        size_t free_b = 0, total_b = 0;
        ck(cudaMemGetInfo(&free_b, &total_b), "cudaMemGetInfo");
        const int need = default_reserve_mib(total_b, ram_tight_);
        if (free_b <= (size_t(need) << 20))
            throw std::runtime_error("layers " + std::to_string(lo_) + " to " + std::to_string(hi_ - 1) + " leave " +
                                     std::to_string(free_b >> 20) + " MiB of VRAM free, less than the " +
                                     std::to_string(need) + " MiB the card needs beside them: give it fewer layers "
                                     "(--layer-split), or free VRAM another program holds");
    }
    // A card that holds every expert has VRAM to spare, and the attention caches take the whole context's rows
    // there now when they fit: VRAM left free is what another program takes, and the caches would then grow out of
    // the expert cache (cover()), which does not get it back when that program ends.  A piece of the expert cache
    // (kArenaPiece) is the margin, so that cover() cuts nothing.
    if (!kv_.empty() && all_cached()) {
        size_t more = 0;
        for (const Kv& k : kv_) more += k.reserved - k.mapped;
        if ((int64_t) ((more + kArenaPiece) >> 20) + reserve_mib_ <= vram_free_mib()) cover(opt_.max_context);
    }
    const int threads = opt_.threads > 0 ? opt_.threads : usable_cpus();
    mind_memory();
    std::vector<std::pair<const uint8_t*, size_t>> spans;   // the routed experts in the mapped files
    for (const Layer& L : layers_) {
        if (!L.moe) continue;
        spans.emplace_back(L.exp_gate, (size_t) n_expert_ * L.gu_stride);
        if (L.gu_stride == L.gu_bytes) spans.emplace_back(L.exp_up, (size_t) n_expert_ * L.gu_bytes);
        spans.emplace_back(L.exp_down, (size_t) n_expert_ * L.d_bytes);
    }
    if (!workers_) {   // a later card's engine has the first one's
        if (opt_.stay_near) {   // before any thread that reads the experts is started
            std::vector<std::string> files = gg_->paths();
            if (dg_) files.insert(files.end(), dg_->paths().begin(), dg_->paths().end());
            node_ = stay_near(spans, files, threads, node_share_);
        }
        workers_ = std::make_shared<Workers>(threads);
    }
    route_tmp_.assign((size_t) workers_->threads() * 2 * n_expert_, 0.f);
    start_swaps();
    reset();
    // nothing is read from the files when every expert is in VRAM
    if (!all_cached())
        mapper_ = std::thread([this, spans = std::move(spans)] { map_pages(spans, map_quit_); });
}

void Engine::load_drafter() {
    check_draft_block();
    auto part = [&](const char* n, int64_t a, int64_t b, Want w) {
        return upload(tensor(n_layer_, n), blk(n_layer_, n), a, b, w);
    };
    d_proj_ = part("nextn.eh_proj.weight", 2 * n_embd_, n_embd_, kAnyMat);
    d_enorm_ = part("nextn.enorm.weight", n_embd_, 1, kF32);
    d_hnorm_ = part("nextn.hnorm.weight", n_embd_, 1, kF32);
    // without a norm of its own, the block's output goes through the model's
    d_head_ = tensor(n_layer_, "nextn.shared_head_norm.weight")
                  ? part("nextn.shared_head_norm.weight", n_embd_, 1, kF32)
                  : out_norm_;
}

// Per-token buffers hold rows_ tokens, row after row.
void Engine::shared_scratch() {
    const size_t n = (size_t) rows_, e = n * n_embd_;
    const size_t ff = n * std::max({n_ff_, n_ff_shexp_, 2 * n_ff_exp_});
    // the draft block works in it too (draft_input()); read() keeps two calls' rows in it
    r_ = fbuf(std::max(streams(), draft_ ? 3 : 1) * (size_t) (read_rows_ > rows_ ? 2 : 1) * rows_ * n_embd_);
    cur_ = fbuf(e);
    xn_ = fbuf(e);
    ao_ = fbuf(e);
    fo_ = fbuf(e);
    g_ = fbuf(ff);   // a routed expert's gate and up rows come out of one launch, back to back
    u_ = fbuf(ff);
    h_ = fbuf(ff);
    d_ = fbuf((size_t) kSumRows * n_embd_);   // the routed experts of a few tokens, until they are added up
    rl_ = fbuf(n * n_expert_);
    logits_ = fbuf((size_t) n_vocab_);
    if (rows_ >= kStageRows && bf16_max_ > 0) wf_ = fbuf(bf16_max_);   // a BF16 matrix as floats, for lin()
    if (mmq_) {
        const size_t rr = n * n_used_;
        gu_ = fbuf(rr * 2 * n_ff_exp_);
        hx_ = fbuf(rr * n_ff_exp_);
        dx_ = fbuf(rr * n_embd_);
    }
}

// The sequence's state, the caches and every buffer, once the weights are known.
void Engine::alloc() {
    // --dump-layers prints one token's stages
    rows_ = opt_.dump ? 1 : std::min(opt_.prompt_batch, opt_.max_context);
    const size_t most = std::min<size_t>(kReadBytes / ((size_t) streams() * n_embd_ * sizeof(float)), opt_.max_context);
    read_rows_ = rows_ < kStageRows ? rows_ : std::max(rows_, (int) most / rows_ * rows_);
    rec_n_ = recurrent_floats();
    state_n_ = rec_n_ + (draft_ ? (size_t) n_embd_ : 0);
    state_ = (float*) dmalloc((state_n_ + (draft_ ? (size_t) rows_ * n_embd_ : 0)) * sizeof(float),
                              "the recurrent state");
    if (draft_) {
        hn_ = state_ + rec_n_;
        h_guess_ = (Likeliest*) hmalloc(sizeof(Likeliest), "pinned memory");   // likeliest() writes it
    }
    // window(): the state after tokens of it, for keep(), and every token's logits
    snap_places_ = snaps_spread() ? std::min(opt_.window - 1, kSnapPlaces) : opt_.window - 1;
    if (snap_places_ > 0)
        snap_ = (float*) dmalloc(size_t(snap_places_) * rec_n_ * sizeof(float), "the recurrent state");
    wlogits_ = (float*) dmalloc(size_t(opt_.window) * n_vocab_ * sizeof(float), "scratch");
    h_wlogits_ = (float*) hmalloc(size_t(opt_.window) * n_vocab_ * sizeof(float), "pinned memory");
    alloc_caches();
    cover(1);   // a short conversation's rows, before the expert cache is sized

    // A prompt batch runs a layer's routed experts through MMQ when the card has a tile for each of their types
    // (batched()).  The model's layers: the draft block's experts never run a batch at a time.
    const size_t n = (size_t) rows_;
    bool tiles = false;
    size_t one_expert = 0;
    for (int il = 0; il < n_layer_; ++il) {
        Layer& L = layers_[(size_t) il];
        if (!L.moe) continue;
        L.mmq = opt_.gpu_experts && rows_ >= kStageRows && Mmq::fits((int) L.gu_type, 2 * n_ff_exp_, n_embd_) &&
                Mmq::fits((int) L.d_type, n_embd_, n_ff_exp_);
        tiles = tiles || L.mmq;
        one_expert = std::max(one_expert, 2 * L.gu_bytes + L.d_bytes);
    }
    if (tiles) mmq_ = std::make_unique<Mmq>();

    shared_scratch();   // the first pass only counts
    assign_scratch();
    scratch_ = (float*) dmalloc(scratch_at_ * sizeof(float), "scratch");
    scratch_at_ = 0;
    shared_scratch();
    assign_scratch();

    const int q_max = std::max({n_embd_, n_ff_, n_ff_exp_, n_ff_shexp_, widest(), 256});
    xq_ = dmalloc(n * kernels::native_q8_1_bytes(n_embd_), "scratch");
    q8_ = dmalloc(n * kernels::native_q8_1_bytes((q_max + 255) / 256 * 256), "scratch");

    h_r_ = (float*) hmalloc(n * streams() * n_embd_ * sizeof(float), "pinned memory");
    at_ = (long long*) dmalloc((size_t) opt_.window * 2 * sizeof(long long), "scratch");
    h_at_ = (long long*) hmalloc((size_t) opt_.window * 2 * sizeof(long long), "pinned memory");
    h_rl_ = (float*) hmalloc(n * n_expert_ * sizeof(float), "pinned memory");
    h_x_ = (float*) hmalloc(n * n_embd_ * sizeof(float), "pinned memory");
    h_cpu_ = (float*) hmalloc(n * n_embd_ * sizeof(float), "pinned memory");
    h_logits_ = (float*) hmalloc((size_t) n_vocab_ * sizeof(float), "pinned memory");

    if (mmq_) {
        const size_t rr = n * n_used_,
                     tab = (3 * rr + 3 * (size_t) n_expert_ + 3 + kClasses * ((size_t) n_expert_ + 1)) * sizeof(int32_t);
        mq_ = dmalloc(Mmq::q8_bytes((int64_t) rr, n_embd_), "scratch");
        mq2_ = dmalloc(Mmq::q8_bytes((int64_t) rr, n_ff_exp_), "scratch");
        mqx_ = dmalloc(Mmq::q8_bytes(rows_, q_max), "scratch");
        d_ids_ = (int32_t*) dmalloc(rr * sizeof(int32_t), "scratch");
        d_wts_ = (float*) dmalloc(rr * sizeof(float), "scratch");
        h_ids_ = (int32_t*) hmalloc(rr * sizeof(int32_t), "pinned memory");
        d_slots_ = (void**) dmalloc((size_t) n_layer_ * n_expert_ * sizeof(void*), "scratch");
        tab_ = (int32_t*) dmalloc(tab, "scratch");
        h_tab_ = (int32_t*) hmalloc(tab, "pinned memory");
        iota_ = (int32_t*) dmalloc(rr * sizeof(int32_t), "scratch");
        std::vector<int32_t> ramp(rr);
        std::iota(ramp.begin(), ramp.end(), 0);
        ck(cudaMemcpy(iota_, ramp.data(), rr * sizeof(int32_t), cudaMemcpyHostToDevice), "scratch");

        // One layer's experts pass through it at a time, so it never holds more than a layer has.  A model of
        // small experts has a few hundred MiB a layer, and the rest of the default would be VRAM that the
        // expert cache of a small card does without.
        stage_bytes_ = std::max(std::min((size_t) opt_.stage_mib << 20, (size_t) n_expert_ * one_expert), one_expert);
        chunk_bytes_ = std::max(kChunkBytes, one_expert);
        stage_ = (uint8_t*) dmalloc(stage_bytes_ + 256 + kTail, "the expert staging buffer");
        for (int b = 0; b < 2; ++b) h_stage_[b] = (uint8_t*) hmalloc(chunk_bytes_, "pinned memory");
        for (void** e : {&ev_stage_[0], &ev_stage_[1], &ev_ahead_, &ev_free_}) {
            cudaEvent_t ev = nullptr;
            ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
            *e = ev;
        }
        cudaStream_t up = nullptr;
        ck(cudaStreamCreateWithFlags(&up, cudaStreamNonBlocking), "cudaStreamCreate");
        up_ = up;
        ids_.resize(rr);
        first_.resize((size_t) n_expert_);

        for (void** e : {&ev_in_[0], &ev_in_[1], &ev_out_[0], &ev_out_[1]}) {   // read()
            cudaEvent_t ev = nullptr;
            ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
            *e = ev;
        }
        cudaStream_t rs = nullptr;
        ck(cudaStreamCreateWithFlags(&rs, cudaStreamNonBlocking), "cudaStreamCreate");
        rs_ = rs;
    }
    use_.assign(layers_.size() * n_expert_, 0);
}

void Engine::save_profile() const {
    if (opt_.profile.empty()) return;
    const std::string tmp = opt_.profile + ".tmp";
    // every card's engine counts its own layers' experts, and the last one's profile has the draft block's too
    const Engine* last = this;
    while (last->next_) last = last->next_.get();
    std::vector<int64_t> all = last->prior_;
    for (const Engine* e = this; e; e = e->next_.get()) {
        if (all.size() < e->use_.size()) all.resize(e->use_.size(), 0);
        for (size_t i = 0; i < e->use_.size(); ++i) all[i] += e->use_[i];
    }
    FILE* f = std::fopen(tmp.c_str(), "w");
    if (!f) throw std::runtime_error("cannot write " + tmp);
    for (size_t i = 0; i < all.size(); ++i) {
        const int64_t n = all[i];
        if (n > 0)
            std::fprintf(f, "%d %d %lld\n", (int) (i / (size_t) n_expert_), (int) (i % (size_t) n_expert_),
                         (long long) n);
    }
    const bool written = std::fclose(f) == 0;
    std::error_code ec;
    if (written) std::filesystem::rename(tmp, opt_.profile, ec);   // std::rename replaces no file on Windows
    if (!written || ec) throw std::runtime_error("cannot write " + opt_.profile);
}

void Engine::reset() {
    OnCard here(device_, split_);
    ck(cudaMemsetAsync(state_, 0, state_n_ * sizeof(float), cs(st_)), "reset");
    pos_ = 0;
    ahead_pos_ = -1;
    if (next_) next_->reset();
}

void Engine::save_state(uint8_t* dst) {
    OnCard here(device_, split_);
    ck(cudaStreamSynchronize(cs(st_)), "save state");
    for (int il = 0; il < (int) layers_.size(); ++il)
        if (runs(il)) save_layer(il, pos_, dst);
    save_last(dst);
    if (next_) next_->save_state(dst + card_state_bytes());
}

// What the state has that is no layer's: the model's output for the last token, which the draft block pairs the
// next one with.  It lies behind the layers' state.
void Engine::save_last(uint8_t* dst) {
    if (!draft_) return;
    ck(cudaMemcpy(dst + rec_n_ * sizeof(float), hn_, (size_t) n_embd_ * sizeof(float), cudaMemcpyDeviceToHost),
       "save state");
}

void Engine::load_state(const uint8_t* src, int64_t pos) {
    if (pos < 0 || pos > opt_.max_context) throw std::logic_error("load_state: the position is outside the context");
    OnCard here(device_, split_);
    ck(cudaStreamSynchronize(cs(st_)), "load state");
    ck(cudaMemcpy(state_, src, state_n_ * sizeof(float), cudaMemcpyHostToDevice), "load state");
    load_extra(src + state_n_ * sizeof(float), pos);
    pos_ = pos;
    ahead_pos_ = -1;
    if (next_) next_->load_state(src + card_state_bytes(), pos);
}

void Engine::save_rows(int64_t first, int64_t n, uint8_t* dst) {
    OnCard here(device_, split_);
    save_card_rows(first, n, dst);
    if (next_) next_->save_rows(first, n, dst + (size_t) n * card_row_bytes());
}

void Engine::load_rows(const uint8_t* src, int64_t from, int64_t count, int64_t first, int64_t n) {
    OnCard here(device_, split_);
    load_card_rows(src, from, count, first, n);
    if (next_) next_->load_rows(src + (size_t) count * card_row_bytes(), from, count, first, n);
}

const Stats& Engine::stats() const {
    if (!next_) return stats_;
    all_stats_ = stats_;   // the tokens are counted once
    for (const Engine* e = next_.get(); e; e = e->next_.get()) {
        const Stats& s = e->stats_;
        all_stats_.gpu_experts += s.gpu_experts;
        all_stats_.cpu_experts += s.cpu_experts;
        all_stats_.batch_cached += s.batch_cached;
        all_stats_.batch_staged += s.batch_staged;
        all_stats_.swapped_experts += s.swapped_experts;
        all_stats_.tail_experts += s.tail_experts;
        all_stats_.cpu_ms += s.cpu_ms;
        all_stats_.route_wait_ms += s.route_wait_ms;
    }
    return all_stats_;
}

void Engine::embed(int32_t token, float* dst) const {
    const uint8_t* row = tok_embd_ + (size_t) token * tok_row_;
    if (tok_type_ == kTypeF32) std::memcpy(dst, row, (size_t) n_embd_ * sizeof(float));
    else ggml_get_type_traits((ggml_type) tok_type_)->to_float(row, dst, n_embd_);
}

void Engine::images(const Image* im, int n) {
    for (int i = 0; i < n; ++i) {
        const Image& a = im[i];
        if (a.n < 1 || a.nx < 1 || a.n % a.nx != 0 || !a.rows || a.at < (i ? im[i - 1].at + im[i - 1].n : 0))
            throw std::logic_error("images: a picture's rows are not a grid, or the pictures are out of order");
    }
    images_.assign(im, im + n);
    ++images_set_;
    if (next_) next_->images(im, n);   // positions and masks depend on where the pictures are
}

const Image* Engine::image_at(int64_t row) const {
    for (const Image& im : images_)
        if (row >= im.at && row < im.at + im.n) return &im;
    return nullptr;
}

// What the model reads at a row of the sequence: a picture's row there, else the token's embedding.  True for a
// picture's row, which goes in as the encoder made it (llama.cpp scales token embeddings only).
bool Engine::input(int32_t token, int64_t row, float* dst) const {
    if (const Image* im = image_at(row)) {
        std::memcpy(dst, im->rows + (size_t) (row - im->at) * n_embd_, (size_t) n_embd_ * sizeof(float));
        return true;
    }
    if (token < 0 || token >= n_vocab_) throw std::runtime_error("a token id is outside the vocabulary");
    embed(token, dst);
    return false;
}

void Engine::dbg(const char* name, int il, const float* dev, int n) {
    if (!opt_.dump) return;
    std::vector<float> h((size_t) n);
    ck(cudaStreamSynchronize(cs(st_)), "dump");
    ck(cudaMemcpy(h.data(), dev, (size_t) n * sizeof(float), cudaMemcpyDeviceToHost), "dump");
    double sum = 0, abs = 0;
    for (float v : h) {
        sum += v;
        abs += std::fabs(v);
    }
    std::fprintf(stderr, "dump pos=%lld blk=%d %-8s n=%d sum=%.6f abs=%.6f first=%.6f %.6f %.6f\n", (long long) pos_,
                 il, name, n, sum, abs, h[0], h[1], h[2]);
}

// y = w x for n tokens.  xq is x already in Q8_1, when the caller has it.  rows_per_x below n_out gives each group
// of rows its own input (see gemv_f32); only the float types take it.
void Engine::lin(const Dev& w, const float* x, float* y, int n, const void* xq, int rows_per_x) {
    if (w.type == kTypeF32 || w.type == kTypeBF16) {
        const int per = rows_per_x ? rows_per_x : w.n_out;
        const size_t count = (size_t) w.n_in * w.n_out;
        // A prompt batch: one matrix product.  The kernel below has a block for every output of every token,
        // and GLM 5.3 Flash's per-head projections are 32,768 outputs a token.
        if (n >= kStageRows && (w.type == kTypeF32 || (wf_ && count <= bf16_max_))) {
            const float* wf = fp(w);
            if (w.type == kTypeBF16) {
                bf16_to_f32((const uint16_t*) w.p, wf_, count, st_);
                wf = wf_;
            }
            return gemm_f32(blas_, wf, x, y, w.n_in, w.n_out, per, n);
        }
        if (w.type == kTypeF32) return gemv_f32(fp(w), x, y, w.n_in, w.n_out, per, n, st_);
        return gemv_bf16((const uint16_t*) w.p, x, y, w.n_in, w.n_out, per, n, st_);
    }
    // a prompt batch: llama.cpp's matrix-matrix kernels read the weights once for all the tokens
    if (mmq_ && n >= kStageRows && Mmq::fits((int) w.type, w.n_out, w.n_in)) {
        Mmq::quantize(x, nullptr, mqx_, (int) w.type, w.n_in, w.n_in, n, st_);
        return mmq_->mul(w.p, (int) w.type, w.n_out, w.n_in, 0, 1, mqx_, nullptr, nullptr, n, n, y, w.n_out, st_);
    }
    if (!xq) {
        // the tokens' blocks are back to back, so they quantize as one long vector
        kernels::native_quantize_q8_1(x, q8_, w.n_in * n, 1, st_);
        xq = q8_;
    }
    // a launch takes up to 8 tokens, and each comes out as it would alone
    const size_t token = kernels::native_q8_1_bytes(w.n_in);
    for (int at = 0; at < n; at += 8)
        kernels::native_mmvq((int) w.type, w.p, (const uint8_t*) xq + (size_t) at * token,
                             y + (size_t) at * w.n_out, w.n_in, w.n_out, std::min(8, n - at), st_);
}

void Engine::router_scores(const Layer& L, int n) { lin(L.gate_inp, xn_, rl_, n, xq_); }

bool Engine::batched(const Layer& L, int n) const { return mmq_ && L.mmq && n >= kStageRows; }

// The router's scores, and the copies of them and of the experts' input to the host.
void Engine::moe_router(const Layer& L, int n) {
    router_scores(L, n);
    copy2(rl_, h_rl_, n * n_expert_, xn_, h_x_, n * n_embd_, st_);
}

// The routed experts of tokens that come a few at a time.  moe_router() is on its way, with ev_ behind it.  True
// when experts were computed on the CPU: their sum waits in h_cpu_ for moe_add().  `first` is run once, by the
// thread that hands the cached experts to the card and ahead of them.
bool Engine::moe(Layer& L, int il, int n, const std::function<void()>* first) {
    const size_t e_row = (size_t) n_embd_;
    Clock::time_point t0 = Clock::now();
    ck(cudaEventSynchronize((cudaEvent_t) ev_), "router");
    stats_.route_wait_ms += ms_since(t0);

    const float limit = clamp_exp_[(size_t) il];
    const size_t xq_row = kernels::native_q8_1_bytes(n_embd_);
    bool any_host = false;
    // A window's tokens together: the CPU reads an expert that several of them chose once.
    for (int at = 0; at < n; at += kCpuRows) {
        const int m = std::min(kCpuRows, n - at);
        struct Cached {
            const uint8_t* slot;
            float w;
            int row;
        };
        CpuExpert host[kCpuRows * kMaxUsed];
        Cached cached[kCpuRows * kMaxUsed];
        int32_t host_e[kCpuRows * kMaxUsed], host_k[kCpuRows * kMaxUsed];   // a host entry's expert and its place
        int n_host = 0, n_cached = 0;
        for (int t = 0; t < m; ++t) {
            int ids[kMaxUsed];
            float w[kMaxUsed];
            route(il, h_rl_ + (size_t) (at + t) * n_expert_, tok_ ? tok_[at + t] : -1, ids, w, route_tmp_.data());
            for (int k = 0; k < n_used_; ++k) {
                const size_t e = (size_t) ids[k];
                ++use_[(size_t) il * n_expert_ + e];
                if (n_swap_ > 0) recent_[(size_t) il * n_expert_ + e] += 1.f;
                if (const uint8_t* s = L.slot[e]) {
                    cached[n_cached++] = {s, w[k], at + t};
                    continue;
                }
                host_e[n_host] = (int32_t) e;
                host_k[n_host] = k;
                host[n_host++] = {L.exp_gate + e * L.gu_stride, L.exp_up + e * L.gu_stride,
                                  L.exp_down + e * L.d_bytes, w[k], t};
            }
        }
        // Strata's STRATA_ROUTE_TAIL_SKIP: an expert outside VRAM that every token here ranks at tail_skip or later
        // adds little to any of them, so it is not computed.  The others keep the weights the router gave them.
        if (opt_.tail_skip > 0) {
            int kept = 0;
            for (int i = 0; i < n_host; ++i) {
                bool low = true;
                for (int j = 0; j < n_host && low; ++j) low = host_e[j] != host_e[i] || host_k[j] >= opt_.tail_skip;
                if (low) ++stats_.tail_experts;
                else host[kept++] = host[i];
            }
            n_host = kept;
        }
        stats_.gpu_experts += n_cached;
        stats_.cpu_experts += n_host;
        // The launches cost this thread more than they cost the card, so each one less counts.  The cached experts
        // go kSlots at a time, a launch a stage for all of them: gate and up, the activation, its quantization,
        // down.  Then one launch adds up to kSumRows of them into their tokens' rows, in the order they came.  A
        // format without a kernel for several matrices gets the four launches an expert.
        // Nothing else goes to the stream until the CPU experts are done, so one of their threads hands these to
        // the card meanwhile: on this thread they held the CPU experts up.
        const std::function<void()> launch = [&] {
            if (split_) cudaSetDevice(device_);   // one of the experts' threads, which every card's engine shares
            if (first && at == 0) (*first)();
            const size_t ff = (size_t) n_ff_exp_, q8_row = kernels::native_q8_1_bytes(n_ff_exp_);
            const int per = rows_ >= kSlots ? kSlots : 1;   // the scratch holds rows_ experts' values
            ScaledRows sum{};
            for (int i0 = 0; i0 < n_cached; i0 += per) {
                const int m = std::min(per, n_cached - i0);
                kernels::MmvqSlots gate_up{}, down{};
                for (int j = 0; j < m; ++j) {
                    const Cached& c = cached[i0 + j];
                    gate_up.w[j] = c.slot;
                    gate_up.x[j] = (const uint8_t*) xq_ + (size_t) c.row * xq_row;
                    gate_up.y[j] = g_ + (size_t) j * 2 * ff;
                    down.w[j] = c.slot + 2 * L.gu_bytes;
                    down.x[j] = (const uint8_t*) q8_ + (size_t) j * q8_row;
                    down.y[j] = d_ + (size_t) ((i0 + j) % kSumRows) * e_row;
                }
                const bool one = m > 1 && kernels::native_mmvq_slots((int) L.gu_type, gate_up, m, n_embd_,
                                                                      2 * n_ff_exp_, st_);
                for (int j = 0; j < m && !one; ++j)
                    kernels::native_mmvq((int) L.gu_type, gate_up.w[j], gate_up.x[j], gate_up.y[j], n_embd_,
                                         2 * n_ff_exp_, 1, st_);
                if (gelu_) geglu_rows(g_, h_, n_ff_exp_, m, st_);
                else swiglu_clamp_rows(g_, h_, n_ff_exp_, m, limit, st_);
                kernels::native_quantize_q8_1(h_, q8_, n_ff_exp_, m, st_);
                if (!(m > 1 && kernels::native_mmvq_slots((int) L.d_type, down, m, n_ff_exp_, n_embd_, st_)))
                    for (int j = 0; j < m; ++j)
                        kernels::native_mmvq((int) L.d_type, down.w[j], down.x[j], down.y[j], n_ff_exp_, n_embd_, 1,
                                             st_);
                for (int j = 0; j < m; ++j) {
                    const Cached& c = cached[i0 + j];
                    const int k = (i0 + j) % kSumRows;
                    sum.x[k] = down.y[j];
                    sum.acc[k] = fo_ + (size_t) c.row * e_row;
                    sum.w[k] = c.w;
                    if (k == kSumRows - 1 || i0 + j == n_cached - 1) add_scaled(sum, k + 1, n_embd_, st_);
                }
            }
        };

        // The next layer's router wait is what makes h_cpu_ free to write again.
        float* cpu = h_cpu_ + (size_t) at * e_row;
        if (n_host == 0) {
            launch();
            std::fill(cpu, cpu + (size_t) m * e_row, 0.f);
            continue;
        }
        any_host = true;
        t0 = Clock::now();
        cpu_experts(*workers_, host, n_host, m, (int) L.gu_type, (int) L.d_type, n_embd_, n_ff_exp_,
                    h_x_ + (size_t) at * e_row, limit, gelu_, cpu,
                    n_cached > 0 || (first && at == 0) ? &launch : nullptr);
        stats_.cpu_ms += ms_since(t0);
    }
    return any_host;
}

// The CPU's experts of n tokens into the routed experts' sum.
void Engine::moe_add(int n) { axpy(h_cpu_, 1.f, fo_, n * n_embd_, true, st_); }

// With every routed expert in VRAM, a few tokens' experts are routed on the card and found there by its picks
// (moe_card()), so a layer never waits for the host.  The cache's slots go to the card again when they changed.
bool Engine::card_moe(int n) {
    if (!all_cached() || !d_slots_ || n * n_used_ > kSumRows || opt_.dump) return false;
    bool same = !h_slots_.empty();
    for (int il = lo_; il < hi_ && same; ++il) {
        const Layer& L = layers_[(size_t) il];
        for (int e = 0; e < n_expert_ && same && L.moe; ++e)
            same = h_slots_[(size_t) il * n_expert_ + e] == L.slot[(size_t) e];
    }
    if (!same) {
        h_slots_.assign((size_t) n_layer_ * n_expert_, nullptr);
        for (int il = lo_; il < hi_; ++il)
            if (layers_[(size_t) il].moe)
                std::copy(layers_[(size_t) il].slot.begin(), layers_[(size_t) il].slot.end(),
                          h_slots_.begin() + (ptrdiff_t) il * n_expert_);
        ck(cudaMemcpy(d_slots_, h_slots_.data(), h_slots_.size() * sizeof(void*), cudaMemcpyHostToDevice),
           "expert slots");
    }
    return true;
}

// moe()'s launches for a few tokens whose experts are all in VRAM, with the card's picks for the host's: routed on
// the card, an expert's slot read from d_slots_ by its pick.  False, with nothing launched: this layer is routed
// on the host, or its formats have no kernel for picks on the card.
bool Engine::moe_card(Layer& L, int il, int n) {
    if (L.card == 0) return false;
    if (!kernels::native_mmvq_ids_supported((int) L.gu_type) || !kernels::native_mmvq_ids_supported((int) L.d_type) ||
        !route_card(il, n, d_ids_, d_wts_)) {
        L.card = 0;
        return false;
    }
    L.card = 1;
    const int rr = n * n_used_;
    const void* const* table = d_slots_ + (size_t) il * n_expert_;
    const kernels::MmvqIds gate_up{table, d_ids_, 0, xq_, kernels::native_q8_1_bytes(n_embd_), n_used_, g_,
                                   2 * (size_t) n_ff_exp_};
    kernels::native_mmvq_ids((int) L.gu_type, gate_up, rr, n_embd_, 2 * n_ff_exp_, st_);
    if (gelu_) geglu_rows(g_, h_, n_ff_exp_, rr, st_);
    else swiglu_clamp_rows(g_, h_, n_ff_exp_, rr, clamp_exp_[(size_t) il], st_);
    kernels::native_quantize_q8_1(h_, q8_, n_ff_exp_ * rr, 1, st_);   // the rows back to back, as one vector
    const kernels::MmvqIds down{table, d_ids_, 2 * L.gu_bytes, q8_, kernels::native_q8_1_bytes(n_ff_exp_), 1, d_,
                                (size_t) n_embd_};
    kernels::native_mmvq_ids((int) L.d_type, down, rr, n_ff_exp_, n_embd_, st_);
    moe_sum(d_, d_wts_, fo_, n_embd_, n_used_, n, st_);
    return true;
}

// m experts from the mapped file into the staging buffer: gate and up of each back to back from the start, the
// downs after them.  The two pinned buffers take turns, so the workers fill one while the other uploads.  `alone`:
// stagers_ fill them, while the workers compute experts.
void Engine::stage(const Layer& L, const int32_t* experts, int m, void* stream, bool alone) {
    staged_il_ = -1;   // whatever the buffer held
    const size_t gu2 = 2 * L.gu_bytes, down_at = pad((size_t) m * gu2);
    const int per = (int) (chunk_bytes_ / (gu2 + L.d_bytes));
    for (int j0 = 0; j0 < m; j0 += per) {
        const size_t c = (size_t) std::min(per, m - j0);
        stage_turn_ ^= 1;
        uint8_t* h = h_stage_[stage_turn_];
        const cudaEvent_t left = (cudaEvent_t) ev_stage_[stage_turn_];
        ck(cudaEventSynchronize(left), "expert staging");
        const auto copy = [&](int i) {
            const size_t j = (size_t) i / 3, e = (size_t) experts[(size_t) j0 + j];
            if (i % 3 == 0) std::memcpy(h + j * gu2, L.exp_gate + e * L.gu_stride, L.gu_bytes);
            else if (i % 3 == 1) std::memcpy(h + j * gu2 + L.gu_bytes, L.exp_up + e * L.gu_stride, L.gu_bytes);
            else std::memcpy(h + c * gu2 + j * L.d_bytes, L.exp_down + e * L.d_bytes, L.d_bytes);
        };
        (alone ? *stagers_ : *workers_).run((int) (3 * c), copy);
        ck(cudaMemcpyAsync(stage_ + (size_t) j0 * gu2, h, c * gu2, cudaMemcpyHostToDevice, cs(stream)),
           "expert staging");
        ck(cudaMemcpyAsync(stage_ + down_at + (size_t) j0 * L.d_bytes, h + c * gu2, c * L.d_bytes,
                           cudaMemcpyHostToDevice, cs(stream)),
           "expert staging");
        ck(cudaEventRecord(left, cs(stream)), "expert staging");
    }
    ck(cudaMemsetAsync(stage_ + down_at + (size_t) m * L.d_bytes, 0, kTail, cs(stream)), "expert staging");
}

// A MoE layer for a prompt batch: the experts on the GPU, by llama.cpp's MMQ.  A row is a token and one
// of the experts it routes to.  The rows are sorted by expert, the experts of the VRAM cache first; the other
// experts pass through the staging buffer, as many at a time as it holds.  A full batch starts the copy of
// L.ahead before it knows the routing, on a stream of its own, so the upload runs under the layer's attention.
// A call that copies only the experts it routes to leaves a share of them to the CPU, which computes them while
// the others are copied: the share that had the two take about as long (Options::cpu_share).
void Engine::moe_rows(Layer& L, int il, int n) {
    const int rr = n * n_used_;
    const size_t cap = (size_t) rows_ * n_used_;
    const bool ahead = opt_.stage_ahead > 0 && (n >= opt_.stage_ahead || shared_ahead_);
    const bool spare = !ahead && opt_.cpu_share != 0 && !all_cached();   // the CPU may take some (below)
    // The layer's tables.  tok and place: a row's token, and its place among that token's experts.  bounds: each
    // expert's first row.  rel: the same for the experts of a staging round, counted from the round's first row.
    // w: the weight of each place.
    int32_t* tok = h_tab_;
    int32_t* place = tok + cap;
    int32_t* bounds = place + cap;
    int32_t* rel = bounds + n_expert_ + 1;
    float* w = (float*) (rel + 2 * n_expert_ + 2);
    int32_t* cb = (int32_t*) (w + cap);   // per class: the cached experts' first rows, in the order of the slots
    router_scores(L, n);
    // Routed on the card, a batch hands the host its picks: the host took longer to route a batch of 1,024 tokens
    // than the card took for its experts.
    const bool on_card = route_card(il, n, d_ids_, d_wts_);
    if (on_card) {
        copy2(d_ids_, h_ids_, rr, d_wts_, w, rr, st_);
        if (spare) copy2(xn_, h_x_, n * n_embd_, nullptr, nullptr, 0, st_);
    } else if (spare) {
        copy2(rl_, h_rl_, n * n_expert_, xn_, h_x_, n * n_embd_, st_);
    } else {
        ck(cudaMemcpyAsync(h_rl_, rl_, (size_t) n * n_expert_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
           "router");
    }
    ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");

    moe_shared(L, il, n);

    // read() takes a layer's calls one after the other, so the buffer may hold this layer's experts from the
    // call before (staged_).  The ones copied ahead do not depend on the routing and are this call's too.  A
    // later round is this call's only if it routes to the same experts, and waiting to see means nothing is
    // copied ahead meanwhile.  That pays when the card gets to the routing in under half the time the copy
    // ahead takes: most of that copy then holds the card up.
    //
    // A small call on its own copies only the experts it routes to.  With another launch's experts the batched
    // kernels would split its sums another way, so it cannot take what a call before it left.  In a read of
    // several calls it copies ahead as the others do, and the copy is theirs too.
    if (ahead) plan_ahead(L, il);
    const int n_ahead = ahead ? (int) L.ahead.size() : 0;
    const bool mine = ahead && staged_il_ == il && (staged_ == L.ahead || 2 * route_ms_ < ahead_ms_);
    const bool early = n_ahead > 0 && !mine;
    if (early) {
        const Clock::time_point t = Clock::now();
        // the buffer is free once the layer before has read it
        ck(cudaStreamWaitEvent(cs(up_), (cudaEvent_t) ev_free_, 0), "expert staging");
        stage(L, L.ahead.data(), n_ahead, up_);
        ck(cudaEventRecord((cudaEvent_t) ev_ahead_, cs(up_)), "expert staging");
        ahead_ms_ = ms_since(t);
        // out already: how long before is not known until a call waits for it
        route_ms_ = cudaEventQuery((cudaEvent_t) ev_) == cudaSuccess ? 0 : ahead_ms_;
    }

    const Clock::time_point t0 = Clock::now();
    ck(cudaEventSynchronize((cudaEvent_t) ev_), "router");
    if (ahead && !early) route_ms_ = ms_since(t0);
    stats_.route_wait_ms += ms_since(t0);

    const size_t cb_row = (size_t) n_expert_ + 1;
    const int32_t* d_tok = tab_;
    const int32_t* d_place = d_tok + cap;
    const int32_t* d_bounds = d_place + cap;
    const int32_t* d_rel = d_bounds + n_expert_ + 1;
    const float* d_w = (const float*) (d_rel + 2 * n_expert_ + 2);
    const int32_t* d_cb = (const int32_t*) (d_w + cap);

    // A token's routing is a softmax over every expert and a search for the largest few.  On one thread it was
    // 0.9 s of the 2.4 s a 7,709-token prompt took with 40 layers of 256 experts, so the host's threads take a
    // stretch of tokens each.
    const int parts = on_card ? 0 : std::min(workers_->threads(), (n + kStageRows - 1) / kStageRows);
    if (on_card) std::copy(h_ids_, h_ids_ + rr, ids_.begin());
    if (parts > 0) workers_->run(parts, [&](int p) {
        float* tmp = route_tmp_.data() + (size_t) p * 2 * n_expert_;
        const int end = (int) ((int64_t) n * (p + 1) / parts);
        for (int t = (int) ((int64_t) n * p / parts); t < end; ++t)
            route(il, h_rl_ + (size_t) t * n_expert_, tok_ ? tok_[t] : -1, ids_.data() + (size_t) t * n_used_,
                  w + (size_t) t * n_used_, tmp);
    });
    count_.assign((size_t) n_expert_, 0);
    for (int i = 0; i < rr; ++i) {
        ++count_[(size_t) ids_[(size_t) i]];
        ++use_[(size_t) il * n_expert_ + (size_t) ids_[(size_t) i]];
    }
    // The CPU computes some of the routed experts outside VRAM while the others are copied: the ones the fewest
    // tokens chose, whose copy costs as much as any other's and which cost the CPU about as much as their tokens.
    int n_spare = 0, n_cpu = 0;
    const int size = std::clamp((int) std::bit_width((unsigned) n) - 6, 0, 4);   // which of share_
    on_cpu_.assign((size_t) n_expert_, 0);
    if (spare) {
        spare_.clear();
        for (int e = 0; e < n_expert_; ++e)
            if (count_[(size_t) e] > 0 && !L.slot[(size_t) e]) spare_.push_back(e);
        std::stable_sort(spare_.begin(), spare_.end(),
                         [&](int32_t a, int32_t b) { return count_[(size_t) a] < count_[(size_t) b]; });
        n_spare = (int) spare_.size();
        const double s = opt_.cpu_share > 0 ? std::min(opt_.cpu_share, 1.f) : share_[size];
        n_cpu = (int) std::lround(s * n_spare);
        for (int j = 0; j < n_cpu; ++j) on_cpu_[(size_t) spare_[(size_t) j]] = 1;
    }
    // The cached experts that have rows, class after class and a class's in the order of the slots (chan_ has
    // every cached expert in that order); the experts already on their way; then the others that have rows.
    chan_.clear();
    for (int e = 0; e < n_expert_; ++e)
        if (L.slot[(size_t) e]) chan_.push_back(e);
    std::sort(chan_.begin(), chan_.end(),
              [&](int32_t a, int32_t b) { return L.slot[(size_t) a] < L.slot[(size_t) b]; });
    const int n_chan = (int) chan_.size();
    group_.clear();
    for (int c = 0; c < kClasses; ++c)
        for (const int32_t e : chan_)
            if (count_[(size_t) e] > 0 && tile_class(count_[(size_t) e]) == c) group_.push_back(e);
    const int n_cached = (int) group_.size();
    group_.insert(group_.end(), L.ahead.begin(), L.ahead.begin() + n_ahead);
    sent_.assign((size_t) n_expert_, 0);
    for (int j = 0; j < n_ahead; ++j) sent_[(size_t) L.ahead[(size_t) j]] = 1;
    for (int e = 0; e < n_expert_; ++e)
        if (count_[(size_t) e] > 0 && !L.slot[(size_t) e] && !sent_[(size_t) e] && !on_cpu_[(size_t) e])
            group_.push_back(e);
    const int n_group = (int) group_.size();
    auto rows_of = [&](int g) { return count_[(size_t) group_[(size_t) g]]; };

    int at = 0;
    for (int g = 0; g < n_group; ++g) {
        bounds[g] = at;
        first_[(size_t) group_[(size_t) g]] = at;
        at += rows_of(g);
    }
    bounds[n_group] = at;
    const int rg = at;   // the card's rows
    host_.clear();
    for (int i = 0; i < rr; ++i) {
        const size_t e = (size_t) ids_[(size_t) i];
        if (on_cpu_[e]) {   // by token, as cpu_experts() takes them, and nothing of it in the card's sum
            host_.push_back({L.exp_gate + e * L.gu_stride, L.exp_up + e * L.gu_stride, L.exp_down + e * L.d_bytes,
                             w[i], i / n_used_});
            w[i] = 0.f;
            continue;
        }
        const int row = first_[e]++;
        tok[row] = i / n_used_;
        place[row] = i;
    }
    const size_t gu2 = 2 * L.gu_bytes;
    // a round is what the staging buffer holds at a time; round_ has the first group of each, and the end
    const int per_round = (int) (stage_bytes_ / (gu2 + L.d_bytes));
    round_.assign(1, n_cached);
    if (n_ahead > 0) round_.push_back(n_cached + n_ahead);
    while (round_.back() < n_group) round_.push_back(std::min(round_.back() + per_round, n_group));
    const int n_round = (int) round_.size() - 1;
    for (int r = 0; r < n_round; ++r)
        for (int g = round_[(size_t) r]; g <= round_[(size_t) r + 1]; ++g)
            rel[g - n_cached + r] = bounds[g] - bounds[round_[(size_t) r]];
    // A class's launch runs over every cached expert of a run of slots, and the others must have no rows in it:
    // cb has, for each class, where an expert's rows start when only that class's experts have any.
    for (int c = 0, at = 0; c < kClasses; ++c) {
        int32_t* b = cb + (size_t) c * cb_row;
        for (int j = 0; j < n_chan; ++j) {
            b[j] = at;
            const int rows = count_[(size_t) chan_[(size_t) j]];
            if (rows > 0 && tile_class(rows) == c) at += rows;
        }
        b[n_chan] = at;
    }
    ck(cudaMemcpyAsync(tab_, h_tab_,
                       (3 * cap + 3 * (size_t) n_expert_ + 3 + kClasses * cb_row) * sizeof(int32_t),
                       cudaMemcpyHostToDevice, cs(st_)),
       "expert rows");

    const int rc = bounds[n_cached];   // the rows of the cached experts
    stats_.batch_cached += rc;
    stats_.batch_staged += rg - rc;

    const float limit = clamp_exp_[(size_t) il];
    const int gt = (int) L.gu_type, dt = (int) L.d_type;
    const int64_t ff = n_ff_exp_, ff2 = 2 * ff;
    int holds = -1;   // the round in the buffer when this call is done
    // The card's part.  With the CPU's beside it, the rows of its experts' places are left at nothing.
    const auto card = [&] {
        if (n_cpu > 0) ck(cudaMemsetAsync(dx_, 0, (size_t) rr * n_embd_ * sizeof(float), cs(st_)), "expert rows");
        if (rg > 0) Mmq::quantize(xn_, d_tok, mq_, gt, n_embd_, n_embd_, rg, st_);

        // The cached experts whose slots lie one behind the other are the matrices of one launch, as the experts of
        // a staging round are; a launch for each class that has rows among them.  The cache is filled a layer's
        // experts side by side, so that is the whole layer until experts trade slots (follow()).  A launch for each
        // expert was a sixth of a long prompt's time in a model with 256 experts a layer.
        auto slot_of = [&](int j) { return L.slot[(size_t) chan_[(size_t) j]]; };
        runs_.clear();
        for (int j = 0; j < n_chan;) {
            int m = 1, most[kClasses] = {};
            while (j + m < n_chan && slot_of(j + m) == slot_of(j + m - 1) + L.slot_bytes) ++m;
            for (int i = j; i < j + m; ++i) {
                const int rows = count_[(size_t) chan_[(size_t) i]];
                if (rows > 0) most[tile_class(rows)] = std::max(most[tile_class(rows)], rows);
            }
            for (int c = 0; c < kClasses; ++c)
                if (most[c] > 0) runs_.push_back({c, j, m, most[c]});
            j += m;
        }
        for (const Run& r : runs_)
            mmq_->mul(slot_of(r.j), gt, ff2, n_embd_, L.slot_bytes, r.m, mq_, d_cb + (size_t) r.c * cb_row + r.j, iota_,
                      rg, r.most, gu_, ff2, st_);
        if (rc > 0) {
            if (gelu_) geglu_rows(gu_, hx_, n_ff_exp_, rc, st_);
            else swiglu_clamp_rows(gu_, hx_, n_ff_exp_, rc, limit, st_);
            Mmq::quantize(hx_, nullptr, mq2_, dt, ff, ff, rc, st_);
        }
        for (const Run& r : runs_)
            mmq_->mul(slot_of(r.j) + gu2, dt, n_embd_, ff, L.slot_bytes, r.m, mq2_, d_cb + (size_t) r.c * cb_row + r.j,
                      d_place, rc, r.most, dx_, n_embd_, st_);

        // The round the buffer holds goes first and is not copied.  If it holds none of this call's, the experts
        // that go ahead are copied last, so the next call finds them.  The order does not change a round's launches.
        int kept = -1;
        for (int r = 0; r < n_round && mine && kept < 0; ++r) {
            const auto first = group_.begin() + round_[(size_t) r], last = group_.begin() + round_[(size_t) r + 1];
            if ((size_t) (last - first) == staged_.size() && std::equal(first, last, staged_.begin())) kept = r;
        }
        const int from = mine && kept < 0 ? 1 : 0;
        holds = early ? 0 : kept;
        for (int turn = 0; turn < n_round; ++turn) {
            const int r = kept < 0 ? (turn + from) % n_round : turn == 0 ? kept : turn - (turn <= kept ? 1 : 0);
            const int g0 = round_[(size_t) r], m = round_[(size_t) r + 1] - g0;
            const size_t down_at = pad((size_t) m * gu2);
            const int r0 = bounds[g0], rows = bounds[g0 + m] - r0;
            if (r == kept) {
                // the buffer has it
            } else if (r == 0 && early) {
                ck(cudaStreamWaitEvent(cs(st_), (cudaEvent_t) ev_ahead_, 0), "expert staging");
            } else if (rows > 0) {
                stage(L, group_.data() + g0, m, st_, n_cpu > 0);
                holds = r;
            }
            if (rows == 0) continue;   // nothing routed to the experts that go ahead
            int most = 0;
            for (int g = g0; g < g0 + m; ++g) most = std::max(most, rows_of(g));
            mmq_->mul(stage_, gt, ff2, n_embd_, gu2, m, mq_, d_bounds + g0, iota_, rg, most, gu_, ff2, st_);
            if (gelu_) geglu_rows(gu_ + (size_t) r0 * ff2, hx_ + (size_t) r0 * ff, n_ff_exp_, rows, st_);
            else swiglu_clamp_rows(gu_ + (size_t) r0 * ff2, hx_ + (size_t) r0 * ff, n_ff_exp_, rows, limit, st_);
            Mmq::quantize(hx_ + (size_t) r0 * ff, nullptr, mq2_, dt, ff, ff, rows, st_);
            mmq_->mul(stage_ + down_at, dt, n_embd_, ff, L.d_bytes, m, mq2_, d_rel + (g0 - n_cached + r), d_place + r0,
                      rows, most, dx_, n_embd_, st_);
        }
        ck(cudaEventRecord((cudaEvent_t) ev_free_, cs(st_)), "expert staging");
    };
    if (n_cpu == 0) {
        card();
    } else {   // the card's part on a thread of its own, and the CPU's on the workers
        if (!stagers_) stagers_ = std::make_unique<Workers>(4);   // one thread copied slower than the bus
        Clock::time_point done;
        Behind behind;
        const Clock::time_point start = Clock::now();
        behind.go([&] {
            cudaSetDevice(device_);   // a thread starts on card 0
            card();
            ck(cudaStreamSynchronize(cs(st_)), "expert staging");
            done = Clock::now();
        });
        cpu_experts(*workers_, host_.data(), (int) host_.size(), n, gt, dt, n_embd_, n_ff_exp_, h_x_, limit, gelu_,
                    h_cpu_, nullptr);
        const double cpu = ms_since(start);
        behind.wait();
        const double gpu = std::chrono::duration<double, std::milli>(done - start).count();
        stats_.cpu_ms += cpu;
        stats_.cpu_experts += (int64_t) host_.size();
        // the share that would have had both done at once, if each one's time grows with its share
        if (opt_.cpu_share < 0 && n_cpu < n_spare) {
            const double s = (double) n_cpu / n_spare, c = cpu / s, g = gpu / (1 - s);
            share_[size] = std::clamp(0.5 * share_[size] + 0.5 * g / (c + g), 0.05, 0.95);
        }
    }
    moe_sum(dx_, d_w, fo_, n_embd_, n_used_, n, st_);
    if (n_cpu > 0) moe_add(n);
    if (ahead && holds >= 0) {
        staged_.assign(group_.begin() + round_[(size_t) holds], group_.begin() + round_[(size_t) holds + 1]);
        staged_il_ = il;
    }
}

// The rows a recorded attention launch is sized for, for n tokens from pos(): the power of two from 1024 that
// holds their rows and the ones before them, or the context if that is less.  room_key() numbers the sizes.
int Engine::room(int n) const { return (int) std::min<int64_t>(int64_t(1024) << room_key(n), opt_.max_context); }

int Engine::room_key(int n) const {
    int key = 0;
    for (int64_t r = 1024; r < pos_ + n; r *= 2) ++key;
    return key;
}

// Records what `work` hands to the stream, as a graph that is launched with one call from then on.
template <class F>
void* Engine::record(F&& work) {
    cudaGraph_t g = nullptr;
    ck(cudaStreamBeginCapture(cs(st_), cudaStreamCaptureModeThreadLocal), "cudaStreamBeginCapture");
    recording_ = recorded_ = true;
    try {
        work();
    } catch (...) {
        recording_ = false;
        if (cudaStreamEndCapture(cs(st_), &g) == cudaSuccess && g) cudaGraphDestroy(g);
        cudaGetLastError();   // a capture the failure broke does not end cleanly: the work's exception says why
        throw;
    }
    recording_ = false;
    ck(cudaStreamEndCapture(cs(st_), &g), "cudaStreamEndCapture");
    cudaGraphExec_t exec = nullptr;
    const cudaError_t made = cudaGraphInstantiateWithFlags(&exec, g, 0);
    cudaGraphDestroy(g);
    ck(made, "cudaGraphInstantiate");
    graphs_.push_back(exec);
    return exec;
}

// n tokens from pos() through every layer.  The sequence is not longer for it: the caller says by how much.
void Engine::run(const int32_t* tokens, int n) {
    if (n < 1 || n > rows_) throw std::logic_error("eval takes 1 to max_rows() tokens");
    if (pos_ + n > opt_.max_context)
        throw std::runtime_error("the context is full (" + std::to_string(opt_.max_context) + " tokens)");
    cover(pos_ + n);

    // h_r_ and h_cpu_ may still be the source of the previous call's copies
    ck(cudaStreamSynchronize(cs(st_)), "eval");
    if (n_swap_ > 0) follow(n < kStageRows ? n : 0);
    tok_ = tokens;
    const size_t e_row = (size_t) n_embd_, r_row = (size_t) streams() * e_row;
    const float scale = embed_scale();
    if (pass_) {   // a later card: the card before it left the rows
        std::memcpy(h_r_, pass_ + (size_t) (pos_ - pass_at_) * r_row, (size_t) n * r_row * sizeof(float));
    } else {
        for (int t = 0; t < n; ++t) {
            float* r = h_r_ + (size_t) t * r_row;
            if (!input(tokens[t], pos_ + t, r) && scale != 1.f)
                for (size_t i = 0; i < e_row; ++i) r[i] *= scale;
            for (int h = 1; h < streams(); ++h) std::memcpy(r + (size_t) h * e_row, r, e_row * sizeof(float));
        }
    }
    if (n < kStageRows) copy2(h_r_, r_, n * (int) r_row, nullptr, nullptr, 0, st_);
    else ck(cudaMemcpyAsync(r_, h_r_, (size_t) n * r_row * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "embed");

    // A token alone, or a window of guessed ones, hands the card the same launches every time, some thirty a
    // layer up to the router, and they cost this thread more than the card.  They are recorded once, a graph a
    // layer and a number of tokens, and replayed with one call.  Only for the layers that can be (Layer::fixed).
    // What a launch needs of the position it reads at at_; an attention over the rows so far is recorded again
    // when the rows outgrow what it was sized for (head_key()).  A prompt's few tokens are not a window: they
    // leave no snapshots of the recurrent state, so their launches are others.
    bool replay = opt_.graphs && n <= opt_.window && (windowed_ || n == 1) && !opt_.dump;
    for (int t = 0; t < n && replay; ++t) replay = !image_at(pos_ + t);
    const auto replayed = [&](std::vector<void*>& made, size_t key, auto&& work) {
        if (made.size() <= key) made.resize(key + 1, nullptr);
        if (!made[key]) made[key] = record(work);
        ck(cudaGraphLaunch((cudaGraphExec_t) made[key], cs(st_)), "layer graph");
    };
    if (replay) {   // a pair a token: a window's tokens may go one by one, each from its own
        for (int t = 0; t < n; ++t) {
            h_at_[2 * t] = pos_ + t;
            h_at_[2 * t + 1] = turn(pos_ + t);
        }
        copy2(h_at_, at_, n * 2 * (int) (sizeof(long long) / 4), nullptr, nullptr, 0, st_);
    }
    // A layer's launches around its routed experts.  up(): to its router's copy, with the shared expert behind it.
    // tail(): what the layer does with the experts' sum.  `rec` numbers the recording of m tokens' launches: a
    // call's tokens together are [0, window) by their number, and a half of a window comes after those, by the
    // window's tokens and which half it is.  Which tokens leave their state is the window's, so its halves' too.
    const size_t wn = (size_t) opt_.window, recs = 3 * wn;
    const auto up = [&](int il, int m, size_t rec) {
        Layer& L = layers_[(size_t) il];
        const size_t head = (size_t) head_key(il, m) * recs + rec;
        if (replay && L.fixed) replayed(L.heads, head, [&] { layer_head(L, il, m); });
        else layer_head(L, il, m);
        if (!L.moe || batched(L, m)) return;
        ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");
        if (replay) replayed(L.shared, rec, [&] { moe_shared(L, il, m); });
        else moe_shared(L, il, m);
    };
    const auto tail = [&](int il, int m, size_t rec) {
        Layer& L = layers_[(size_t) il];
        dbg("ffn", il, fo_, n_embd_);
        if (replay && L.fixed) replayed(L.tail, rec, [&] { layer_tail(L, il, m); });
        else layer_tail(L, il, m);
    };

    // A window's tokens, with experts on the CPU, go through the layers together or in two halves, one behind
    // the other (halves_, which window() sets).  Together, the card's part of a layer and the CPU's take turns:
    // the card waits out the CPU's experts and the CPU the card's layer.  In halves each has work while the
    // other is busy: this thread does nothing but route a half and compute its experts, half after half, and
    // while it does, one of the experts' threads hands the card the other half: its experts' sum, the rest of
    // its layer and the next layer up to the router.  That half's scores are there when its turn comes again.
    //
    // The card's part of a layer takes about as long for two tokens as for one: it reads the layer's dense
    // weights once for all of them, and its launches are the same.  So a window is two halves and not a part a
    // token, which had the card read every layer once for each token.  Halves have the card read it twice where
    // the tokens together have it read once, and that is what the choice between the two turns on.
    //
    // A half's launches are the ones its tokens have in a call of their own, at their rows of the buffers
    // (shift_rows()), so each token comes out as it would alone.
    if (n > 1 && windowed_ && !all_cached() && halves_) {
        const int first[3] = {0, (n + 1) / 2, n};   // half k is the tokens [first[k], first[k + 1])
        const unsigned bits = snap_bits_;           // the window's tokens that leave the recurrent state
        int at = 0;
        // The buffers at half k's rows, and its share of the window's states.  Returns the half's tokens.
        const auto half = [&](int k) {
            shift_rows(first[k] - at);
            at = first[k];
            const int m = first[k + 1] - at;
            snap_bits_ = bits >> at & ((1u << m) - 1u);
            snap_at_ = snap_ + size_t(std::popcount(bits & ((1u << at) - 1u))) * rec_n_;
            ev_ = ev_row_[k];
            return m;
        };
        // Half k on from layer il, whose experts it has (-1: from its start), to the next router's copy.
        const auto go_on = [&](int k, int il, bool add) {
            const int m = half(k);
            const size_t rec = wn + 2 * (size_t) (n - 1) + (size_t) k;
            if (il >= 0) {
                if (add) moe_add(m);
                tail(il, m, rec);
            }
            for (int j = il < 0 ? lo_ : il + 1; j < hi_; ++j) {
                up(j, m, rec);
                if (layers_[(size_t) j].moe) break;
            }
        };
        const auto back = [&] {
            shift_rows(-at);
            at = 0;
            ev_ = ev_row_[0];
            snap_bits_ = bits;
            snap_at_ = snap_;
        };
        struct {
            int k = -1, il = -1;
            bool add = false;
        } late;   // the half whose experts are computed and that has not gone on
        std::exception_ptr failed;
        try {
            for (int k = 0; k < 2; ++k) go_on(k, -1, false);
            for (int il = lo_; il < hi_; ++il) {
                Layer& L = layers_[(size_t) il];
                if (!L.moe) continue;
                for (int k = 0; k < 2; ++k) {
                    // by another thread while this one computes: it must leave the rows as it found them
                    const std::function<void()> other = [&] {
                        try {
                            if (split_) cudaSetDevice(device_);
                            go_on(late.k, late.il, late.add);
                            half(k);
                        } catch (...) {
                            failed = std::current_exception();
                        }
                    };
                    const int m = half(k);
                    const bool add = moe(L, il, m, late.k >= 0 ? &other : nullptr);
                    if (failed) std::rethrow_exception(failed);
                    late = {k, il, add};
                }
            }
            if (late.k >= 0) go_on(late.k, late.il, late.add);
        } catch (...) {
            back();
            throw;
        }
        back();
        return;
    }
    // The routed experts on the card alone, when every one of them is in VRAM: recorded too, once a layer has
    // shown it routes there (Layer::card).
    const bool card = n < kStageRows && card_moe(n);
    const auto on_card = [&](int il, int m, size_t rec) {
        Layer& L = layers_[(size_t) il];
        if (!card) return false;
        if (replay && L.card == 1) replayed(L.card_moe, rec, [&] { moe_card(L, il, m); });
        else if (!moe_card(L, il, m)) return false;
        stats_.gpu_experts += m * n_used_;
        return true;
    };
    // Once every layer has shown that, a call's layers are one recording: on an RTX 5090 the card waited 2 to 4
    // microseconds from one layer's recording to the next, four of them a layer.
    bool whole = card && replay;
    for (int il = lo_; il < hi_ && whole; ++il) {
        const Layer& L = layers_[(size_t) il];
        whole = L.fixed && (!L.moe || L.card == 1);
    }
    if (whole) {
        std::vector<int> key{n};
        for (int il = lo_; il < hi_; ++il) key.push_back(head_key(il, n));
        void*& pass = passes_[key];
        if (!pass)
            pass = record([&] {
                for (int il = lo_; il < hi_; ++il) {
                    Layer& L = layers_[(size_t) il];
                    layer_head(L, il, n);
                    if (!L.moe) continue;
                    moe_shared(L, il, n);
                    moe_card(L, il, n);
                    layer_tail(L, il, n);
                }
            });
        ck(cudaGraphLaunch((cudaGraphExec_t) pass, cs(st_)), "layer graph");
        for (int il = lo_; il < hi_; ++il) stats_.gpu_experts += layers_[(size_t) il].moe ? n * n_used_ : 0;
        return;
    }
    for (int il = lo_; il < hi_; ++il) {
        Layer& L = layers_[(size_t) il];
        up(il, n, (size_t) n - 1);
        if (!L.moe) continue;
        if (batched(L, n)) moe_rows(L, il, n);
        else if (!on_card(il, n, (size_t) n - 1) && moe(L, il, n)) moe_add(n);
        tail(il, n, (size_t) n - 1);
    }
}

void Engine::shift_rows(ptrdiff_t by) {
    const ptrdiff_t e = n_embd_;
    r_ += by * streams() * e;
    xn_ += by * e;
    fo_ += by * e;
    rl_ += by * n_expert_;
    xq_ = (uint8_t*) xq_ + by * (ptrdiff_t) kernels::native_q8_1_bytes(n_embd_);
    h_rl_ += by * n_expert_;
    h_x_ += by * e;
    h_cpu_ += by * e;
    at_ += by * 2;
    pos_ += by;
    if (tok_) tok_ += by;
}

// Pinned memory for n rows of the residual on their way to the next card.
float* Engine::pass_rows(int n) {
    if (h_pass_rows_ < n) {
        if (h_pass_) {
            host_allocs_.erase(std::find(host_allocs_.begin(), host_allocs_.end(), (void*) h_pass_));
            ck(cudaFreeHost(h_pass_), "freeing pinned memory");
            h_pass_ = nullptr;
        }
        h_pass_rows_ = std::max(n, rows_);
        h_pass_ = (float*) hmalloc((size_t) h_pass_rows_ * streams() * n_embd_ * sizeof(float), "pinned memory");
    }
    return h_pass_;
}

// The n rows of the residual a call from `pos` leaves in r_, to the next card, which reads them for its layers.
void Engine::hand_on(int64_t pos, int n, float* h) {
    ck(cudaMemcpyAsync(h, r_, (size_t) n * streams() * n_embd_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
       "the next card");
    ck(cudaStreamSynchronize(cs(st_)), "the next card");
    next_->pass_ = h;
    next_->pass_at_ = pos;
}

void Engine::eval(const int32_t* tokens, int n, bool want_logits) {
    OnCard here(device_, split_);
    run(tokens, n);
    if (next_) {
        hand_on(pos_, n, pass_rows(n));
        next_->eval(tokens, n, want_logits);
        pos_ += n;
        stats_.tokens += n;
        return;
    }
    // The model's output: the last token's for the logits, and every token's for the draft block.
    if (want_logits || draft_) {
        const bool replay = opt_.graphs && n == 1 && want_logits && !opt_.dump;
        const int m = draft_ ? n : 1;
        float* h = draft_ ? hn_ + n_embd_ : xn_;
        const auto tail = [&] {
            output_rows(n - m, m, h);
            dbg("out_norm", n_layer_, h, n_embd_);
            if (!want_logits) return;
            lin(output_, h + (size_t) (m - 1) * n_embd_, logits_, 1);
            finish_logits(logits_, 1);
            ck(cudaMemcpyAsync(h_logits_, logits_, (size_t) n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost,
                               cs(st_)),
               "logits");
        };
        if (replay) {
            if (!tail_) tail_ = record(tail);
            ck(cudaGraphLaunch((cudaGraphExec_t) tail_, cs(st_)), "logits graph");
        } else {
            tail();
        }
        if (want_logits) ck(cudaStreamSynchronize(cs(st_)), "eval");
    }
    // behind the logits: the caller has them while the card does this
    if (draft_) draft_rows(tokens, ahead_pos_ == pos_ && tokens[0] == ahead_tok_ ? 1 : 0, n);
    pos_ += n;
    stats_.tokens += n;
}

// A layer for n tokens, as run() hands it to the card when nothing is replayed or printed.
void Engine::layer(int il, int n) {
    Layer& L = layers_[(size_t) il];
    layer_head(L, il, n);
    if (!L.moe) return;
    if (batched(L, n)) {
        moe_rows(L, il, n);
    } else {
        ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");
        moe_shared(L, il, n);
        if (moe(L, il, n)) moe_add(n);
    }
    layer_tail(L, il, n);
}

// A prompt's tokens in the calls eval() would get, max_rows() at a time.  With experts outside the cache, each
// call copies every layer's through the staging buffer, which is most of what reading a prompt takes.  So the
// calls go through a layer one after the other before any goes on to the next, and the second one finds the
// layer's experts where the first one's copy left them (moe_rows()).  A call's tokens meet what they met in a
// call of their own, in the same order, so the result is eval()'s.  But a call under stage_ahead tokens takes
// those experts too, which on its own it would not, and its sums are split as a larger call's.  A read of fewer
// than stage_ahead tokens is the calls themselves: the copy of every layer's experts took a second, where its
// calls copy the experts they route to.
//
// All a call needs kept from one layer to the next is its rows of the residual.  r_ holds two calls' rows.  With
// more calls, a call's rows go to host memory when the card is done with them and come back before the call's
// turn at the next layer, in the half of r_ the call two before it has left: both copies are on a stream of
// their own, beside the card's work on the call between.
//
// A checkpoint inside the read (a mark) is taken a layer at a time: when a layer has run the tokens before it,
// the layer's state is what save_state() would find there, and it is copied before the next call changes it.
bool Engine::read(const int32_t* tokens, int n, const Mark* marks, int n_marks,
                  const std::function<bool(int, int)>* turn) {
    if (n < 1 || n > read_rows_) throw std::logic_error("read takes 1 to max_read() tokens");
    for (const Image& im : images_)
        if (im.at < pos_ + n && im.at + im.n > pos_)
            throw std::logic_error("read takes no rows of a picture: eval() does, a picture a call");
    for (int k = 0; k < n_marks; ++k)
        if (marks[k].at <= (k ? marks[k - 1].at : 0) || marks[k].at >= n)
            throw std::logic_error("read: the marks rise, above 0 and below the tokens read");

    // the calls: up to max_rows() tokens each, and none across a mark
    calls_.clear();
    for (int at = 0, k = 0; at < n;) {
        const int m = std::min(rows_, (k < n_marks ? marks[k].at : n) - at);
        const bool ends = k < n_marks && at + m == marks[k].at;
        calls_.push_back({at, m, ends ? k : -1});
        at += m;
        k += ends;
    }
    const int n_calls = (int) calls_.size();
    OnCard here(device_, split_);
    cover(pos_ + n);
    // On several cards, each reads through its own layers and hands the rows on, and `turn` hears of a card's
    // turns as its share of the model's layers: [lo_, to).
    const std::function<bool(int, int)>* asked = turn;
    std::function<bool(int, int)> share;
    const auto share_to = [&](int to) {
        if (!asked || (lo_ == 0 && to == n_layer_)) return asked;   // all of them
        share = [&, to](int done, int of) { return (*asked)(lo_ * of + done * (to - lo_), n_layer_ * of); };
        return (const std::function<bool(int, int)>*) &share;
    };
    if (n_calls == 1 || !mmq_ || all_cached() || n < opt_.stage_ahead) {
        turn = share_to(n_layer_);   // eval() takes the calls through the cards after this one
        // On several cards the next one takes a call on a thread of its own, while this one runs the call after
        // it.  Each card runs the calls as eval() does, in the same order.
        Behind behind;
        for (int c = 0; c < n_calls; ++c) {
            const Call& call = calls_[(size_t) c];
            if (next_) {
                run(tokens + call.at, call.m);
                behind.wait();
                hand_on(pos_, call.m, pass_rows(call.m));
                behind.go([this, t = tokens + call.at, m = call.m] { next_->eval(t, m, false); });
                pos_ += call.m;
                stats_.tokens += call.m;
            } else {
                eval(tokens + call.at, call.m, false);
            }
            if (call.mark >= 0 && marks[call.mark].state) {
                behind.wait();
                save_state(marks[call.mark].state);
            }
            if (turn && c + 1 < n_calls && !(*turn)(c + 1, n_calls)) {
                behind.wait();
                return false;
            }
        }
        behind.wait();
        return true;
    }
    turn = share_to(hi_);
    if (pos_ + n > opt_.max_context)
        throw std::runtime_error("the context is full (" + std::to_string(opt_.max_context) + " tokens)");

    ck(cudaStreamSynchronize(cs(st_)), "read");
    if (n_swap_ > 0) follow(0);
    const size_t e_row = (size_t) n_embd_, r_row = (size_t) streams() * e_row, r_bytes = r_row * sizeof(float);
    const size_t part = (size_t) std::max(streams(), draft_ ? 3 : 1) * rows_ * e_row;   // a call's rows of r_
    const float scale = embed_scale();
    float* const r0 = r_;
    const int64_t pos0 = pos_;
    const bool host = n_calls > 2;
    if (host && h_res_rows_ < n) {   // for this read's tokens and some more, so that a longer one seldom asks again
        if (h_res_) {
            host_allocs_.erase(std::find(host_allocs_.begin(), host_allocs_.end(), (void*) h_res_));
            ck(cudaFreeHost(h_res_), "freeing pinned memory");
            h_res_ = nullptr;
        }
        h_res_rows_ = std::min(read_rows_, (n + 4095) / 4096 * 4096);
        h_res_ = (float*) hmalloc((size_t) h_res_rows_ * r_bytes, "pinned memory");
    }
    const auto home = [&](int c) { return h_res_ + (size_t) calls_[(size_t) c].at * r_row; };
    const auto half = [&](int c) { return r0 + (size_t) (host ? c % 2 : c) * part; };
    // call c's rows back into its half of r_, behind whatever the stream has there of the half's last call
    const auto fetch = [&](int c) {
        ck(cudaMemcpyAsync(half(c), home(c), (size_t) calls_[(size_t) c].m * r_bytes, cudaMemcpyHostToDevice,
                           cs(rs_)),
           "read");
        ck(cudaEventRecord((cudaEvent_t) ev_in_[c % 2], cs(rs_)), "read");
    };
    for (int c = 0; c < n_calls; ++c) {
        const Call& call = calls_[(size_t) c];
        float* rows = host ? home(c) : h_r_;
        if (pass_)
            std::memcpy(rows, pass_ + (size_t) (pos0 + call.at - pass_at_) * r_row, (size_t) call.m * r_bytes);
        for (int t = 0; t < call.m && !pass_; ++t) {
            float* r = rows + (size_t) t * r_row;
            if (!input(tokens[call.at + t], pos0 + call.at + t, r) && scale != 1.f)
                for (size_t i = 0; i < e_row; ++i) r[i] *= scale;
            for (int h = 1; h < streams(); ++h) std::memcpy(r + (size_t) h * e_row, r, e_row * sizeof(float));
        }
        if (host) continue;
        ck(cudaMemcpyAsync(half(c), h_r_, (size_t) call.m * r_bytes, cudaMemcpyHostToDevice, cs(st_)), "embed");
        ck(cudaStreamSynchronize(cs(st_)), "embed");   // h_r_ is filled again
    }
    if (host) {
        fetch(0);
        fetch(1);
    }
    // a turn for each layer, and one for the model's output, which the draft block keeps a row of
    const int turns = hi_ - lo_ + (draft_ ? 1 : 0);
    shared_ahead_ = true;
    try {
        for (int k = 0; k < turns; ++k) {
            const int il = lo_ + k;
            // Between two turns the caller may give the read up.  The layers so far have the read's tokens in
            // their state and the others have not: the sequence cannot go on from here, nor from where it was.
            if (turn && k > 0 && !(*turn)(k, turns)) {
                if (host) ck(cudaStreamSynchronize(cs(rs_)), "read");
                ck(cudaStreamSynchronize(cs(st_)), "read");
                shared_ahead_ = false;
                r_ = r0;
                pos_ = pos0;
                return false;
            }
            for (int c = 0; c < n_calls; ++c) {
                const Call& call = calls_[(size_t) c];
                r_ = half(c);
                pos_ = pos0 + call.at;
                tok_ = tokens + call.at;
                if (host) ck(cudaStreamWaitEvent(cs(st_), (cudaEvent_t) ev_in_[c % 2], 0), "read");
                if (il < n_layer_) {
                    layer(il, call.m);
                } else {
                    output_rows(0, call.m, hn_ + n_embd_);
                    draft_rows(tokens + call.at, ahead_pos_ == pos_ && tokens[call.at] == ahead_tok_ ? 1 : 0, call.m);
                    ++read_draft_calls_;
                }
                if (uint8_t* state = call.mark >= 0 ? marks[call.mark].state : nullptr) {
                    ck(cudaStreamSynchronize(cs(st_)), "read");
                    const int64_t pos = pos_ + call.m;
                    if (il < n_layer_) save_layer(il, pos, state);
                    if (k == turns - 1) {   // what no turn is for: a layer behind the model's, the last output
                        for (int j = n_layer_; j < (int) layers_.size(); ++j) save_layer(j, pos, state);
                        save_last(state);
                    }
                }
                // The card's work on the next call starts now.  This call's rows leave beside it, unless this was
                // the last turn, and the half is the next call's that uses it: the one two on, or the one with
                // this half at the next turn.  After the last turn the next card reads them.
                const bool last = k == turns - 1, out = last && next_;
                if (!host || (last && c + 2 >= n_calls && !out)) continue;
                ck(cudaEventRecord((cudaEvent_t) ev_out_[c % 2], cs(st_)), "read");
                ck(cudaStreamWaitEvent(cs(rs_), (cudaEvent_t) ev_out_[c % 2], 0), "read");
                if (!last || out)
                    ck(cudaMemcpyAsync(home(c), half(c), (size_t) call.m * r_bytes, cudaMemcpyDeviceToHost,
                                       cs(rs_)),
                       "read");
                if (!last || c + 2 < n_calls) fetch(c + 2 < n_calls ? c + 2 : c % 2);
            }
        }
        if (host) ck(cudaStreamSynchronize(cs(rs_)), "read");
        if (next_ && !host) {   // the rows are in r_, a call's in each half
            float* h = pass_rows(n);
            for (int c = 0; c < n_calls; ++c)
                ck(cudaMemcpyAsync(h + (size_t) calls_[(size_t) c].at * r_row, half(c), calls_[(size_t) c].m * r_bytes,
                                   cudaMemcpyDeviceToHost, cs(st_)),
                   "the next card");
            ck(cudaStreamSynchronize(cs(st_)), "the next card");
        }
    } catch (...) {
        if (host) cudaStreamSynchronize(cs(rs_));
        shared_ahead_ = false;
        r_ = r0;
        pos_ = pos0;
        throw;
    }
    shared_ahead_ = false;
    r_ = r0;
    pos_ = pos0;
    if (next_) {
        next_->pass_ = host ? h_res_ : h_pass_;
        next_->pass_at_ = pos0;
        std::vector<Mark> on(marks, marks + n_marks);   // the next card's part of each state is after this one's
        for (Mark& m : on)
            if (m.state) m.state += card_state_bytes();
        if (!next_->read(tokens, n, on.data(), n_marks, asked)) return false;
    }
    pos_ = pos0 + n;
    stats_.tokens += n;
    return true;
}

// The tokens of a window of t that leave the recurrent state as it is after them: each but the last when there
// is a place for each, else as many as there are places, the first token among them and the others spread over
// the window.
unsigned Engine::snap_tokens(int t) const {
    const int most = t - 1;
    if (most <= snap_places_) return (1u << most) - 1u;
    unsigned bits = 0;
    for (int i = 0; i < snap_places_; ++i) bits |= 1u << (i * most / snap_places_);
    return bits;
}

// The recurrent layers are the only ones a wrong guess would spoil: every other layer's row of a position is
// written again when the position gets its real token.  So they leave the state as it was after tokens of the
// window (snap_, snap_tokens()), and keep() puts back the one after the last token that was right.
void Engine::window(const int32_t* tokens, int t) {
    if (t < 1 || t > opt_.window) throw std::logic_error("a window takes 1 to max_window() tokens");
    OnCard here(device_, split_);
    // the two ways are a choice only with experts on the CPU; a call that records launches says nothing of them
    const bool timed = t > 1 && !all_cached();
    const bool recorded = std::exchange(recorded_, false);
    if (timed) halves_ = pick_way(t);
    const Clock::time_point t0 = Clock::now();
    windowed_ = true;
    snap_bits_ = snap_tokens(t);
    snap_at_ = snap_;
    try {
        run(tokens, t);
    } catch (...) {
        windowed_ = false;
        snap_bits_ = 0;
        recorded_ = recorded_ || recorded;
        throw;
    }
    windowed_ = false;
    snap_bits_ = 0;
    std::copy(tokens, tokens + t, win_tok_);
    if (next_) {   // a card's way is timed by its own layers
        hand_on(pos_, t, pass_rows(t));
        if (timed && !recorded_) took_way(t, ms_since(t0));
        recorded_ = recorded_ || recorded;
        win_ = t;
        next_->window(tokens, t);
        return;
    }
    float* h = draft_ ? hn_ + n_embd_ : xn_;
    output_rows(0, t, h);
    lin(output_, h, wlogits_, t);
    finish_logits(wlogits_, t);
    ck(cudaMemcpyAsync(h_wlogits_, wlogits_, (size_t) t * n_vocab_ * sizeof(float), cudaMemcpyDeviceToHost, cs(st_)),
       "logits");
    ck(cudaStreamSynchronize(cs(st_)), "window");
    if (timed && !recorded_) took_way(t, ms_since(t0));
    recorded_ = recorded_ || recorded;
    win_ = t;
}

// How a window of t tokens goes through the layers: in halves, or its tokens together.  Each way is timed a few
// times, then the one that took less is taken.  The other still gets a window now and then: what each takes
// moves with the experts the cache holds and with what else the machine is doing.
bool Engine::pick_way(int t) {
    const Way& whole = way_[0][t];
    const Way& split = way_[1][t];
    if (split.n < kWayFirst || whole.n < kWayFirst) return split.n < whole.n;   // in turn, the whole way first
    const bool halves = split.ms < whole.ms;
    if (++way_calls_[t] < std::min(kWayAgain << way_backoff_[t], kWayAgainMost)) return halves;
    way_calls_[t] = 0;
    if ((kWayAgain << way_backoff_[t]) < kWayAgainMost) ++way_backoff_[t];
    return !halves;
}

void Engine::took_way(int t, double ms) {
    Way& w = way_[halves_ ? 1 : 0][t];
    const Way& other = way_[halves_ ? 0 : 1][t];
    const bool behind = other.n >= kWayFirst && w.ms > other.ms;
    w.ms = w.n < kWayFirst ? (w.ms * w.n + ms) / (w.n + 1) : w.ms + (behind ? kWayBehind : kWayWeight) * (ms - w.ms);
    ++w.n;
    if (behind && w.ms < other.ms) way_calls_[t] = way_backoff_[t] = 0;   // the other way is behind now
}

void Engine::keep(int n) {
    if (n < 1 || n > win_) throw std::logic_error("keep takes 1 to the window's tokens");
    OnCard here(device_, split_);
    // The next card reads the window's rows again for the tokens it puts through its layers again: a right token's
    // row is the same, whatever came after it.
    if (n < win_) {
        // The state after the last right token, when that token left it.  Else the one of the last token before
        // it that did, and the right tokens after that one go through the layers again: they come out as they
        // did, and the state with them.
        const unsigned left = snap_tokens(win_) & ((1u << n) - 1u);   // the right tokens that left their state
        const int from = (int) std::bit_width(left);                  // the tokens the state at hand is after
        ck(cudaMemcpyAsync(state_, snap_ + size_t(std::popcount(left) - 1) * rec_n_, rec_n_ * sizeof(float),
                           cudaMemcpyDeviceToDevice, cs(st_)),
           "keep");
        if (from < n) {
            pos_ += from;
            try {
                run(win_tok_ + from, n - from);
            } catch (...) {
                pos_ -= from;
                throw;
            }
            pos_ -= from;
        }
    }
    if (draft_) draft_rows(win_tok_, ahead_pos_ == pos_ && win_tok_[0] == ahead_tok_ ? 1 : 0, n);
    if (next_) next_->keep(n);
    pos_ += n;
    stats_.tokens += n;
    win_ = 0;
}

// The draft block's input for n tokens: each token's embedding and the model's output for the token before it
// (h, a row a token), each normed, the two side by side through the block's projection.  Returns the block's
// residual, and leaves its normed attention input in xn_ and xq_.  r_ is free between the model's calls, and the
// block works in it: n rows of the two halves, then the residual.
//
// h_r_ is written once the card has read the last copy out of it (ev_hr_): read()'s draft turn calls this for one
// call after another without waiting for the card in between.
float* Engine::draft_input(const int32_t* tokens, int64_t row, const float* h, int n) {
    const size_t e = (size_t) n_embd_, rows = (size_t) n;
    // read()'s draft turn calls this for one call after another without waiting for the card: the copy of the call
    // before may still be reading h_r_, so the host waits for it before writing the next call's rows
    if (!ev_hr_) {
        cudaEvent_t ev = nullptr;
        ck(cudaEventCreateWithFlags(&ev, cudaEventDisableTiming), "cudaEventCreate");
        ev_hr_ = ev;
    } else {
        ck(cudaEventSynchronize((cudaEvent_t) ev_hr_), "embed");
    }
    for (int t = 0; t < n; ++t) input(tokens[t], row + t, h_r_ + (size_t) t * e);
    if (n < kStageRows) copy2(h_r_, cur_, n * n_embd_, nullptr, nullptr, 0, st_);
    else ck(cudaMemcpyAsync(cur_, h_r_, rows * e * sizeof(float), cudaMemcpyHostToDevice, cs(st_)), "embed");
    ck(cudaEventRecord((cudaEvent_t) ev_hr_, cs(st_)), "embed");
    rms_norm(cur_, fp(d_enorm_), r_, n_embd_, n, rms_eps_, st_, 2 * n_embd_);
    rms_norm(h, fp(d_hnorm_), r_ + e, n_embd_, n, rms_eps_, st_, 2 * n_embd_);
    float* x = r_ + 2 * rows * e;
    lin(d_proj_, r_, x, n);
    rms_norm(x, fp(layers_[(size_t) n_layer_].attn_norm), xn_, n_embd_, n, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_ * n, 1, st_);
    return x;
}

// The rows the draft block's attention keeps for tokens [first, n) of a call that has run: token i is at pos() + i,
// and hn_'s row i is the model's output for the token before it.  A row is all a later guess reads of a token, so
// the rest of the block is not computed for it.  Then the call's last output is the one the next token pairs with.
void Engine::draft_rows(const int32_t* tokens, int first, int n) {
    const size_t e = (size_t) n_embd_;
    if (block_layer_ && first < n) {
        draft_input(tokens + first, pos_ + first, hn_ + (size_t) first * e, n - first);
        block_rows(pos_ + first, n - first);
    }
    copy2(hn_ + (size_t) n * e, hn_, n_embd_, nullptr, nullptr, 0, st_);
    ahead_pos_ = -1;
}

// One token through the draft block, at pos() + i.  The first token's row is then what draft_rows() would leave
// for it, and eval() and keep() skip it; the rows of the guesses after it are written again when the positions get
// their real tokens.
void Engine::draft_step(int32_t token, int i) {
    Layer& L = layers_[(size_t) n_layer_];
    float* own = hn_ + n_embd_;   // the block's output, which stands in for the model's from the second guess on
    float* x = draft_input(&token, pos_ + i, i == 0 ? hn_ : own, 1);
    block_attend(pos_ + i, 1);
    axpy(ao_, 1.f, x, n_embd_, true, st_);
    rms_norm(x, fp(L.ffn_norm), xn_, n_embd_, 1, rms_eps_, st_);
    kernels::native_quantize_q8_1(xn_, xq_, n_embd_, 1, st_);
    moe_router(L, 1);
    ck(cudaEventRecord((cudaEvent_t) ev_, cs(st_)), "router");
    moe_shared(L, n_layer_, 1);
    tok_ = nullptr;   // a guess's row has no token of the sequence
    if (moe(L, n_layer_, 1)) moe_add(1);
    axpy(fo_, 1.f, x, n_embd_, true, st_);
    rms_norm(x, fp(d_head_), own, n_embd_, 1, rms_eps_, st_);
    Dev head = output_;   // the matrix's first rows are the common tokens'
    head.n_out = draft_rows();
    lin(head, own, logits_, 1);
    finish_logits(logits_, 1);
}

int32_t Engine::draft(int32_t token, int i, float* prob) {
    if (next_) return next_->draft(token, i, prob);   // the last card has the draft block
    OnCard here(device_, split_);
    if (!draft_) throw std::logic_error("draft: this model has no block that guesses ahead");
    if (token < 0 || token >= n_vocab_) throw std::runtime_error("a token id is outside the vocabulary");
    if (i < 0 || pos_ + i >= opt_.max_context) throw std::logic_error("draft: the position is outside the context");
    cover(pos_ + i + 1);
    ck(cudaStreamSynchronize(cs(st_)), "draft");
    draft_step(token, i);
    // The guess and its probability are found on the card.  On the host a vocabulary's softmax takes up to 2 ms
    // where the block is unsure, four times what the block takes.
    likeliest(logits_, draft_rows(), h_guess_, st_);
    ck(cudaStreamSynchronize(cs(st_)), "draft");
    if (i == 0) {
        ahead_pos_ = pos_;
        ahead_tok_ = token;
    }
    *prob = h_guess_->prob;
    return h_guess_->token;
}

int most_likely(const float* logits, int n) {
    const float* at = std::find(logits, logits + n, largest(logits, n));
    return at == logits + n ? 0 : (int) (at - logits);
}

}  // namespace strata::poly
