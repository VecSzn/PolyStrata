// Host-side expert I/O. The engine owns mappings and staging slots until the queue has stopped.
#pragma once

#include "cpu_experts.hpp"

#include <algorithm>
#include <array>
#include <atomic>
#include <cerrno>
#include <chrono>
#include <condition_variable>
#include <cstdint>
#include <cstdlib>
#include <deque>
#include <functional>
#include <immintrin.h>
#include <limits>
#include <map>
#include <mutex>
#include <span>
#include <thread>
#include <vector>

#if defined(_WIN32)
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#elif defined(__linux__)
#include <fcntl.h>
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace strata::poly {

struct ExpertRange {
    const uint8_t* data;
    size_t bytes;
};

inline bool expert_prefetch_enabled() {
    const char* v = std::getenv("STRATA_POLY_PREFETCH_EXPERTS");
    return v && std::atoi(v) == 1;
}

inline void unique_expert_ranges(std::vector<ExpertRange>& ranges) {
    std::erase_if(ranges, [](const ExpertRange& r) {
        return !r.data || !r.bytes || r.bytes > std::numeric_limits<uintptr_t>::max() - (uintptr_t) r.data;
    });
    std::sort(ranges.begin(), ranges.end(), [](const ExpertRange& a, const ExpertRange& b) {
        return (uintptr_t) a.data < (uintptr_t) b.data || (a.data == b.data && a.bytes < b.bytes);
    });
    ranges.erase(std::unique(ranges.begin(), ranges.end(), [](const ExpertRange& a, const ExpertRange& b) {
        return a.data == b.data && a.bytes == b.bytes;
    }), ranges.end());
}

// Start each Linux hint on a page boundary. Small chunks ask for the whole matrix even when the kernel's
// readahead window is smaller than it. No rounding past the last byte is needed: madvise rounds the length.
template <class Hint>
void expert_hint_chunks(const ExpertRange& r, size_t page, Hint&& hint) {
    if (!page || !r.data || !r.bytes || r.bytes > std::numeric_limits<uintptr_t>::max() - (uintptr_t) r.data) return;
    const uintptr_t end = (uintptr_t) r.data + r.bytes;
    const size_t step = std::max(page, (size_t(128) << 10) / page * page);
    for (uintptr_t at = (uintptr_t) r.data / page * page; at < end;) {
        const size_t bytes = std::min<size_t>(step, end - at);
        hint((const uint8_t*) at, bytes);
        at += bytes;
    }
}

#if defined(_WIN32)
using ExpertPrefetch = BOOL(WINAPI*)(HANDLE, ULONG_PTR, PWIN32_MEMORY_RANGE_ENTRY, ULONG);
inline ExpertPrefetch expert_prefetch_api() {
    static const auto fn = (ExpertPrefetch) (void*) GetProcAddress(GetModuleHandleW(L"kernel32.dll"),
                                                                  "PrefetchVirtualMemory");
    return fn;
}
#endif

inline bool expert_prefetch_available() {
#if defined(_WIN32)
    return expert_prefetch_api() != nullptr;
#elif defined(__linux__)
    return true;
#else
    return false;
#endif
}

// Only mapped sources belong here. A staging destination already has a queued read and completion state.
// Hints never replace reads, alter weights, or inherit the other engine's STRATA_READ_AHEAD setting.
inline void prefetch_expert_ranges(std::span<const ExpertRange> ranges) {
#if defined(_WIN32)
    const auto fn = expert_prefetch_api();
    if (!fn || ranges.empty()) return;
    thread_local std::vector<WIN32_MEMORY_RANGE_ENTRY> native;
    native.clear();
    for (const auto& r : ranges) native.push_back({(PVOID) r.data, (SIZE_T) r.bytes});
    (void) fn(GetCurrentProcess(), (ULONG_PTR) native.size(), native.data(), 0);
#elif defined(__linux__)
    static const long page = sysconf(_SC_PAGESIZE);
    if (page <= 0) return;
    for (const auto& r : ranges)
        expert_hint_chunks(r, (size_t) page, [](const uint8_t* p, size_t n) {
            (void) madvise((void*) p, n, MADV_WILLNEED);
        });
#else
    (void) ranges;
#endif
}

template <class Mapped, class Launch, class Hint>
void prepare_expert_prefetch(bool enabled, std::span<const CpuExpert> experts, int rows,
                            const std::array<size_t, 3>& bytes, Mapped&& mapped, Launch&& launch, Hint&& hint) {
    if (!enabled || rows <= 0 || rows > 8 || experts.empty()) return;
    thread_local std::vector<ExpertRange> ranges;
    ranges.clear();
    for (const CpuExpert& e : experts) {
        if (!mapped(e)) continue;
        ranges.push_back({e.gate, bytes[0]});
        ranges.push_back({e.up, bytes[1]});
        ranges.push_back({e.down, bytes[2]});
    }
    unique_expert_ranges(ranges);
    if (ranges.empty()) return;
    launch();
    hint(std::span<const ExpertRange>(ranges));
}

// Each read thread owns its file handles. A short/failed read returns false; the engine must not publish its
// partially written slot and retries it before falling back to a copy from the still-owned mapping.
inline bool read_expert_file(const std::string& path, uint64_t offset, uint8_t* dst, size_t bytes) {
#if defined(_WIN32)
    struct Handles {
        std::map<std::string, HANDLE> files;
        ~Handles() { for (auto& [_, h] : files) if (h != INVALID_HANDLE_VALUE) CloseHandle(h); }
    };
    thread_local Handles mine;
    auto it = mine.files.find(path);
    if (it == mine.files.end())
        it = mine.files.emplace(path, CreateFileA(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr,
                                                 OPEN_EXISTING, 0, nullptr)).first;
    if (it->second == INVALID_HANDLE_VALUE) return false;
    for (size_t done = 0; done < bytes;) {
        OVERLAPPED o{};
        o.Offset = (DWORD) (offset + done);
        o.OffsetHigh = (DWORD) ((offset + done) >> 32);
        DWORD got = 0;
        const DWORD want = (DWORD) std::min<size_t>(bytes - done, size_t(1) << 30);
        if (!ReadFile(it->second, dst + done, want, &got, &o) || !got) return false;
        done += got;
    }
#elif defined(__linux__)
    struct Handles {
        std::map<std::string, int> files;
        ~Handles() { for (auto& [_, fd] : files) if (fd >= 0) ::close(fd); }
    };
    thread_local Handles mine;
    auto it = mine.files.find(path);
    if (it == mine.files.end()) it = mine.files.emplace(path, ::open(path.c_str(), O_RDONLY | O_CLOEXEC)).first;
    if (it->second < 0) return false;
    for (size_t done = 0; done < bytes;) {
        const ssize_t got = ::pread(it->second, dst + done, bytes - done, (off_t) (offset + done));
        if (got < 0 && errno == EINTR) continue;
        if (got <= 0) return false;
        done += (size_t) got;
    }
#else
    (void) path; (void) offset; (void) dst; (void) bytes;
    return false;
#endif
    return true;
}

struct ExpertReadJob {
    uint8_t* dst;
    const uint8_t* src;
    size_t n;
    int32_t s;   // engine's slot, or a negative page-work operation
};

// The existing pool's scheduling: demand jobs first, predicted reads next, page maintenance last.
// A waiting CPU worker can help with a read. Stop discards queued work and joins every in-flight callback.
class ExpertReadQueue {
public:
    enum class Priority { demand, predicted, pages };
    using Run = std::function<void(const ExpertReadJob&)>;

    ExpertReadQueue(int threads, Run run) : run_(std::move(run)) {
        try {
            for (int i = 0; i < threads; ++i) threads_.emplace_back([this] { loop(); });
        } catch (...) {
            stop();
            throw;
        }
    }
    ~ExpertReadQueue() { stop(); }
    ExpertReadQueue(const ExpertReadQueue&) = delete;
    ExpertReadQueue& operator=(const ExpertReadQueue&) = delete;

    void submit(std::span<const ExpertReadJob> jobs, Priority priority) {
        if (jobs.empty()) return;
        {
            std::lock_guard<std::mutex> l(m_);
            if (quit_) return;
            auto& q = priority == Priority::pages ? pages_ : reads_;
            q.insert(priority == Priority::demand ? q.begin() : q.end(), jobs.begin(), jobs.end());
            count_.fetch_add((int) jobs.size(), std::memory_order_release);
        }
        cv_.notify_all();
    }

    bool help() {
        ExpertReadJob job;
        {
            std::lock_guard<std::mutex> l(m_);
            if (quit_ || reads_.empty()) return false;
            job = reads_.front();
            reads_.pop_front();
            count_.fetch_sub(1, std::memory_order_relaxed);
        }
        run_(job);
        return true;
    }

    // The owner stops submitting/helping before its storage is released. A producer still shutting down may
    // submit after stop; it is ignored, as no consumer may touch that storage again.
    void stop() {
        {
            std::lock_guard<std::mutex> l(m_);
            quit_ = true;
        }
        cv_.notify_all();
        for (auto& t : threads_) if (t.joinable()) t.join();
        std::lock_guard<std::mutex> l(m_);
        reads_.clear();
        pages_.clear();
        count_.store(0, std::memory_order_relaxed);
    }

private:
    void loop() {
        for (;;) {
            const auto start = std::chrono::steady_clock::now();
            while (count_.load(std::memory_order_acquire) == 0 &&
                   std::chrono::steady_clock::now() - start < std::chrono::milliseconds(2)) _mm_pause();
            ExpertReadJob job;
            {
                std::unique_lock<std::mutex> l(m_);
                cv_.wait(l, [&] { return quit_ || !reads_.empty() || !pages_.empty(); });
                if (quit_) return;
                auto& q = reads_.empty() ? pages_ : reads_;
                job = q.front();
                q.pop_front();
                count_.fetch_sub(1, std::memory_order_relaxed);
            }
            run_(job);
        }
    }

    Run run_;
    std::mutex m_;
    std::condition_variable cv_;
    std::deque<ExpertReadJob> reads_, pages_;
    std::atomic<int> count_{0};
    bool quit_ = false;
    std::vector<std::thread> threads_;
};

}  // namespace strata::poly
