#include "stage_copy.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <cstdlib>
#include <vector>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#elif defined(__unix__) || defined(__APPLE__)
#include <sys/mman.h>
#include <unistd.h>
#endif

namespace {
constexpr size_t kAlignment = 64;
constexpr unsigned char kGuard = 0xd3;

unsigned char* aligned(std::vector<unsigned char>& storage) {
    const auto address = reinterpret_cast<uintptr_t>(storage.data());
    return reinterpret_cast<unsigned char*>((address + kAlignment - 1) & ~(uintptr_t(kAlignment) - 1));
}

void check(bool ok, size_t n, size_t from, size_t to) {
    if (ok) return;
    std::fprintf(stderr, "FAIL n=%zu src=%zu dst=%zu\n", n, from, to);
    std::exit(1);
}

// Guard bytes catch writes outside the destination. Explicit alignment makes each
// offset an actual cache-line residue, independent of the allocator's alignment.
size_t check_offsets(size_t n, size_t step) {
    std::vector<unsigned char> source(n + 4 * kAlignment), destination(source.size());
    auto* s = aligned(source) + kAlignment;
    auto* d = aligned(destination) + kAlignment;
    for (size_t i = 0; i < source.size(); ++i) source[i] = (i * 37 + n) % 251;
    const auto before = source;
    size_t cases = 0;
    for (size_t from = 0; from < kAlignment; from += step) {
        for (size_t to = 0; to < kAlignment; to += step) {
            std::fill(destination.begin(), destination.end(), kGuard);
            auto expected = destination;
            std::copy_n(s + from, n, expected.data() + (d - destination.data()) + to);
            strata::poly::stage_copy_streaming(d + to, s + from, n);
            check(destination == expected && source == before, n, from, to);
            ++cases;
        }
    }
    return cases;
}

#if defined(_WIN32) || defined(__unix__) || defined(__APPLE__)
// A readable guard byte cannot detect an over-read. Put both ends of the usable
// pages against inaccessible pages, then test a copy flush with either end.
class GuardedBuffer {
public:
    explicit GuardedBuffer(size_t bytes) {
#if defined(_WIN32)
        SYSTEM_INFO info;
        GetSystemInfo(&info);
        page_ = info.dwPageSize;
#else
        const long page = sysconf(_SC_PAGESIZE);
        check(page > 0, bytes, 0, 0);
        page_ = static_cast<size_t>(page);
#endif
        usable_ = ((bytes + page_ - 1) / page_) * page_;
        total_ = usable_ + 2 * page_;
#if defined(_WIN32)
        mapping_ = static_cast<unsigned char*>(VirtualAlloc(nullptr, total_, MEM_RESERVE, PAGE_NOACCESS));
        check(mapping_ != nullptr, bytes, 0, 0);
        check(VirtualAlloc(mapping_ + page_, usable_, MEM_COMMIT, PAGE_READWRITE) != nullptr, bytes, 0, 0);
#else
        void* mapping = mmap(nullptr, total_, PROT_NONE, MAP_PRIVATE | MAP_ANONYMOUS, -1, 0);
        check(mapping != MAP_FAILED, bytes, 0, 0);
        mapping_ = static_cast<unsigned char*>(mapping);
        check(mprotect(mapping_ + page_, usable_, PROT_READ | PROT_WRITE) == 0, bytes, 0, 0);
#endif
    }
    ~GuardedBuffer() {
#if defined(_WIN32)
        VirtualFree(mapping_, 0, MEM_RELEASE);
#else
        munmap(mapping_, total_);
#endif
    }
    GuardedBuffer(const GuardedBuffer&) = delete;
    GuardedBuffer& operator=(const GuardedBuffer&) = delete;
    unsigned char* at(size_t bytes, bool end) { return mapping_ + page_ + (end ? usable_ - bytes : 0); }
private:
    unsigned char* mapping_ = nullptr;
    size_t page_ = 0, usable_ = 0, total_ = 0;
};

size_t check_protected_ends() {
    size_t cases = 0;
    // Traverse every tail residue above the streaming threshold and both sides
    // of the 64/128-byte loop boundaries. Small lengths also cover the fallback.
    for (size_t n = 1; n <= 4353; ++n) {
        if (n > 129 && n < 4095) continue;
        GuardedBuffer source(n), destination(n);
        for (bool source_end : {false, true}) {
            for (bool destination_end : {false, true}) {
                auto* s = source.at(n, source_end);
                auto* d = destination.at(n, destination_end);
                for (size_t i = 0; i < n; ++i) s[i] = (i * 37 + n) % 251;
                const std::vector<unsigned char> before(s, s + n);
                std::fill_n(d, n, kGuard);
                strata::poly::stage_copy_streaming(d, s, n);
                check(std::equal(before.begin(), before.end(), d) && std::equal(before.begin(), before.end(), s),
                      n, source_end, destination_end);
                ++cases;
            }
        }
    }
    return cases;
}
#endif
} // namespace

int main() {
    constexpr std::array<size_t, 21> lengths = {
        0, 1, 15, 16, 17, 63, 64, 65, 127, 128, 129,
        4095, 4096, 4097, 4159, 4160, 4161, 4223, 4224, 4225, 65535
    };
    size_t offsets = 0;
    for (size_t n : lengths) offsets += check_offsets(n, 1);
    // Real expert sizes; a sparse alignment sweep keeps this host-only test fast.
    for (size_t n : {size_t(589824), size_t(720896)}) offsets += check_offsets(n, 15);
    size_t protected_ends = 0;
#if defined(_WIN32) || defined(__unix__) || defined(__APPLE__)
    protected_ends = check_protected_ends();
#else
    std::puts("SKIP protected-page checks: no platform implementation");
#endif
    std::printf("PASS %zu aligned-offset cases, %zu protected-end cases\n", offsets, protected_ends);
    return 0;
}
