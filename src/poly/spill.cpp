#include "spill.hpp"

#include <algorithm>
#include <atomic>
#include <condition_variable>
#include <cstdlib>
#include <cwchar>
#include <deque>
#include <functional>
#include <new>
#include <stdexcept>
#include <thread>
#include <vector>

#ifdef _WIN32
#ifndef NOMINMAX
#define NOMINMAX
#endif
#include <windows.h>
#else
#include <fcntl.h>
#include <sys/mman.h>
#include <sys/vfs.h>
#include <unistd.h>
#endif

namespace strata::poly {
namespace {

// The thread that writes what release() hands it and closes the files of what is gone, one job at a time.  It lives
// as long as the process, and so does what it holds: nothing of it is destroyed at exit while it may still run.
struct Writer {
    std::mutex m;
    std::condition_variable cv;
    std::deque<std::function<void()>> jobs;

    Writer() {
        std::thread([this] {
            for (;;) {
                std::function<void()> job;
                {
                    std::unique_lock<std::mutex> l(m);
                    cv.wait(l, [&] { return !jobs.empty(); });
                    job = std::move(jobs.front());
                    jobs.pop_front();
                }
                job();
            }
        }).detach();
    }
    void add(std::function<void()> job) {
        {
            std::lock_guard<std::mutex> l(m);
            jobs.push_back(std::move(job));
        }
        cv.notify_one();
    }
};

Writer& writer() {
    static Writer* w = new Writer;
    return *w;
}

// A temporary file that goes when it is closed, or -1.  On Linux $TMPDIR, else /var/tmp, which outlives a reboot and
// so is on a disk: a directory in RAM would gain nothing.
intptr_t make_file() {
#ifdef _WIN32
    // Made and opened in one call: GetTempFileNameW makes the file first, and a process ended before it was opened
    // to go on close left it behind.
    static std::atomic<unsigned> count{0};
    wchar_t dir[MAX_PATH + 1], path[MAX_PATH + 64];
    if (!GetTempPathW(MAX_PATH, dir)) return -1;
    for (int tries = 0; tries < 16; ++tries) {
        swprintf(path, MAX_PATH + 64, L"%lsstrata-poly-%lu-%u.tmp", dir, GetCurrentProcessId(), count++);
        const HANDLE f = CreateFileW(path, GENERIC_READ | GENERIC_WRITE, 0, nullptr, CREATE_NEW,
                                     FILE_ATTRIBUTE_NOT_CONTENT_INDEXED | FILE_FLAG_DELETE_ON_CLOSE, nullptr);
        if (f != INVALID_HANDLE_VALUE) return (intptr_t) f;
        if (GetLastError() != ERROR_FILE_EXISTS) return -1;
    }
    return -1;
#else
    const char* dirs[] = {std::getenv("TMPDIR"), "/var/tmp", "/tmp"};
    for (const char* dir : dirs) {
        struct statfs fs;
        if (!dir || statfs(dir, &fs) != 0 || fs.f_type == 0x01021994 /* tmpfs */ || fs.f_type == 0x858458f6 /* ramfs */)
            continue;
        const int fd = open(dir, O_TMPFILE | O_RDWR | O_CLOEXEC, 0600);   // nameless
        if (fd >= 0) return fd;
    }
    return -1;
#endif
}

void close_file(intptr_t f) {
#ifdef _WIN32
    CloseHandle((HANDLE) f);
#else
    close((int) f);
#endif
}

bool write_all(intptr_t f, const uint8_t* p, size_t n) {
    for (size_t done = 0; done < n;) {
#ifdef _WIN32
        DWORD got = 0;
        if (!WriteFile((HANDLE) f, p + done, (DWORD) std::min<size_t>(n - done, size_t(1) << 30), &got, nullptr) ||
            got == 0)
            return false;
#else
        const ssize_t got = pwrite((int) f, p + done, n - done, (off_t) done);
        if (got <= 0) return false;
#endif
        done += (size_t) got;
    }
#ifdef __linux__
    sync_file_range((int) f, 0, (off_t) n, SYNC_FILE_RANGE_WRITE);   // to the disk now, not in 30 s
#endif
    return true;
}

// The pages of [p, p + n) brought into the process by several threads at once.  The copy that reads them takes them a
// fault at a time, on one thread: 1.1 GB of a parked conversation came back in 234 ms where the heap's took 121.
void bring_in(const uint8_t* p, size_t n) {
    const auto touch = [](const uint8_t* a, size_t m) {
        volatile uint8_t seen = 0;
        for (size_t i = 0; i < m; i += 4096) seen = (uint8_t) (seen + a[i]);
    };
    constexpr size_t kPer = size_t(64) << 20;
    const size_t k = std::min<size_t>(8, (n + kPer - 1) / kPer);
    const size_t per = (n / std::max<size_t>(k, 1) + 4095) & ~size_t(4095);
    std::vector<std::thread> more;
    for (size_t i = 1; i < k && i * per < n; ++i) more.emplace_back(touch, p + i * per, std::min(per, n - i * per));
    touch(p, std::min(per, n));
    for (std::thread& t : more) t.join();
}

}  // namespace

Spill::Spill(size_t bytes) : n_(bytes) { heap_ = new (std::nothrow) uint8_t[bytes]; }

// The file and its mapping go on the writer's thread: the close lets go of the file's pages and its disk blocks, which
// a request on its way in need not wait for.
Spill::~Spill() {
    delete[] heap_;
    if (file_ == -1) return;
    writer().add([view = view_, n = n_, mapping = mapping_, file = file_] {
#ifdef _WIN32
        if (view) UnmapViewOfFile(view);
        if (mapping) CloseHandle(mapping);
#else
        (void) mapping;
        if (view) munmap(view, n);
#endif
        close_file(file);   // and the file goes with it
    });
}

uint8_t* Spill::data() const {
    std::lock_guard<std::mutex> l(m_);
    used_ = true;
    if (heap_ || !filed_) return heap_;   // null: no memory for it
    if (!view_) map();
    if (cold_) bring_in(view_, n_);
    cold_ = false;
    return view_;
}

void Spill::release() const {
    std::lock_guard<std::mutex> l(m_);
    const bool used = used_;
    used_ = false;
    if (heap_) {
        if (filed_) {   // written out while a request used it
            delete[] heap_;
            heap_ = nullptr;
        } else if (!queued_ && n_ > 0) {
            queued_ = true;
            writer().add([s = shared_from_this()] { s->write(); });
        }
        return;
    }
    if (view_ && used) drop();
}

// Nothing writes or frees the heap while it is queued and not filed: it is read without the lock.  A file that
// cannot be made or written leaves it in the heap, and it is not tried again.
void Spill::write() const {
    const intptr_t f = make_file();
    const bool ok = f != -1 && write_all(f, heap_, n_);
    std::lock_guard<std::mutex> l(m_);
    if (!ok) {
        if (f != -1) close_file(f);
        return;
    }
    file_ = f;
    filed_ = true;
    if (!used_) {
        delete[] heap_;
        heap_ = nullptr;
    }
}

void Spill::map() const {
#ifdef _WIN32
    mapping_ = CreateFileMappingW((HANDLE) file_, nullptr, PAGE_READONLY, 0, 0, nullptr);
    if (mapping_) view_ = (uint8_t*) MapViewOfFile(mapping_, FILE_MAP_READ, 0, 0, n_);
#else
    void* p = mmap(nullptr, n_, PROT_READ, MAP_SHARED, (int) file_, 0);
    if (p != MAP_FAILED) view_ = (uint8_t*) p;
#endif
    if (!view_) throw std::runtime_error("the conversation cache could not map its temporary file");
    cold_ = true;
}

void Spill::drop() const {
#ifdef _WIN32
    VirtualUnlock(view_, n_);   // pages that are not locked leave the working set (and it returns FALSE)
#else
    static const size_t page = (size_t) sysconf(_SC_PAGESIZE);
    madvise(view_, (n_ + page - 1) / page * page, MADV_DONTNEED);   // they stay in the file cache
#endif
    cold_ = true;
}

}  // namespace strata::poly
