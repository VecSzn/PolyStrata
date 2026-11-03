// Real mapped files and the same read queue the engine uses; no model, GPU or ggml required.
#include "expert_io.hpp"
#ifdef STRATA_TEST_EXPERT_ENGINE
#include "qwen35.hpp"
#endif

#include <cstdio>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <future>
#include <semaphore>
#include <stdexcept>

using namespace strata::poly;
using Priority = ExpertReadQueue::Priority;

static void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }

struct File {
    std::filesystem::path path;
    std::vector<uint8_t> bytes;
    uint8_t* mapped = nullptr;
#ifdef _WIN32
    HANDLE file = INVALID_HANDLE_VALUE, mapping = nullptr;
#else
    int fd = -1;
#endif
    File() : bytes(4 * 1024 * 1024) {
        path = std::filesystem::temp_directory_path() /
               ("poly-expert-io-" + std::to_string(std::chrono::steady_clock::now().time_since_epoch().count()));
        for (size_t i = 0; i < bytes.size(); ++i) bytes[i] = (uint8_t) ((i * 17 + i / 4096) % 251);
        { std::ofstream f(path, std::ios::binary); f.write((const char*) bytes.data(), (std::streamsize) bytes.size()); }
#ifdef _WIN32
        file = CreateFileW(path.c_str(), GENERIC_READ, FILE_SHARE_READ, nullptr, OPEN_EXISTING, 0, nullptr);
        require(file != INVALID_HANDLE_VALUE, "open mapping");
        mapping = CreateFileMappingW(file, nullptr, PAGE_READONLY, 0, 0, nullptr);
        mapped = (uint8_t*) MapViewOfFile(mapping, FILE_MAP_READ, 0, 0, bytes.size());
#else
        fd = ::open(path.c_str(), O_RDONLY | O_CLOEXEC);
        mapped = (uint8_t*) mmap(nullptr, bytes.size(), PROT_READ, MAP_PRIVATE, fd, 0);
        if (mapped == MAP_FAILED) mapped = nullptr;
#endif
        require(mapped != nullptr, "map fixture");
    }
    ~File() {
#ifdef _WIN32
        if (mapped) UnmapViewOfFile(mapped);
        if (mapping) CloseHandle(mapping);
        if (file != INVALID_HANDLE_VALUE) CloseHandle(file);
#else
        if (mapped) munmap(mapped, bytes.size());
        if (fd >= 0) ::close(fd);
#endif
        std::error_code error;
        std::filesystem::remove(path, error);
    }
};

static void ranges(File& f) {
    const CpuExpert mapped{f.mapped + 17, f.mapped + 20003, f.mapped + 40007, 1.f};
    uint8_t staging[8192]{};
    const CpuExpert staged{staging, staging + 20, staging + 50, 1.f};
    const CpuExpert uses[] = {mapped, staged, mapped};
    const std::array<size_t, 3> sizes{701, 1503, 1907};   // deliberately unequal gate / up
    for (bool enabled : {false, true}) for (bool pool : {false, true}) for (int rows : {1, 8, 9}) {
        int launched = 0, hinted = 0;
        prepare_expert_prefetch(enabled, uses, rows, sizes,
            [&](const CpuExpert& e) { return !pool || e.gate != staging; },
            [&] { ++launched; }, [&](std::span<const ExpertRange> r) {
                require(launched == 1, "GPU work was not launched before hinting");
                ++hinted;
                require(r.size() == (pool ? 3 : 6), "duplicate expert or staging destination hinted");
                for (int i = 0; i < 3; ++i) {
                    const uint8_t* p = i == 0 ? mapped.gate : i == 1 ? mapped.up : mapped.down;
                    require(std::any_of(r.begin(), r.end(), [&](auto a) { return a.data == p && a.bytes == sizes[i]; }),
                            "matrix range uses the wrong quantization's size");
                }
                return false;   // hint failure must not change the computation's source bytes
            });
        require(hinted == (enabled && rows <= 8 ? 1 : 0) && launched == hinted, "prefetch policy or launch count");
    }
    int called = 0;
    prepare_expert_prefetch(true, std::span<const CpuExpert>(&staged, 1), 1, sizes,
        [](const CpuExpert&) { return false; }, [&] { ++called; }, [&](auto) { ++called; });
    require(!called, "a pool-only call must keep its ordinary GPU launch path");

    for (size_t page : {4096u, 65536u}) {
        const ExpertRange r{f.mapped + 123, 300007};
        uintptr_t end = (uintptr_t) r.data / page * page;
        int chunks = 0;
        expert_hint_chunks(r, page, [&](const uint8_t* p, size_t n) {
            require((uintptr_t) p == end && end % page == 0 && n <= 128 * 1024, "hint alignment or coverage");
            end += n;
            ++chunks;
        });
        require(end == (uintptr_t) r.data + r.bytes && chunks >= 3, "hint omitted the matrix's tail");
    }
    std::vector<ExpertRange> invalid{{nullptr, 1}, {f.mapped, 0},
                                    {(const uint8_t*) (std::numeric_limits<uintptr_t>::max() - 1), 8}};
    unique_expert_ranges(invalid);
    require(invalid.empty(), "invalid hint range");
    const ExpertRange real[] = {{f.mapped + 13, 300001}, {f.mapped + 1000007, 4099}};
    prefetch_expert_ranges(real);
    require(std::memcmp(f.mapped, f.bytes.data(), f.bytes.size()) == 0, "OS hint changed mapped contents");
}

static void priorities() {
    std::binary_semaphore entered(0), release(0), done(0);
    std::vector<int> order;
    ExpertReadQueue q(1, [&](const ExpertReadJob& j) {
        if (j.s == 0) { entered.release(); release.acquire(); }
        order.push_back(j.s);
        if (j.s == 9) done.release();
    });
    const ExpertReadJob block[] = {{nullptr, nullptr, 0, 0}};
    q.submit(block, Priority::demand);
    entered.acquire();
    const ExpertReadJob pages[] = {{nullptr, nullptr, 0, 9}};
    const ExpertReadJob ahead[] = {{nullptr, nullptr, 0, 1}, {nullptr, nullptr, 0, 2}};
    const ExpertReadJob demand[] = {{nullptr, nullptr, 0, 3}, {nullptr, nullptr, 0, 4}};
    q.submit(pages, Priority::pages);
    q.submit(ahead, Priority::predicted);
    q.submit(demand, Priority::demand);
    release.release();
    require(done.try_acquire_for(std::chrono::seconds(5)), "read queue stalled");
    q.stop();
    require(order == std::vector<int>({0, 3, 4, 1, 2, 9}), "demand reads must precede predictions and page work");

    int helped = 0;
    ExpertReadQueue manual(0, [&](const ExpertReadJob& j) { helped += j.s; });
    manual.submit(pages, Priority::pages);
    manual.submit(demand, Priority::demand);
    require(manual.help() && manual.help() && !manual.help() && helped == 7, "CPU help stole page work or lost a read");
}

static void reads(File& f) {
    std::vector<uint8_t> dst(f.bytes.size(), 0);
    std::atomic<int> failures{0}, pending{16};
    std::binary_semaphore done(0);
    ExpertReadQueue q(4, [&](const ExpertReadJob& j) {
        const auto offset = (uint64_t) (j.src - f.mapped);
        if (!read_expert_file(f.path.string(), offset, j.dst, j.n)) ++failures;
        if (pending.fetch_sub(1, std::memory_order_acq_rel) == 1) done.release();
    });
    std::vector<ExpertReadJob> jobs;
    for (size_t at = 0; at < dst.size(); at += 256 * 1024)
        jobs.push_back({dst.data() + at, f.mapped + at, 256 * 1024, 0});
    q.submit(jobs, Priority::demand);
    require(done.try_acquire_for(std::chrono::seconds(10)), "file read did not complete");
    q.stop();
    require(failures == 0 && dst == f.bytes, "parallel file reads lost bytes or published early");

    std::binary_semaphore recovered(0);
    std::array<uint8_t, 16> fallback{};
    ExpertReadQueue retry(1, [&](const ExpertReadJob&) {
        require(!read_expert_file(f.path.string(), f.bytes.size() - 2, fallback.data(), fallback.size()),
                "EOF/partial read was accepted as a complete expert");
        require(!read_expert_file(f.path.string() + ".missing", 0, fallback.data(), fallback.size()),
                "failed open was accepted");
        std::memcpy(fallback.data(), f.mapped, fallback.size());
        recovered.release();
    });
    retry.submit(std::span<const ExpertReadJob>(jobs.data(), 1), Priority::demand);
    require(recovered.try_acquire_for(std::chrono::seconds(5)), "read fallback stalled");
    retry.stop();
    require(std::memcmp(fallback.data(), f.bytes.data(), fallback.size()) == 0, "fallback did not restore all bytes");
}

static void shutdown() {
    std::binary_semaphore entered(0), release(0);
    std::atomic<int> finished{0};
    ExpertReadQueue q(1, [&](const ExpertReadJob&) { entered.release(); release.acquire(); ++finished; });
    const ExpertReadJob job[] = {{nullptr, nullptr, 0, 0}};
    q.submit(job, Priority::demand);
    entered.acquire();
    q.submit(job, Priority::predicted);
    auto stopping = std::async(std::launch::async, [&] { q.stop(); });
    require(stopping.wait_for(std::chrono::milliseconds(100)) == std::future_status::timeout,
            "shutdown returned while a read still owned its buffer");
    release.release();
    require(stopping.wait_for(std::chrono::seconds(5)) == std::future_status::ready, "shutdown did not join reader");
    stopping.get();
    q.submit(job, Priority::demand);
    require(!q.help() && finished == 1, "queued work ran after shutdown");
}

#ifdef STRATA_TEST_EXPERT_ENGINE
namespace strata::poly {
struct ExpertIoProbe {
    static void run(File& f) {
        Qwen35 model;
        Engine& e = model;
        e.n_expert_ = 4;
        e.layers_.resize(1);
        Layer& l = e.layers_[0];
        l.gu_bytes = l.gu_stride = 256;
        l.u_bytes = l.u_stride = 128;
        l.d_bytes = 64;
        l.exp_gate = f.mapped;
        l.exp_up = f.mapped + 4096;
        l.exp_down = f.mapped + 8192;
        e.views_.push_back({f.mapped, f.bytes.size(), f.path.string()});
        const size_t slot_bytes = l.gu2() + l.d_bytes;
        e.pool_.reset(new uint8_t[2 * slot_bytes]);
        e.pool_sizes_.push_back({slot_bytes, 0, 0, 2, 0, 0, 1});
        e.pool_size_of_ = {0};
        e.pool_of_.assign(4, -1);
        e.pool_who_.assign(2, -1);
        e.pool_tick_.assign(2, 0);
        e.pool_pin_.assign(2, 0);
        e.pool_ref_.assign(2, 0);
        e.pool_prot_.assign(2, 0);
        e.kept_.assign(4, 0);
        e.pool_state_.reset(new std::atomic<uint8_t>[2]{});
        e.pool_failed_.reset(new std::atomic<uint8_t>[2]{});
        e.pool_pending_.reset(new std::atomic<int>[2]{});
        // No reader threads yet: CPU help drives the real queue deterministically.
        e.pool_reader_ = std::make_unique<ExpertReadQueue>(0, [&](const ExpertReadJob& job) { e.pool_run(job); });
        const auto source = [&](int i) {
            return CpuExpert{l.exp_gate + i * l.gu_stride, l.exp_up + i * l.u_stride,
                             l.exp_down + i * l.d_bytes, 1.f};
        };
        CpuExpert host[] = {source(0), source(0), source(1), source(2)};
        e.pool_take(l, 0, host, 4);
        require(e.pool_reads_.size() == 2 && host[0].gate == host[1].gate,
                "repeated expert did not reuse its pending slot");
        require(host[3].gate == source(2).gate, "exhausted slots must leave the source mapped");
        for (auto r : e.pool_reads_) e.pool_queue(l, r.e, r.s, true);
        require(!e.pool_has(0, 0) && !e.pool_has(0, 1), "incomplete slot was published");
        CpuExpert again[] = {source(0)};
        e.pool_take(l, 0, again, 1);
        require(e.pool_reads_.empty() && again[0].gate == host[0].gate, "pending expert was queued twice");
        int launches = 0;
        const std::function<void()> launch = [&] { ++launches; };
        const std::function<void()>* meanwhile = &launch;
        e.prefetch_experts_ = true;
        e.prefetch_cpu_experts(l, 0, host, 3, 1, meanwhile);
        require(launches == 0 && meanwhile == &launch, "staging-only call issued mapped hints");
        e.prefetch_cpu_experts(l, 0, host, 4, 1, meanwhile);
        require(launches == 1 && meanwhile == nullptr, "mapped fallback lost or duplicated the GPU launch");
        while (e.pool_help()) {}
        for (int i : {0, 1}) {
            const uint8_t* p = e.pool_has(0, i);
            const auto original = source(i);
            require(p && std::memcmp(p, original.gate, l.gu_bytes) == 0 &&
                    std::memcmp(p + l.gu_bytes, original.up, l.u_bytes) == 0 &&
                    std::memcmp(p + l.gu2(), original.down, l.d_bytes) == 0, "slot's mixed matrices differ");
        }
        // A real file read fails when the source cannot be resolved. The slot stays unpublished; retry from
        // the valid view succeeds before CPU computation is allowed to use it.
        const int slot = e.pool_of_[0];
        e.views_.clear();
        e.pool_queue(l, 0, slot, true);
        while (e.pool_help()) {}
        require(e.pool_state_[(size_t) slot] == 2 && !e.pool_has(0, 0), "failed read published partial data");
        e.views_.push_back({f.mapped, f.bytes.size(), f.path.string()});
        Engine::PoolRead retry{0, slot, false};
        e.pool_read(l, retry);
        require(retry.ok, "file retry failed");
        e.pool_state_[(size_t) slot].store(0, std::memory_order_release);
        require(e.pool_has(0, 0) && std::memcmp(e.pool_has(0, 0), source(0).gate, l.gu_bytes) == 0,
                "retry did not restore expert bytes");
        std::puts("expert pipeline: deduplicated pending reads, slot exhaustion, mixed matrices and failed-read retry passed");
    }
};
}
#endif

int main() {
    try {
        File f;
#ifdef STRATA_TEST_EXPERT_ENGINE
        // File handles owned by this thread close before the fixture is removed.
        std::thread check([&] { ExpertIoProbe::run(f); });
        check.join();
#else
        ranges(f);
        priorities();
        reads(f);
        shutdown();
#endif
#ifndef STRATA_TEST_EXPERT_ENGINE
        std::puts("expert_io_test: mapped hints, mixed ranges, policy, read priority, file bytes, failure fallback and shutdown passed");
#endif
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL: %s\n", e.what());
        return 1;
    }
}
