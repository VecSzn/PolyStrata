#include "engine.hpp"
#include "runtime.hpp"
#include "strata/gpu/metal.hpp"
#include <algorithm>
#include <cstdio>
#include <filesystem>
#include <fstream>
#include <stdexcept>
#include <vector>
#include <unistd.h>

namespace g = strata::gpu;
using strata::poly::Layer;
static void ck(g::Error e) { if (e != g::kSuccess) throw std::runtime_error(g::error_string(e)); }
static void require(bool ok, const char* what) { if (!ok) throw std::runtime_error(what); }

// Exercise the real engine allocation, sequence and graph machinery without a model or memory-pressure
// dependency. Only the host-memory reading and architecture-specific operations are replaced.
class MemoryEngine final : public strata::poly::Engine {
public:
    std::optional<uint64_t> host = uint64_t(64) << 30;
    bool fail_rebind = false;
    bool mock_cache = false;
    unsigned resizes = 0;
    size_t resized_to = 0;
    void *key = nullptr, *value = nullptr, *ring = nullptr;

    explicit MemoryEngine(bool balanced = true, bool whole = false) {
        opt_.metal_balanced = balanced;
        opt_.kv_grow = !whole;
        opt_.max_context = 12288;
        opt_.gpu_experts = false;
        g::Stream stream; ck(g::stream_create(&stream)); st_ = stream;
        state_n_ = 4;
        state_ = static_cast<float*>(dmalloc(16, "test state"));
        layers_.resize(1);
        alloc_caches(); cover(1); seqs_.resize(1);
    }
    void grow(int64_t rows) { cover(rows); }
    size_t capacity() const { return bufs_[0].capacity; }
    size_t allocations() const { return dev_allocs_.size(); }
    int64_t covered() const { return kv_rows_; }
    size_t headroom() { return metal_host_headroom(); }
    void policy_fixture() {
        mock_cache = true; opt_.gpu_experts = true; n_moe_ = 1; n_expert_ = 2;
        cached_ = 0; metal_cache_bytes_ = size_t(1) << 20;
    }
    bool mind() { return metal_mind_memory(); }
    void freeze_policy() { freeze_experts_ = true; }
    void fill(unsigned char byte) {
        for (const auto& b : bufs_) ck(g::memset_async(b.p, byte, b.capacity, static_cast<g::Stream>(st_)));
        ck(g::stream_synchronize(static_cast<g::Stream>(st_)));
    }
    void check_prefix(size_t key_bytes, unsigned char byte) {
        std::vector<unsigned char> result(key_bytes);
        ck(g::memcpy_sync(result.data(), key, result.size(), g::kDeviceToHost));
        require(std::all_of(result.begin(), result.end(), [=](auto v) { return v == byte; }), "old KV contents changed");
        result.resize(17);
        ck(g::memcpy_sync(result.data(), ring, result.size(), g::kDeviceToHost));
        require(std::all_of(result.begin(), result.end(), [=](auto v) { return v == byte; }), "fixed ring changed");
    }
    void add_sequence() {
        seqs_.resize(2); park_bound(); bound_ = 1; pos_ = kv_rows_ = kv_reach_ = 0;
        state_ = static_cast<float*>(dmalloc(16, "test sequence state"));
        alloc_caches(); cover(1); park_bound(); take_bound(0);
    }
    void select(int seq) { hold(seq); }
    void set_position(int64_t pos) { pos_ = pos; }
    void check_other_capacity() {
        bool rejected = false;
        try { (void) seq_buf(1, static_cast<char*>(key) + 5000 * 8); }
        catch (const std::logic_error&) { rejected = true; }
        require(rejected, "seq_buf accepted an offset past destination allocation");
    }
    void cache_graph(void* resource = nullptr) {
        auto s = static_cast<g::Stream>(st_);
        g::Graph graph; g::GraphExec exec;
        ck(g::begin_capture(s)); ck(g::memset_async(resource ? resource : key, 9, 8, s)); ck(g::end_capture(s, &graph));
        ck(g::graph_instantiate(&exec, graph)); ck(g::graph_destroy(graph)); graphs_.push_back(exec);
        auto& l = layers_[0];
        l.heads = l.shared = l.tail = l.card_moe = {exec}; l.rows_heads[{1}] = exec;
        passes_[{1}] = exec; tail_ = exec;
        for (auto& q : seqs_) { q.heads = {{exec}}; q.passes[{1}] = exec; q.tail = exec; }
    }
    void check_graphs_empty() {
        require(graphs_.empty() && passes_.empty() && !tail_, "engine graph survived relocation");
        const auto& l = layers_[0];
        require(l.heads.empty() && l.shared.empty() && l.tail.empty() && l.card_moe.empty() && l.rows_heads.empty(),
                "layer graph survived relocation");
        for (const auto& q : seqs_) require(q.heads.empty() && q.passes.empty() && !q.tail, "parked graph survived relocation");
    }
    void freeze_with_cache() {
        exp_ = static_cast<uint8_t*>(dmalloc(256, "test frozen expert allocation"));
        metal_cache_bytes_ = 256; freeze_experts_ = true;
    }
    void legacy_cache_fixture() {
        // A valid empty GGUF supplies only the file-size query in fill_expert_cache(); no weights are loaded.
        const auto path = std::filesystem::temp_directory_path() /
                          ("poly-metal-memory-growth-" + std::to_string(getpid()) + ".gguf");
        const uint64_t header[] = {(uint64_t(3) << 32) | 0x46554747, 0, 0, 0};
        { std::ofstream out(path, std::ios::binary); out.write(reinterpret_cast<const char*>(header), sizeof(header)); }
        try { gg_ = std::make_unique<strata::poly::Shards>(path.string()); }
        catch (...) { std::filesystem::remove(path); throw; }
        std::filesystem::remove(path);
        opt_.gpu_experts = true; n_moe_ = n_expert_ = cached_ = 1;
        exp_ = static_cast<uint8_t*>(dmalloc(1024, "test legacy expert allocation"));
        metal_cache_bytes_ = 1024; layers_[0].slot = {exp_};
        cache_graph(exp_);
    }
protected:
    std::optional<uint64_t> metal_host_available() const override { return host; }
    void metal_resize_cache(size_t limit) override {
        if (!mock_cache) { Engine::metal_resize_cache(limit); return; }
        ++resizes; resized_to = limit; metal_cache_bytes_ = limit;
    }
    void alloc_caches() override {
        key = kv_alloc(8, 1);
        if (fail_rebind && again_ != SIZE_MAX) { fail_rebind = false; throw std::runtime_error("injected rebind failure"); }
        value = kv_alloc(4, 2);
        ring = seq_alloc(17, "test fixed ring");
    }
    void load_geometry() override {}
    void load_layer(int) override {}
    size_t recurrent_floats() const override { return 4; }
    size_t card_row_bytes() const override { return 10; }
    void save_card_rows(int64_t, int64_t, uint8_t*) override {}
    void load_card_rows(const uint8_t*, int64_t, int64_t, int64_t, int64_t) override {}
    void assign_scratch() override {}
    int widest() const override { return 1; }
    int streams() const override { return 1; }
    void layer_head(Layer&, int, int) override {}
    void layer_tail(Layer&, int, int) override {}
    void output_rows(int, int, float*) override {}
    void route(int, const float*, int32_t, int*, float*, float*) const override {}
    void moe_shared(const Layer&, int, int) override {}
    void save_layer(int, int64_t, uint8_t*) override {}
};

int main() {
    try {
        g::metal::set_scoped_residency(true);
        {
            MemoryEngine e;
            require(e.capacity() == 4096 * 8 && e.covered() == 4096, "balanced KV is not initially 4096 rows");
            e.fill(23); e.add_sequence(); e.cache_graph();
            auto old_key = e.key, old_ring = e.ring;
            e.grow(4097);
            require(e.capacity() == 8192 * 8 && e.covered() == 8192 && e.key != old_key, "KV did not relocate to 8192 rows");
            require(e.ring == old_ring, "fixed allocation relocated");
            e.check_prefix(4096 * 8, 23); e.check_graphs_empty(); e.check_other_capacity();
            e.set_position(6000); e.copy_seq(0, 1); e.select(1);
            require(e.capacity() == 8192 * 8, "copied sequence did not grow");
            e.check_prefix(4096 * 8, 23);
            e.select(0); require(e.key != old_key, "sequence rebind restored retired pointer");
        }
        {
            MemoryEngine e; e.fill(31);
            const auto old = e.key; const size_t count = e.allocations();
            e.fail_rebind = true;
            bool failed = false;
            try { e.grow(4097); } catch (const std::runtime_error&) { failed = true; }
            require(failed && e.key == old && e.capacity() == 4096 * 8 && e.covered() == 4096,
                    "failed relocation changed the live KV allocation");
            require(e.allocations() == count, "failed relocation leaked allocations");
            e.check_prefix(4096 * 8, 31);
            e.grow(4097); e.check_prefix(4096 * 8, 31);
        }
        {
            MemoryEngine e; e.fill(29);
            const auto old = e.key; const size_t count = e.allocations();
            const auto allocated = g::metal::testing::allocated_bytes();
            g::metal::testing::fail_allocation_after(1); // first new KV succeeds, the second fails
            bool failed = false;
            try { e.grow(4097); } catch (const std::runtime_error&) { failed = true; }
            g::metal::testing::fail_allocation_after(-1);
            require(failed && e.key == old && e.covered() == 4096, "allocation failure changed live KV");
            require(e.allocations() == count && g::metal::testing::allocated_bytes() == allocated,
                    "allocation failure leaked a replacement buffer");
            e.check_prefix(4096 * 8, 29);
        }
        {
            MemoryEngine e; e.fill(17); e.freeze_with_cache(); e.host = uint64_t(4) << 30;
            const auto old = e.key; bool failed = false;
            try { e.grow(4097); } catch (const std::runtime_error&) { failed = true; }
            require(failed && e.key == old && e.covered() == 4096, "frozen cache OOM changed old pointers");
            e.check_prefix(4096 * 8, 17);
            require(e.headroom() == 0, "host reserve was not applied");
            e.host = (uint64_t(4) << 30) + 123;
            require(e.headroom() == 123, "host reserve was counted twice");
            e.host.reset(); require(e.headroom() == SIZE_MAX, "unknown memory did not use GPU fallback");
        }
        {
            MemoryEngine e; e.policy_fixture();
            e.host = (uint64_t(4) << 30) - (uint64_t(512) << 10);
            require(e.mind() && e.resizes == 1 && e.resized_to == (size_t(512) << 10),
                    "pressure did not immediately shrink by the host reserve deficit");
            e.host = (uint64_t(4) << 30) + (uint64_t(640) << 20);
            require(!e.mind() && !e.mind() && e.resizes == 1,
                    "growth margin ignored the refill cushion");
            e.host = (uint64_t(4) << 30) + (uint64_t(768) << 20) + 1;
            require(!e.mind() && e.resizes == 1, "cache grew on its first high-headroom observation");
            e.host = (uint64_t(4) << 30) + (uint64_t(64) << 10);
            require(!e.mind(), "small headroom changed the cache");
            e.host = (uint64_t(4) << 30) + (uint64_t(768) << 20) + 1;
            require(!e.mind() && e.resizes == 1, "low headroom did not reset growth hysteresis");
            require(e.mind() && e.resizes == 2, "two high-headroom boundaries did not grow the cache");
            e.freeze_policy(); e.host = 0;
            require(!e.mind() && e.resizes == 2, "memory pressure resized frozen experts");
        }
        {
            MemoryEngine legacy(false), whole(true, true);
            require(legacy.capacity() == 12288 * 8 && whole.capacity() == 12288 * 8, "legacy/whole KV changed");
            const auto before = g::metal::testing::allocated_bytes();
            legacy.legacy_cache_fixture(); legacy.resize_cache(-1); legacy.check_graphs_empty();
            require(g::metal::testing::allocated_bytes() == before, "legacy cache resize retained the retired arena");
        }
        std::puts("Metal KV growth, rollback, graphs, sequences and memory reserve: OK");
    } catch (const std::exception& e) { std::fprintf(stderr, "%s\n", e.what()); return 1; }
}
