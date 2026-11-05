// metal/src/gpu/runtime.cpp - include/strata/gpu/runtime.hpp on Metal (metal-cpp).  Skeleton of MAC-PLAN Phase 1.
//
// Memory model (measured on the M5 Max, phaseA-tests/mac-design.md):
//   - every allocation is one MTLBuffer, storageModeShared, hazard tracking untracked (the engine orders its own work
//     on streams, as on CUDA); the "device pointer" handed to the host code is the buffer's gpuAddress, which differs
//     from contents() on Apple GPUs, so host code that wrongly dereferences a device pointer faults instead of
//     silently working.  host_view() is the sanctioned CPU view.
//   - all buffers sit in one MTLResidencySet attached to every queue, so kernels may take raw gpuAddress pointers.
// Streams: a command queue + the open command buffer + the open compute encoder + a shared-event timeline.  Work is
// committed at event records, synchronizations, host functions and every kFlushEvery dispatches.
// Graphs: begin_capture switches the stream to recording; dispatch/copy/memset/event ops are kept as data and
// re-encoded by graph_launch (encoding measured at ~0.17 us per dispatch on the M5 Max).
#define NS_PRIVATE_IMPLEMENTATION
#define MTL_PRIVATE_IMPLEMENTATION
#include <execinfo.h>
#include <Foundation/Foundation.hpp>
#include <Metal/Metal.hpp>

#include "strata/gpu/metal.hpp"
#include "strata/gpu/runtime.hpp"

#include <sys/sysctl.h>

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <memory>
#include <set>
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <iterator>
#include <map>
#include <mutex>
#include <thread>
#include <vector>

namespace strata::gpu {
struct Completion {
    MTL::SharedEvent* timeline = nullptr;
    std::mutex mu;
    std::condition_variable cv;
    uint64_t completed = 0;
    bool failed = false;
    ~Completion() { if (timeline) timeline->release(); }
};
struct EventState {
    std::mutex mu;
    std::shared_ptr<Completion> completion;
    uint64_t value = 0;
};


namespace {

constexpr int kFlushEvery = 4096;   // dispatches per command buffer before an automatic commit

struct Alloc {
    MTL::Buffer* buf = nullptr;
    uint64_t gpu = 0;
    uint8_t* cpu = nullptr;
    size_t bytes = 0;
    bool host = false;   // from host_alloc (the host code holds the CPU address)
    size_t holds = 0;
    bool retired = false;
    bool purged = false; // purge_device_allocations: its memory given back (any later copy is an error)
};

struct HeldBuf {
    uint64_t key = 0;
    MTL::Buffer* buf = nullptr;
    size_t bytes = 0;
};
void release_resources(const std::vector<HeldBuf>& items);
struct ResourceHold {
    std::vector<HeldBuf> items;
    ~ResourceHold() { release_resources(items); }
};
struct Op {
    enum Kind { Dispatch, Copy, Fill, Record, Wait, Submit } kind = Dispatch;
    MTL::ComputePipelineState* pipe = nullptr;
    uint8_t args[256];
    size_t arg_bytes = 0;
    std::vector<uint8_t> big_args;
    metal::Grid groups, threads;
    size_t tg_bytes = 0;
    void* dst = nullptr;
    const void* src = nullptr;
    size_t bytes = 0;
    int value = 0;
    std::shared_ptr<EventState> ev;
    std::shared_ptr<ResourceHold> resources;
    bool complete_resources = false;
    const void* indirect_table = nullptr;
};

}  // namespace

struct EventImpl { std::shared_ptr<EventState> state = std::make_shared<EventState>(); };

struct GraphImpl {
    std::vector<Op> ops;
    std::vector<std::shared_ptr<ResourceHold>> holds;
};
struct GraphExecImpl {
    std::vector<Op> ops;
    std::vector<std::shared_ptr<ResourceHold>> holds;
};

struct StreamImpl {
    std::recursive_mutex mu;
    MTL::CommandQueue* q = nullptr;
    MTL::CommandBuffer* cb = nullptr;
    MTL::ComputeCommandEncoder* enc = nullptr;
    MTL::Fence* fence = nullptr;
    bool fence_written = false;
    MTL::SharedEvent* tl = nullptr;
    std::shared_ptr<Completion> completion = std::make_shared<Completion>();
    uint64_t submitted = 0;   // the timeline value the last committed command buffer signals
    int pending = 0;          // dispatches in the open command buffer
    GraphImpl* rec = nullptr;
    std::vector<std::shared_ptr<ResourceHold>> resources;
    std::vector<uint64_t> enc_keys;
    bool fallback_residency = false;
};

namespace {

struct Runtime {
    MTL::Device* dev = nullptr;
    MTL::ResidencySet* rs = nullptr;
    std::mutex mu;
    std::recursive_mutex streams_mu;
    std::set<StreamImpl*> streams;
    std::atomic<bool> failed{false};
    std::atomic<bool> optimize{false};
    std::atomic<bool> scoped{false};
#if defined(STRATA_METAL_TESTING)
    std::atomic<bool> fail_next{false};
    std::atomic<int> fail_allocation{-1};
    std::atomic<size_t> residency_bytes{0};
#endif
    std::map<uint64_t, std::vector<const void*>> indirect;
    std::map<uint64_t, Alloc> by_gpu;   // gpuAddress -> allocation
    std::map<uintptr_t, uint64_t> by_cpu;   // contents() -> gpuAddress
    StreamImpl* legacy = nullptr;
    std::map<std::string, MTL::ComputePipelineState*> pipes;
    std::map<std::string, MTL::Library*> libs;
    struct Counters {
        std::atomic<uint64_t> dispatches{0}, command_buffers{0}, blits{0}, host_syncs{0};
    } stats;
};

Runtime& rt() {
    static Runtime* r = [] {
        auto* x = new Runtime;
        x->dev = MTL::CreateSystemDefaultDevice();
        NS::Error* e = nullptr;
        MTL::ResidencySetDescriptor* d = MTL::ResidencySetDescriptor::alloc()->init();
        x->rs = x->dev->newResidencySet(d, &e);
        d->release();
        return x;
    }();
    return *r;
}

thread_local Error t_last = kSuccess;
Error fail(Error e) {
    if (e != kSuccess && e != kNotReady) t_last = e;
    return e;
}

StreamImpl* make_stream() {
    auto* s = new StreamImpl;
    s->q = rt().dev->newCommandQueue();
    s->fence = rt().dev->newFence();
    // Residency is attached per command buffer, with a conservative fallback.
    s->tl = rt().dev->newSharedEvent();
    s->tl->setSignaledValue(0);
    s->completion->timeline = s->tl;
    std::lock_guard<std::recursive_mutex> lock(rt().streams_mu);
    rt().streams.insert(s);
    return s;
}

StreamImpl* S(Stream s) {
    if (s) return s;
    Runtime& r = rt();
    std::lock_guard<std::recursive_mutex> lk(r.streams_mu);
    if (!r.legacy) r.legacy = make_stream();
    return r.legacy;
}

/// The allocation that holds GPU address `p` (or CPU address `p` of a host allocation), with the offset into it.
const Alloc* find_unlocked(Runtime& r, const void* p, size_t* off) {
    const uint64_t a = (uint64_t) (uintptr_t) p;
    auto it = r.by_gpu.upper_bound(a);
    if (it != r.by_gpu.begin()) {
        --it;
        if (a < it->first + it->second.bytes) { *off = a - it->first; return &it->second; }
    }
    auto ic = r.by_cpu.upper_bound((uintptr_t) p);
    if (ic != r.by_cpu.begin()) {
        --ic;
        const Alloc& al = r.by_gpu[ic->second];
        if ((uintptr_t) p < ic->first + al.bytes) { *off = (uintptr_t) p - ic->first; return &al; }
    }
    return nullptr;
}

const Alloc* find(const void* p, size_t* off) {
    Runtime& r = rt();
    std::lock_guard<std::mutex> lk(r.mu);
    return find_unlocked(r, p, off);
}

void dispose_alloc(std::map<uint64_t, Alloc>::iterator it) {
    auto& r = rt();
    auto a = it->second;
    r.by_cpu.erase((uintptr_t)a.cpu);
    r.indirect.erase(a.gpu);
    r.rs->removeAllocation(a.buf);
    r.rs->commit();
    r.by_gpu.erase(it);
    if (!a.purged) a.buf->setPurgeableState(MTL::PurgeableStateEmpty);
    a.buf->release();
}
void release_resources(const std::vector<HeldBuf>& items) {
    if (items.empty()) return;
    auto& r = rt();
    std::lock_guard<std::mutex> lock(r.mu);
    for (const auto& item : items) {
        auto it = r.by_gpu.find(item.key);
        if (it != r.by_gpu.end() && --it->second.holds == 0 && it->second.retired)
            dispose_alloc(it);
    }
}
std::shared_ptr<ResourceHold> hold_resources(const std::vector<const void*>& pointers, bool all = false) {
    auto h = std::make_shared<ResourceHold>();
    auto& r = rt();
    std::lock_guard<std::mutex> lock(r.mu);
    if (all) {
        h->items.reserve(r.by_gpu.size());
        for (auto& [key, a] : r.by_gpu) {
            if (!a.retired && !a.purged) {
                ++a.holds;
                h->items.push_back({key, a.buf, a.bytes});
            }
        }
        return h;
    }
    std::vector<uint64_t> keys;
    keys.reserve(pointers.size());
    for (auto p : pointers) if (p) {
        size_t offset = 0;
        const auto* a = find_unlocked(r, p, &offset);
        if (!a || a->purged) return {};
        if (std::find(keys.begin(), keys.end(), a->gpu) == keys.end())
            keys.push_back(a->gpu);
    }
    h->items.reserve(keys.size());
    for (auto key : keys) {
        auto& a = r.by_gpu.at(key);
        ++a.holds;
        h->items.push_back({key, a.buf, a.bytes});
    }
    return h;
}

MTL::CommandBuffer* open_cb(StreamImpl* s) {
    if (!s->cb) {
        s->cb = s->q->commandBuffer();
        s->cb->retain();
        ++rt().stats.command_buffers;
    }
    return s->cb;
}
// These buffers are untracked. A serial compute encoder orders its own dispatches,
// but copies/fills in a blit encoder need a fence before and after compute work.
void end_encoder(StreamImpl* s) {
    if (s->enc) {
        s->enc->updateFence(s->fence);
        s->fence_written = true;
        s->enc->endEncoding();
        s->enc = nullptr;
        s->enc_keys.clear();
    }
}
MTL::ComputeCommandEncoder* encoder(StreamImpl* s) {
    if (!s->enc) {
        s->enc = open_cb(s)->computeCommandEncoder();
        if (s->fence_written) s->enc->waitForFence(s->fence);
    }   // serial dispatch within an encoder; explicit ordering across blits
    return s->enc;
}
void use_hold(StreamImpl* s, MTL::ComputeCommandEncoder* e, const std::shared_ptr<ResourceHold>& h) {
    if (!h) return;
    if (s->resources.empty() || s->resources.back() != h) s->resources.push_back(h);
    for (const auto& item : h->items) {
        if (std::find(s->enc_keys.begin(), s->enc_keys.end(), item.key) == s->enc_keys.end()) {
            s->enc_keys.push_back(item.key);
            e->useResource(item.buf, MTL::ResourceUsageRead | MTL::ResourceUsageWrite);
        }
    }
}
void flush(StreamImpl* s) {
    if (!s->cb) return;
    end_encoder(s);
    auto holds = std::move(s->resources);
    if (!rt().scoped.load() || s->fallback_residency) {
        holds.push_back(hold_resources({}, true));
        s->cb->useResidencySet(rt().rs);
    }
#if defined(STRATA_METAL_TESTING)
    {
        std::vector<uint64_t> seen;
        size_t bytes = 0;
        for (const auto& h : holds) if (h) {
            for (const auto& item : h->items) {
                if (std::find(seen.begin(), seen.end(), item.key) == seen.end()) {
                    seen.push_back(item.key);
                    bytes += item.bytes;
                }
            }
        }
        rt().residency_bytes.store(bytes);
    }
#endif
    s->fallback_residency = false;
    const auto value = ++s->submitted;
    s->cb->encodeSignalEvent(s->tl, value);
    const auto completion = s->completion;
    bool injected_failure = false;
#if defined(STRATA_METAL_TESTING)
    injected_failure = rt().fail_next.exchange(false);
#endif
    s->cb->addCompletedHandler([completion, value, holds, injected_failure](MTL::CommandBuffer* cb) mutable {
        holds.clear(); // release submission ownership before publishing completion
        {
            std::lock_guard<std::mutex> lock(completion->mu);
            if (injected_failure || cb->status() == MTL::CommandBufferStatusError) {
                completion->failed = true;
                rt().failed.store(true);
                std::fprintf(stderr, "strata metal: command buffer failed: %s\n",
                    injected_failure ? "injected test failure" : cb->error() ? cb->error()->localizedDescription()->utf8String() : "unknown error");
            }
            completion->completed = std::max(completion->completed, value);
        }
        completion->cv.notify_all();
    });
    s->cb->commit();
    s->cb->release();
    s->cb = nullptr;
    s->pending = 0;
}
Error wait_value(const std::shared_ptr<Completion>& state, uint64_t v) {
    ++rt().stats.host_syncs;
    if (!state || !v) return rt().failed.load() ? fail(kErrorLaunchFailure) : kSuccess;
    std::unique_lock<std::mutex> lock(state->mu);
    // Notification handles normal completion. Poll only to notice a failure on a
    // different queue whose event this queue may be waiting for.
    while (state->completed < v && !rt().failed.load())
        state->cv.wait_for(lock, std::chrono::milliseconds(10));
    return state->failed || rt().failed.load() ? fail(kErrorLaunchFailure) : kSuccess;
}

void check_purged(const Alloc* a, const char* what) {
    if (a && a->purged) {
        std::fprintf(stderr, "strata metal runtime: a %s touches a device allocation of %.1f MB that was given back "
                             "(STRATA_METAL_KEEP_HOST_WEIGHTS=1 keeps it)\n", what, a->bytes / 1e6);
        void* fr[48];
        backtrace_symbols_fd(fr, backtrace(fr, 48), 2);   // which host code still reads it
        std::abort();
    }
}

Error encode_copy(StreamImpl* s, void* dst, const void* src, size_t n) {
    size_t od = 0, os = 0;
    const Alloc* ad = find(dst, &od);
    const Alloc* as = find(src, &os);
    check_purged(ad, "copy");
    check_purged(as, "copy");
    if (ad && as) {   // buffer to buffer: a blit, in stream order
        s->resources.push_back(hold_resources({dst,src}));
        end_encoder(s);
        MTL::BlitCommandEncoder* b = open_cb(s)->blitCommandEncoder();
        if (s->fence_written) b->waitForFence(s->fence);
        b->copyFromBuffer(as->buf, os, ad->buf, od, n);
        b->updateFence(s->fence);
        s->fence_written = true;
        b->endEncoding();
        ++rt().stats.blits;
        return kSuccess;
    }
    // ordinary host memory on one side: the stream's earlier work must be complete first (the conservative CUDA
    // pageable-memory semantics: the call returns when the copy is done)
    flush(s);
    if (auto e = wait_value(s->completion, s->submitted); e != kSuccess) return e;
    uint8_t* d = ad ? ad->cpu + od : (uint8_t*) dst;
    const uint8_t* sp = as ? as->cpu + os : (const uint8_t*) src;
    std::memcpy(d, sp, n);
    return kSuccess;
}

Error encode_fill(StreamImpl* s, void* p, int value, size_t n) {
    size_t off = 0;
    const Alloc* a = find(p, &off);
    if (!a) return fail(kErrorInvalidValue);
    check_purged(a, "fill");
    s->resources.push_back(hold_resources({p}));
    end_encoder(s);
    MTL::BlitCommandEncoder* b = open_cb(s)->blitCommandEncoder();
    if (s->fence_written) b->waitForFence(s->fence);
    b->fillBuffer(a->buf, NS::Range::Make(off, n), (uint8_t) value);
    b->updateFence(s->fence);
    s->fence_written = true;
    b->endEncoding();
    ++rt().stats.blits;
    return kSuccess;
}

Error encode_dispatch(StreamImpl* s, const Op& o) {
    std::shared_ptr<ResourceHold> indirect_hold;
    if (o.indirect_table) {
        std::vector<const void*> targets;
        bool known = false;
        {
            Runtime& r = rt();
            std::lock_guard<std::mutex> lock(r.mu);
            size_t offset = 0;
            const auto* table = find_unlocked(r, o.indirect_table, &offset);
            auto it = table ? r.indirect.find(table->gpu) : r.indirect.end();
            if (it != r.indirect.end()) { targets = it->second; known = true; }
        }
        indirect_hold = known ? hold_resources(targets) : nullptr;
        if (!indirect_hold) s->fallback_residency = true;
    }
    if (!o.complete_resources) s->fallback_residency = true;
    MTL::ComputeCommandEncoder* e = encoder(s);
    use_hold(s, e, o.resources);
    use_hold(s, e, indirect_hold);
    e->setComputePipelineState(o.pipe);
    if (o.big_args.empty()) e->setBytes(o.args, o.arg_bytes, 0);
    else e->setBytes(o.big_args.data(), o.big_args.size(), 0);
    if (o.tg_bytes) e->setThreadgroupMemoryLength(o.tg_bytes, 0);
    e->dispatchThreadgroups(MTL::Size(o.groups.x, o.groups.y, o.groups.z), MTL::Size(o.threads.x, o.threads.y, o.threads.z));
    ++rt().stats.dispatches;
    if (++s->pending >= kFlushEvery) flush(s);
    return kSuccess;
}

Error record_event(const std::shared_ptr<EventState>& e, StreamImpl* s) {
    flush(s);   // CUDA submits on launch: a recorded event must become complete without a further call
    std::lock_guard<std::mutex> lock(e->mu);
    e->completion = s->completion;
    e->value = s->submitted;
    return kSuccess;
}
Error wait_event(StreamImpl* s, const std::shared_ptr<EventState>& e) {
    if (!e) return kSuccess;
    std::lock_guard<std::mutex> lock(e->mu);
    if (e->value == 0) return kSuccess;
    end_encoder(s);
    open_cb(s)->encodeWait(e->completion->timeline, e->value);
    return kSuccess;
}

Error run_op(StreamImpl* s, const Op& o) {
    switch (o.kind) {
        case Op::Dispatch: return encode_dispatch(s, o);
        case Op::Copy: return encode_copy(s, o.dst, o.src, o.bytes);
        case Op::Fill: return encode_fill(s, o.dst, o.value, o.bytes);
        case Op::Record: return record_event(o.ev, s);
        case Op::Wait: return wait_event(s, o.ev);
        case Op::Submit: flush(s); return kSuccess;
    }
    return kSuccess;
}

// STRATA_METAL_ALLOC_LOG=1: every allocation of 16 MB or more, with the running total (memory budgets on small Macs)
void alloc_log(const char* what, size_t n) {
    static const bool on = std::getenv("STRATA_METAL_ALLOC_LOG") != nullptr;
    static std::atomic<uint64_t> tot{0};
    const uint64_t t = tot += n;
    if (on && n >= (16u << 20)) std::fprintf(stderr, "strata metal runtime: %s %.1f MB (total %.2f GB)\n", what, n / 1e6, t / 1e9);
}

Error new_alloc(void** p, size_t bytes, bool host) {
    Runtime& r = rt();
#if defined(STRATA_METAL_TESTING)
    int left=r.fail_allocation.load();
    if(left>=0 && r.fail_allocation.fetch_sub(1)==0){*p=nullptr;return fail(kErrorMemoryAllocation);}
#endif
    const size_t n = bytes ? bytes : 1;
    MTL::Buffer* b = r.dev->newBuffer(n, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked);
    if (!b) return fail(kErrorMemoryAllocation);
    Alloc a{b, b->gpuAddress(), (uint8_t*) b->contents(), n, host, 0, false, false};
    alloc_log(host ? "host alloc" : "device alloc", n);
    {
        std::lock_guard<std::mutex> lk(r.mu);
        r.by_gpu[a.gpu] = a;
        r.by_cpu[(uintptr_t) a.cpu] = a.gpu;
        r.rs->addAllocation(b);
        r.rs->commit();
    }
    *p = host ? (void*) a.cpu : (void*) (uintptr_t) a.gpu;
    return kSuccess;
}

Error free_alloc(void* p) {
    if (!p) return kSuccess;
    Runtime& r = rt();
    size_t off = 0;
    const Alloc* a = find(p, &off);
    if (!a || off != 0) return fail(kErrorInvalidValue);
    if (auto e = device_synchronize(); e != kSuccess) return e;
    std::lock_guard<std::mutex> lk(r.mu);
    auto it = r.by_gpu.find(a->gpu);
    it->second.retired = true;
    if (!it->second.holds) dispose_alloc(it);
    return kSuccess;
}

}  // namespace

// ---------------------------------------------------------------------------------------------------- runtime.hpp
void* host_view(const void* p) {
    size_t off = 0;
    const Alloc* a = find(p, &off);
    return a ? a->cpu + off : nullptr;
}

// The Metal decode window reads the model itself; the device copies of the weights the shared host code uploads for
// the CUDA-shaped kernels (the pack's dense pool, the native head, ...) are never read on Metal.  On a small Mac their
// memory is worth more as expert arena: every device (not host) allocation of min_bytes or more is made purgeable-empty
// and dropped from the residency set.  Returns the bytes given back.
uint64_t purge_device_allocations(size_t min_bytes) {
    Runtime& r = rt();
    if (device_synchronize() != kSuccess) return 0;
    std::lock_guard<std::mutex> lk(r.mu);
    uint64_t n = 0;
    for (auto& [gpu, a] : r.by_gpu) {
        if (a.host || a.purged || a.holds || a.bytes < min_bytes) continue;
        r.rs->removeAllocation(a.buf);
        a.buf->setPurgeableState(MTL::PurgeableStateEmpty);
        a.purged = true;
        n += a.bytes;
    }
    r.rs->commit();
    return n;
}

Error get_last_error() { const Error e = t_last; t_last = kSuccess; return e; }
Error peek_last_error() { return t_last; }
const char* error_string(Error e) {
    switch (e) {
        case kSuccess: return "no error";
        case kNotReady: return "device not ready";
        case kErrorInvalidValue: return "invalid argument";
        case kErrorMemoryAllocation: return "out of memory";
        case kErrorUnsupported: return "operation not supported on Metal";
        case kErrorLaunchFailure: return "command buffer failed";
    }
    return "unknown Metal runtime error";
}

Error get_device(int* dev) { *dev = 0; return kSuccess; }
Error set_device(int dev) { return dev == 0 ? kSuccess : fail(kErrorInvalidValue); }
Error device_count(int* n) { *n = rt().dev ? 1 : 0; return kSuccess; }
Error device_synchronize() {
    Runtime& r = rt();
    // Stream destruction uses this separate registry lock as well. Completion
    // callbacks never acquire the registry lock. Resource cleanup may acquire
    // the allocation mutex, which is never held here while waiting.
    std::lock_guard<std::recursive_mutex> lock(r.streams_mu);
    std::vector<std::pair<std::shared_ptr<Completion>, uint64_t>> submitted;
    for (auto* s : r.streams) {
        std::lock_guard<std::recursive_mutex> stream_lock(s->mu);
        flush(s);
        submitted.emplace_back(s->completion, s->submitted);
    }
    for (const auto& [completion,value] : submitted)
        if (auto e = wait_value(completion, value); e != kSuccess) return e;
    return kSuccess;
}

Error mem_get_info(size_t* free_b, size_t* total_b) {
    // the GPU's budget is the working set the OS grants this process; what is free is that minus what is allocated
    MTL::Device* d = rt().dev;
    const uint64_t budget = d->recommendedMaxWorkingSetSize();
    const uint64_t used = d->currentAllocatedSize();
    *total_b = (size_t) budget;
    *free_b = (size_t) (used < budget ? budget - used : 0);
    return kSuccess;
}
Error device_integrated(int* v, int) { *v = 1; return kSuccess; }
Error device_sm_count(int* v, int) { *v = 0; return fail(kErrorUnsupported); }   // Apple reports no core count
Error device_clock_khz(int* v, int) { *v = 0; return fail(kErrorUnsupported); }
Error runtime_version(int* v) { *v = 0; return kSuccess; }
Error device_can_access_peer(int* v, int, int) { *v = 0; return kSuccess; }
Error device_enable_peer_access(int, unsigned) { return fail(kErrorUnsupported); }
Error get_device_properties(DeviceProp* p, int dev) {
    if (dev != 0) return fail(kErrorInvalidValue);
    *p = DeviceProp{};
    MTL::Device* d = rt().dev;
    std::snprintf(p->name, sizeof p->name, "%s", d->name()->utf8String());
    std::snprintf(p->gcnArchName, sizeof p->gcnArchName, "%s", d->supportsFamily(MTL::GPUFamilyApple10) ? "apple10" : "apple9-or-older");
    p->warpSize = 32;   // the SIMD-group width of every Apple GPU
    p->totalGlobalMem = (size_t) d->recommendedMaxWorkingSetSize();
    return kSuccess;
}

Error malloc_device_raw(void** p, size_t bytes) { return new_alloc(p, bytes, false); }
Error free_device(void* p) { return free_alloc(p); }
Error host_alloc_raw(void** p, size_t bytes, unsigned) { return new_alloc(p, bytes, true); }
Error free_host(void* p) {
    {
        Runtime& r = rt();
        std::lock_guard<std::mutex> lk(r.mu);
        auto it = r.by_cpu.find((uintptr_t) p);
        if (it == r.by_cpu.end()) return fail(kErrorInvalidValue);
    }
    return free_alloc(p);
}
Error host_register(void* p, size_t bytes, unsigned) {
    // existing host memory (an mmapped file, an arena) as a NoCopy buffer: page-aligned start and length required
    const uintptr_t pg = 16384;
    if (((uintptr_t) p % pg) != 0) return fail(kErrorInvalidValue);
    Runtime& r = rt();
    const size_t n = (bytes + pg - 1) / pg * pg;
    MTL::Buffer* b = r.dev->newBuffer(p, n, MTL::ResourceStorageModeShared | MTL::ResourceHazardTrackingModeUntracked, nullptr);
    if (!b) return fail(kErrorMemoryAllocation);
    Alloc a{b, b->gpuAddress(), (uint8_t*) b->contents(), n, true, 0, false, false};
    alloc_log("host register", n);
    std::lock_guard<std::mutex> lk(r.mu);
    r.by_gpu[a.gpu] = a;
    r.by_cpu[(uintptr_t) a.cpu] = a.gpu;
    r.rs->addAllocation(b);
    r.rs->commit();
    return kSuccess;
}
Error host_unregister(void* p) { return free_alloc(p); }
Error host_device_pointer_raw(void** dev, void* host) {
    size_t off = 0;
    const Alloc* a = find(host, &off);
    if (!a) return fail(kErrorInvalidValue);
    *dev = (void*) (uintptr_t) (a->gpu + off);
    return kSuccess;
}
Error memset_device(void* p, int value, size_t bytes) {
    StreamImpl* s = S(nullptr);
    std::lock_guard<std::recursive_mutex> stream_lock(s->mu);
    const Error e = encode_fill(s, p, value, bytes);
    flush(s);
    const auto done = wait_value(s->completion, s->submitted);
    return e == kSuccess ? done : e;
}
Error memset_async(void* p, int value, size_t bytes, Stream st) {
    StreamImpl* s = S(st);
    std::lock_guard<std::recursive_mutex> stream_lock(s->mu);
    if (s->rec) {
        Op o;
        o.kind = Op::Fill; o.dst = p; o.value = value; o.bytes = bytes;
        o.resources = hold_resources({p});
        s->rec->ops.push_back(std::move(o));
        return kSuccess;
    }
    return encode_fill(s, p, value, bytes);
}

Error memcpy_sync(void* dst, const void* src, size_t bytes, CopyKind) {
    StreamImpl* s = S(nullptr);
    std::lock_guard<std::recursive_mutex> stream_lock(s->mu);
    const Error e = encode_copy(s, dst, src, bytes);
    flush(s);
    const auto done = wait_value(s->completion, s->submitted);
    return e == kSuccess ? done : e;
}
Error memcpy_2d_async(void* dst, size_t dpitch, const void* src, size_t spitch, size_t width, size_t height, CopyKind k, Stream st) {
    for (size_t r = 0; r < height; ++r) {   // skeleton: one copy per row (rare in the engine)
        const Error e = memcpy_async((uint8_t*) dst + r * dpitch, (const uint8_t*) src + r * spitch, width, k, st);
        if (e != kSuccess) return e;
    }
    return kSuccess;
}
Error memcpy_peer_async(void* dst, int dst_dev, const void* src, int src_dev, size_t bytes, Stream st) {
    if (dst_dev != 0 || src_dev != 0) return fail(kErrorInvalidValue);   // one GPU
    return memcpy_async(dst, src, bytes, kDeviceToDevice, st);
}
Error memcpy_async(void* dst, const void* src, size_t bytes, CopyKind, Stream st) {
    StreamImpl* s = S(st);
    std::lock_guard<std::recursive_mutex> stream_lock(s->mu);
    if (s->rec) {
        Op o;
        o.kind = Op::Copy; o.dst = dst; o.src = src; o.bytes = bytes;
        std::vector<const void*> refs;
        size_t off=0;
        if(find(dst,&off)) refs.push_back(dst);
        if(find(src,&off)) refs.push_back(src);
        o.resources = hold_resources(refs);
        s->rec->ops.push_back(std::move(o));
        return kSuccess;
    }
    return encode_copy(s, dst, src, bytes);
}

Error stream_create(Stream* s, unsigned) { *s = make_stream(); return kSuccess; }
Error stream_destroy(Stream s) {
    if (!s) return kSuccess;
    std::lock_guard<std::recursive_mutex> lock(rt().streams_mu);
    std::unique_lock<std::recursive_mutex> stream_lock(s->mu);
    flush(s);
    const auto result = wait_value(s->completion, s->submitted);
    rt().streams.erase(s);
    delete s->rec;
    s->rec = nullptr;
    s->fence->release();
    s->q->release();
    stream_lock.unlock();
    delete s;
    return result;
}
Error stream_synchronize(Stream st) {
    StreamImpl* s = S(st);
    std::lock_guard<std::recursive_mutex> stream_lock(s->mu);
    flush(s);
    return wait_value(s->completion, s->submitted);
}
Error stream_query(Stream st) {
    StreamImpl* s = S(st);
    std::lock_guard<std::recursive_mutex> stream_lock(s->mu);
    flush(s);
    if (rt().failed.load()) return fail(kErrorLaunchFailure);
    std::lock_guard<std::mutex> lock(s->completion->mu);
    return s->completion->completed >= s->submitted ? kSuccess : kNotReady;
}
Error stream_wait_event(Stream st, Event e, unsigned) {
    StreamImpl* s = S(st);
    std::lock_guard<std::recursive_mutex> stream_lock(s->mu);
    if (s->rec) {
        Op o;
        o.kind = Op::Wait; o.ev = e->state;
        s->rec->ops.push_back(std::move(o));
        return kSuccess;
    }
    return wait_event(s, e->state);
}
Error launch_host_func(Stream st, HostFn fn, void* arg) {
    // skeleton: in stream order by completing the stream first (the engine's uses are rare; Phase 2 replaces this by a
    // completion handler + a shared-event wait so the host thread does not block)
    StreamImpl* s = S(st);
    std::lock_guard<std::recursive_mutex> stream_lock(s->mu);
    if (s->rec) return fail(kErrorUnsupported);
    flush(s);
    if (auto e = wait_value(s->completion, s->submitted); e != kSuccess) return e;
    fn(arg);
    return kSuccess;
}

Error event_create(Event* e, unsigned) { *e = new EventImpl; return kSuccess; }
Error event_destroy(Event e) { delete e; return kSuccess; }
Error event_record(Event e, Stream st) {
    StreamImpl* s = S(st);
    std::lock_guard<std::recursive_mutex> stream_lock(s->mu);
    if (s->rec) {
        Op o;
        o.kind = Op::Record; o.ev = e->state;
        s->rec->ops.push_back(std::move(o));
        return kSuccess;
    }
    return record_event(e->state, s);
}
Error event_query(Event e) {
    if (rt().failed.load()) return fail(kErrorLaunchFailure);
    std::lock_guard<std::mutex> event_lock(e->state->mu);
    auto c = e->state->completion;
    if (!c) return kSuccess;
    std::lock_guard<std::mutex> lock(c->mu);
    return c->completed >= e->state->value ? kSuccess : kNotReady;
}
Error event_synchronize(Event e) {
    std::shared_ptr<Completion> completion; uint64_t value;
    { std::lock_guard<std::mutex> lock(e->state->mu); completion=e->state->completion; value=e->state->value; }
    return wait_value(completion,value);
}

Error event_elapsed_ms(float* ms, Event, Event) { *ms = 0.0f; return fail(kErrorUnsupported); }

Error begin_capture(Stream st) {
    StreamImpl* s = S(st);
    std::lock_guard<std::recursive_mutex> stream_lock(s->mu);
    if (s->rec) return fail(kErrorInvalidValue);
    flush(s);
    s->rec = new GraphImpl;
    return kSuccess;
}
Error end_capture(Stream st, Graph* g) {
    StreamImpl* s = S(st);
    std::lock_guard<std::recursive_mutex> stream_lock(s->mu);
    if (!s->rec) { *g = nullptr; return fail(kErrorInvalidValue); }
    if (std::any_of(s->rec->ops.begin(),s->rec->ops.end(),[](const Op& o) {
            return o.kind == Op::Dispatch && !o.complete_resources;
        })) {
        auto hold = hold_resources({}, true);
        for (auto& op : s->rec->ops)
            if (op.kind == Op::Dispatch && !op.complete_resources) op.resources = hold;
    }
    *g = s->rec;
    s->rec = nullptr;
    return kSuccess;
}
Error graph_node_count(Graph g, size_t* n) { *n = g ? g->ops.size() : 0; return kSuccess; }
Error graph_instantiate(GraphExec* x, Graph g) {
    *x = new GraphExecImpl{g->ops, g->holds};
    return kSuccess;
}
Error graph_launch(GraphExec x, Stream st) {
    StreamImpl* s = S(st);
    std::lock_guard<std::recursive_mutex> stream_lock(s->mu);
    if (s->rec) {   // a graph launched into a capture becomes part of it (CUDA: a child graph)
        s->rec->ops.insert(s->rec->ops.end(), x->ops.begin(), x->ops.end());
        s->rec->holds.insert(s->rec->holds.end(), x->holds.begin(), x->holds.end());
        return kSuccess;
    }
    s->resources.insert(s->resources.end(),x->holds.begin(),x->holds.end());
    for (const Op& o : x->ops) {
        const Error e = run_op(s, o);
        if (e != kSuccess) return e;
    }
    return kSuccess;
}
Error graph_upload(GraphExec, Stream) { return kSuccess; }
Error graph_destroy(Graph g) { delete g; return kSuccess; }
Error graph_exec_destroy(GraphExec x) { delete x; return kSuccess; }

// ---------------------------------------------------------------------------------------------------- metal.hpp
namespace metal {

MTL::Device* device() { return rt().dev; }
#if defined(STRATA_METAL_TESTING)
namespace testing {
void fail_next_completion() { rt().fail_next.store(true); }
void fail_allocation_after(int count) { rt().fail_allocation.store(count); }
size_t last_residency_bytes() { return rt().residency_bytes.load(); }
size_t allocated_bytes() {
    std::lock_guard<std::mutex> lock(rt().mu);size_t bytes=0;
    for(const auto& [key,a]:rt().by_gpu)bytes+=a.bytes;
    return bytes;
}
}
#endif
void set_scoped_residency(bool enabled) { rt().scoped.store(enabled); }
void set_optimize(bool enabled) { rt().optimize.store(enabled); }
bool optimize() { return rt().optimize.load(); }
Error submit(Stream st) {
    auto* s = S(st);
    std::lock_guard<std::recursive_mutex> stream_lock(s->mu);
    if (s->rec) { Op o; o.kind = Op::Submit; s->rec->ops.push_back(o); }
    else flush(s);
    return rt().failed.load() ? fail(kErrorLaunchFailure) : kSuccess;
}


MTL::ComputePipelineState* pipeline(const char* source_name, const char* source, const char* function, std::string& err) {
    Runtime& r = rt();
    const std::string key = std::string(source_name) + "::" + function;
    std::lock_guard<std::mutex> lk(r.mu);
    if (auto it = r.pipes.find(key); it != r.pipes.end()) return it->second;
    MTL::Library*& lib = r.libs[source_name];
    NS::Error* e = nullptr;
    if (!lib) {
        MTL::CompileOptions* co = MTL::CompileOptions::alloc()->init();
        co->setMathMode(MTL::MathModeSafe);   // no fast-math reassociation: parity with the CUDA reference first
        lib = r.dev->newLibrary(NS::String::string(source, NS::UTF8StringEncoding), co, &e);
        co->release();
        if (!lib) {
            err = std::string("MSL compile of ") + source_name + ": " + (e ? e->localizedDescription()->utf8String() : "?");
            return nullptr;
        }
    }
    MTL::Function* f = lib->newFunction(NS::String::string(function, NS::UTF8StringEncoding));
    if (!f) { err = std::string("no MSL function ") + function; return nullptr; }
    MTL::ComputePipelineState* p = r.dev->newComputePipelineState(f, &e);
    f->release();
    if (!p) { err = std::string("pipeline ") + function + ": " + (e ? e->localizedDescription()->utf8String() : "?"); return nullptr; }
    r.pipes[key] = p;
    return p;
}

Error register_indirect(const void* table, const void* const* pointers, size_t count) {
    Runtime& r = rt();
    std::lock_guard<std::mutex> lock(r.mu);
    size_t offset = 0;
    const auto* a = find_unlocked(r, table, &offset);
    if (!a) return fail(kErrorInvalidValue);
    const auto key = a->gpu;
    std::vector<const void*> buffers;
    const Alloc* last = nullptr;
    for (size_t i = 0; i < count; ++i) if (pointers && pointers[i]) {
        const uint64_t addr = (uint64_t) (uintptr_t) pointers[i];
        const auto* target = (last && addr >= last->gpu && addr < last->gpu + last->bytes)
                                 ? last
                                 : find_unlocked(r, pointers[i], &offset);
        if (!target || target->purged) return fail(kErrorInvalidValue);
        last = target;
        const void* base = reinterpret_cast<const void*>(target->gpu);
        if (std::find(buffers.begin(), buffers.end(), base) == buffers.end())
            buffers.push_back(base);
    }
    r.indirect[key] = std::move(buffers);
    return kSuccess;
}

Error dispatch(Stream st, MTL::ComputePipelineState* p, const void* args, size_t arg_bytes, Grid groups, Grid threads,
               size_t tg_bytes, Resources resources) {
    if (arg_bytes > 4096) return fail(kErrorInvalidValue);   // setBytes limit
    {   // Metal skips a dispatch above the pipeline's maxTotalThreadsPerThreadgroup without any error (#2323): stop instead
        const uint64_t want = (uint64_t) threads.x * threads.y * threads.z;
        if (p && want > (uint64_t) p->maxTotalThreadsPerThreadgroup()) {
            Runtime& r = rt();
            std::string name = "?";
            {
                std::lock_guard<std::mutex> lk(r.mu);
                for (const auto& kv : r.pipes)
                    if (kv.second == p) { name = kv.first; break; }
            }
            std::fprintf(stderr,
                         "strata metal: the kernel %s can run at most %lu threads per threadgroup on this GPU (%s), but the "
                         "launch asks for %lu.\nMetal skips such a dispatch without any error, so every answer would be "
                         "wrong; stopping here instead.  Please report this (the GPU, macOS version and this line) at "
                         "https://github.com/Niko1221/Strata/issues .  (#2323)\n",
                         name.c_str(), (unsigned long) p->maxTotalThreadsPerThreadgroup(),
                         r.dev->name()->utf8String(), (unsigned long) want);
            std::fflush(stderr);
            std::exit(6);
        }
    }
    StreamImpl* s = S(st);
    std::lock_guard<std::recursive_mutex> stream_lock(s->mu);
    Op o;
    o.kind = Op::Dispatch;
    std::vector<const void*> pointers;
    if (resources.count) pointers.assign(resources.pointers, resources.pointers + resources.count);
    bool complete = resources.complete;
    if (resources.indirect_table) {
        Runtime& r = rt();
        std::lock_guard<std::mutex> lock(r.mu);
        size_t off = 0;
        const auto* allocation = find_unlocked(r, resources.indirect_table, &off);
        auto it = allocation ? r.indirect.find(allocation->gpu) : r.indirect.end();
        if (it == r.indirect.end()) complete = false;
    }
    o.resources = complete ? hold_resources(pointers) : nullptr;
    if (complete && !o.resources) return fail(kErrorInvalidValue);
    o.complete_resources = complete;
    o.indirect_table = resources.indirect_table;
    o.pipe = p;
    if (arg_bytes <= sizeof o.args) { std::memcpy(o.args, args, arg_bytes); o.arg_bytes = arg_bytes; }
    else o.big_args.assign((const uint8_t*) args, (const uint8_t*) args + arg_bytes);
    o.groups = groups;
    o.threads = threads;
    o.tg_bytes = tg_bytes;
    if (s->rec) { s->rec->ops.push_back(std::move(o)); return kSuccess; }
    return encode_dispatch(s, o);
}

Stats stats() {
    const auto& s=rt().stats;
    return {s.dispatches.load(),s.command_buffers.load(),s.blits.load(),s.host_syncs.load()};
}

}  // namespace metal
}  // namespace strata::gpu
