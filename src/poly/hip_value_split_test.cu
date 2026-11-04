// HIP attention A/B, shape fallback, graph replay, and an independent FP64
// reference for the weighted-value reduction. No GGUF or model is needed.
#include "strata/poly/kernels.hpp"
#include "hip_value_split_policy.hpp"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#if defined(STRATA_USE_HIP)
#include "hip_value_split.cuh"
#endif
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <random>
#include <stdexcept>
#include <string>
#include <vector>

#if defined(STRATA_USE_HIP)
namespace {
using namespace strata::poly;
constexpr int H = 16, KH = 2, D = 256, N = 100352, R = 17, CHUNK = 8;
constexpr const char* key = "STRATA_HIP_VALUE_SPLIT";
int checks = 0, changed = 0;
FILE* default_dump = nullptr;
float largest = 0.f;
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void ck(hipError_t e) { if (e != hipSuccess) throw std::runtime_error(hipGetErrorString(e)); }
void flag(const char* s) {
#ifdef _WIN32
    require(_putenv_s(key, s ? s : "") == 0, "set environment");
#else
    require((s ? setenv(key, s, 1) : unsetenv(key)) == 0, "set environment");
#endif
}
struct Restore {
    bool had = false;
    std::string old;
    Restore() { if (const char* s = std::getenv(key)) { had = true; old = s; } }
    ~Restore() { flag(had ? old.c_str() : nullptr); }
};
struct Device {
    void* p = nullptr;
    explicit Device(size_t bytes) { ck(hipMalloc(&p, bytes)); }
    ~Device() { if (p) hipFree(p); }
    template <class T> T* as() const { return static_cast<T*>(p); }
    template <class T> void upload(const std::vector<T>& x) {
        ck(hipMemcpy(p, x.data(), x.size() * sizeof(T), hipMemcpyHostToDevice));
    }
};
std::vector<float> read(const Device& d, size_t n) {
    std::vector<float> x(n);
    ck(hipMemcpy(x.data(), d.p, n * sizeof(float), hipMemcpyDeviceToHost));
    return x;
}
float delta(const std::vector<float>& a, const std::vector<float>& b) {
    require(a.size() == b.size(), "output size");
    float error = 0.f;
    for (size_t i = 0; i < a.size(); ++i) {
        require(std::isfinite(a[i]) && std::isfinite(b[i]), "nonfinite output");
        error = std::max(error, std::fabs(a[i] - b[i]));
    }
    largest = std::max(largest, error);
    return error;
}
struct Fixture {
    std::vector<float> q;
    std::vector<int8_t> k, v;
    std::vector<uint16_t> ks, vs, kh, vh;
    Device dq{size_t(R) * H * D * sizeof(float)}, dk{size_t(N) * KH * D}, dv{size_t(N) * KH * D};
    Device dks{size_t(N) * KH * D / 32 * 2}, dvs{size_t(N) * KH * D / 32 * 2};
    Device dkh{size_t(N) * KH * D * 2}, dvh{size_t(N) * KH * D * 2};
    Device scratch{size_t(CHUNK) * H * N * sizeof(float)}, out{size_t(R) * H * D * sizeof(float)};
    Device at{sizeof(long long)}, sink{H * sizeof(float)};
    explicit Fixture(bool wide = false) : q(size_t(R) * H * D), k(size_t(N) * KH * D), v(k.size()),
                ks(k.size() / 32), vs(ks.size()), kh(k.size()), vh(k.size()) {
        std::mt19937 rng(71829);
        std::uniform_real_distribution<float> f(-2.f, 2.f);
        std::uniform_int_distribution<int> b(-127, 127);
        for (float& x : q) x = f(rng) * (wide ? 2.f : 1.f);
        for (auto& x : k) x = (int8_t) b(rng);
        for (auto& x : v) x = (int8_t) b(rng);
        for (size_t i = 0; i < ks.size(); ++i) {
            ks[i] = __half_as_ushort(__float2half((.002f + float(i % 19) * .0003f) * (wide ? 8.f : 1.f)));
            vs[i] = __half_as_ushort(__float2half((.002f + float(i % 23) * .0004f) * (wide ? 8.f : 1.f)));
        }
        for (size_t i = 0; i < k.size(); ++i) {
            kh[i] = __half_as_ushort(__float2half(float(k[i]) * __half2float(__ushort_as_half(ks[i / 32]))));
            vh[i] = __half_as_ushort(__float2half(float(v[i]) * __half2float(__ushort_as_half(vs[i / 32]))));
        }
        dq.upload(q); dk.upload(k); dv.upload(v); dks.upload(ks); dvs.upload(vs); dkh.upload(kh); dvh.upload(vh);
        std::vector<float> s(H);
        for (int h = 0; h < H; ++h) s[h] = float(h - 8) * .2f;
        sink.upload(s);
    }
    void position(long long pos) { ck(hipMemcpy(at.p, &pos, sizeof pos, hipMemcpyHostToDevice)); }
    void attend_now(bool quant, int depth, int rows, bool dynamic, int dim = D, int heads = H,
                    int kv_heads = KH, int window = 0, int ring = 0, bool whole = false,
                    bool sinks = false, hipStream_t stream = nullptr) {
        attend(dq.as<float>(), quant ? nullptr : dkh.as<uint16_t>(), quant ? nullptr : dvh.as<uint16_t>(),
               dynamic ? 0 : depth - 1, rows, scratch.as<float>(), CHUNK, out.as<float>(), dim, heads, kv_heads,
               1.f / std::sqrt(float(dim)), (void*) stream, window, ring, whole,
               dynamic ? at.as<long long>() : nullptr, dynamic ? N : 0, size_t(CHUNK) * H * N,
               quant ? Kvq8{dk.as<int8_t>(), dks.as<uint16_t>()} : Kvq8{},
               quant ? Kvq8{dv.as<int8_t>(), dvs.as<uint16_t>()} : Kvq8{}, sinks ? sink.as<float>() : nullptr);
    }
};
void attention_checks(Fixture& f) {
    for (bool quant : {false, true}) for (bool dynamic : {false, true})
        for (int rows : {1, 2, 4, 9, 12, 17})
            for (int depth : {8191, 8192, 8193, 9215, 9216, 9217, 32768, 98304}) {
                f.position(depth - 1);
                flag(nullptr); f.attend_now(quant, depth, rows, dynamic);
                const auto ref = read(f.out, size_t(rows) * H * D);
                if (default_dump)
                    require(std::fwrite(ref.data(), sizeof(float), ref.size(), default_dump) == ref.size(),
                            "write default output");
                flag("0"); f.attend_now(quant, depth, rows, dynamic);
                const auto zero = read(f.out, ref.size());
                require(!std::memcmp(ref.data(), zero.data(), ref.size() * sizeof(float)), "0 changed default bits");
                flag("1"); f.attend_now(quant, depth, rows, dynamic);
                const auto on = read(f.out, ref.size());
                changed += std::memcmp(ref.data(), on.data(), ref.size() * sizeof(float)) != 0;
                const float error = delta(ref, on);
                if (error > 2e-5f) {
                    std::fprintf(stderr, "quant=%d dynamic=%d rows=%d depth=%d error=%g\n", quant, dynamic,
                                 rows, depth, error);
                    throw std::runtime_error("attention tolerance");
                }
                ++checks;
            }
    // The layout/geometry fallbacks must retain exactly the original result.
    for (const auto shape : {std::vector<int>{128, 16, 2, 0, 0, 0}, {64, 32, 8, 0, 0, 0},
                             {256, 16, 1, 0, 0, 0}, {256, 16, 2, 1024, 8192, 0},
                             {256, 16, 2, 0, 0, 1}}) {
        flag(nullptr); f.attend_now(false, 16384, 4, false, shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
        const auto ref = read(f.out, size_t(4) * shape[0] * shape[1]);
        flag("1"); f.attend_now(false, 16384, 4, false, shape[0], shape[1], shape[2], shape[3], shape[4], shape[5]);
        const auto on = read(f.out, ref.size());
        require(!std::memcmp(ref.data(), on.data(), ref.size() * sizeof(float)), "fallback changed bits");
        ++checks;
    }
    for (bool quant : {false, true}) {
        flag(nullptr); f.attend_now(quant, 32768, 4, false, D, H, KH, 0, 0, false, true);
        const auto ref = read(f.out, size_t(4) * H * D);
        flag("1"); f.attend_now(quant, 32768, 4, false, D, H, KH, 0, 0, false, true);
        require(delta(ref, read(f.out, ref.size())) <= 2e-5f, "attention sinks");
        ++checks;
    }
}
void graph_checks(Fixture& f) {
    flag("1");
    for (bool quant : {false, true}) for (int rows : {1, 4}) {
        hipStream_t stream; ck(hipStreamCreate(&stream));
        hipGraph_t graph; hipGraphExec_t exec;
        f.position(8191);
        ck(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
        f.attend_now(quant, 8192, rows, true, D, H, KH, 0, 0, false, false, stream);
        ck(hipStreamEndCapture(stream, &graph));
        ck(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        for (int depth : {8192, 32768, 98304}) {
            f.position(depth - 1);
            f.attend_now(quant, depth, rows, true);
            const auto eager = read(f.out, size_t(rows) * H * D);
            ck(hipGraphLaunch(exec, stream)); ck(hipStreamSynchronize(stream));
            const auto replay = read(f.out, eager.size());
            require(!std::memcmp(eager.data(), replay.data(), eager.size() * sizeof(float)), "graph replay differs");
            ++checks;
        }
        ck(hipGraphExecDestroy(exec)); ck(hipGraphDestroy(graph)); ck(hipStreamDestroy(stream));
    }
}
void value_reference(Fixture& f) {
    constexpr int keys = 8193, rows = 4;
    std::mt19937 rng(951);
    std::uniform_real_distribution<float> rand(.01f, 1.f);
    std::vector<float> weights(size_t(rows) * H * keys, 0.f);
    for (int t = 0; t < rows; ++t) for (int h = 0; h < H; ++h) {
        const int end = keys - rows + t + 1;
        double sum = 0;
        for (int k = 0; k < end; ++k) {
            float& w = weights[((size_t(h / 8) * rows + t) * keys + k) * 8 + h % 8];
            w = rand(rng); sum += w;
        }
        for (int k = 0; k < end; ++k)
            weights[((size_t(h / 8) * rows + t) * keys + k) * 8 + h % 8] /= float(sum);
    }
    Device wt(weights.size() * sizeof(float)); wt.upload(weights);
    for (bool quant : {false, true}) {
        hip_value_split::run(wt.as<float>(), quant ? nullptr : f.dvh.as<uint16_t>(),
                             quant ? f.dv.as<int8_t>() : nullptr, quant ? f.dvs.as<uint16_t>() : nullptr,
                             f.scratch.as<float>(), f.out.as<float>(), keys, H, KH, 0, rows,
                             keys - rows, nullptr, nullptr);
        const auto on = read(f.out, size_t(rows) * H * D);
        double error = 0;
        for (int t = 0; t < rows; ++t) for (int h = 0; h < H; ++h) for (int d = 0; d < D; ++d) {
            double sum = 0;
            for (int k = 0; k < keys - rows + t + 1; ++k) {
                const size_t v = (size_t(k) * KH + h / 8) * D + d;
                const double value = quant ? double(f.v[v]) * __half2float(__ushort_as_half(f.vs[v / 32]))
                                           : __half2float(__ushort_as_half(f.vh[v]));
                sum += value * weights[((size_t(h / 8) * rows + t) * keys + k) * 8 + h % 8];
            }
            error = std::max(error, std::fabs(sum - on[(size_t(t) * H + h) * D + d]));
        }
        require(error <= 2e-6, "FP64 weighted-value reference");
        std::printf("FP64 kv=%s max_abs=%.9g\n", quant ? "q8_0" : "fp16", error);
        const auto unchanged = read(wt, weights.size());
        require(!std::memcmp(unchanged.data(), weights.data(), weights.size() * sizeof(float)), "weights overwritten");
        ++checks;
    }
}
}  // namespace
#endif

int main(int argc, char** argv) {
#if defined(STRATA_USE_HIP)
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count < 1) return 77;
    Restore restore;
    try {
        if (argc == 3 && !std::strcmp(argv[1], "--dump-default")) {
            default_dump = std::fopen(argv[2], "wb");
            require(default_dump != nullptr, "open default output");
        } else require(argc == 1, "usage: poly_hip_value_split_test [--dump-default FILE]");
        {
            Fixture f;
            attention_checks(f);
            if (!default_dump) { graph_checks(f); value_reference(f); }
        }
        // Larger values and sharper scores also exercise peaked softmax weights
        // and cancellation, rather than only near-uniform long-context sums.
        {
            Fixture wide(true);
            attention_checks(wide);
            if (!default_dump) value_reference(wide);
        }
        if (default_dump) {
            require(std::fclose(default_dump) == 0, "close default output");
            std::printf("poly_hip_value_split_test: default output written, checks=%d\n", checks);
            return 0;
        }
        require(changed > 0, "the opt-in path was not exercised");
        ck(hipDeviceSynchronize()); ck(hipGetLastError());
        std::printf("poly_hip_value_split_test: PASS checks=%d max_attention_abs=%.9g\n", checks, largest);
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
    return 0;
#else
    (void) argc; (void) argv;
    std::puts("poly_hip_value_split_test: SKIP (HIP only)");
    return 77;
#endif
}
