// HIP split attention (STRATA_HIP_SPLIT_ATTEND=1 and 2): attend() for heads of 256 values in groups of 8 over a q8_0 cache, against an independent
// FP64 reference and the old kernels, with the position on the card, the fallback where the room for its partial sums is short, and a graph replay.
// No GGUF or model is needed; needs a card (exit 77 without a HIP device or without 32-lane waves).
#include "strata/poly/kernels.hpp"

#include <cuda_fp16.h>
#include <cuda_runtime.h>

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
constexpr int H = 16, KH = 2, D = 256, MAXC = 100352 + 16, CHUNK = 4;
constexpr const char* key = "STRATA_HIP_SPLIT_ATTEND";
int checks = 0, engaged = 0;
void require(bool ok, const std::string& why) { if (!ok) throw std::runtime_error(why); }
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
    template <class T> void upload(const std::vector<T>& x) { ck(hipMemcpy(p, x.data(), x.size() * sizeof(T), hipMemcpyHostToDevice)); }
};
std::vector<float> read(const Device& d, size_t n) {
    std::vector<float> x(n);
    ck(hipMemcpy(x.data(), d.p, n * sizeof(float), hipMemcpyDeviceToHost));
    return x;
}

struct Fixture {
    size_t score_floats = size_t(CHUNK) * H * MAXC;
    std::vector<int8_t> kc, vc;
    std::vector<uint16_t> ks, vs;
    std::vector<float> q;
    Device dkc, dvc, dks, dvs, dq, dscore, dout, dat;
    Fixture()
        : kc(size_t(MAXC) * KH * D), vc(kc.size()), ks(kc.size() / 32), vs(ks.size()), q(size_t(8) * H * D), dkc(kc.size()), dvc(kc.size()),
          dks(ks.size() * 2), dvs(ks.size() * 2), dq(q.size() * 4), dscore(score_floats * 4), dout(size_t(8) * H * D * 4), dat(8) {
        std::mt19937 rng(20261105);
        std::uniform_int_distribution<int> code(-127, 127), sc(0, 7);
        std::uniform_real_distribution<float> qv(-2.0f, 2.0f);
        for (auto& c : kc) c = int8_t(code(rng));
        for (auto& c : vc) c = int8_t(code(rng));
        for (auto& s : ks) s = __half_as_ushort(__float2half(0.004f + 0.0005f * sc(rng)));
        for (auto& s : vs) s = __half_as_ushort(__float2half(0.004f + 0.0005f * sc(rng)));
        for (auto& x : q) x = qv(rng);
        dkc.upload(kc); dvc.upload(vc); dks.upload(ks); dvs.upload(vs); dq.upload(q);
    }
    // attend() as the engine calls it for a few tokens: q8 K and V, part_floats the scratch for the split kernel's sums
    void attend_rows(long long pos, int rows, const long long* at = nullptr, int room = 0, size_t part = ~size_t(0), hipStream_t stream = nullptr) {
        attend(dq.as<float>(), nullptr, nullptr, pos, rows, dscore.as<float>(), CHUNK, dout.as<float>(), D, H, KH, 1.0f / 16.0f, stream, 0, 0, false, at, room,
               0, Kvq8{dkc.as<int8_t>(), dks.as<uint16_t>()}, Kvq8{dvc.as<int8_t>(), dvs.as<uint16_t>()}, nullptr, part == ~size_t(0) ? score_floats : part);
    }
    std::vector<float> out(int rows) { ck(hipDeviceSynchronize()); ck(hipGetLastError()); return read(dout, size_t(rows) * H * D); }
};

float lo_hi(const std::vector<float>& a, const std::vector<float>& b, float& largest) {
    require(a.size() == b.size(), "output size");
    float e = 0.f;
    for (size_t i = 0; i < a.size(); ++i) { e = std::max(e, std::fabs(a[i] - b[i])); largest = std::max(largest, std::fabs(b[i])); }
    return e;
}

// an independent reference in FP64 from the dequantized caches
std::vector<float> reference(const Fixture& f, long long pos, int rows) {
    std::vector<float> out(size_t(rows) * H * D);
    std::vector<double> s;
    for (int t = 0; t < rows; ++t)
        for (int h = 0; h < H; ++h) {
            const int g = h / (H / KH), keys = int(pos) + t + 1;
            s.assign(keys, 0.0);
            double top = -1e300;
            for (int k = 0; k < keys; ++k) {
                const size_t row = (size_t(k) * KH + g) * D;
                double a = 0;
                for (int d = 0; d < D; ++d)
                    a += double(f.q[(size_t(t) * H + h) * D + d]) * double(f.kc[row + d]) * double(__half2float(__ushort_as_half(f.ks[(row + d) / 32])));
                s[k] = a / 16.0;
                top = std::max(top, s[k]);
            }
            double sum = 0;
            for (int k = 0; k < keys; ++k) { s[k] = std::exp(s[k] - top); sum += s[k]; }
            for (int d = 0; d < D; ++d) {
                double a = 0;
                for (int k = 0; k < keys; ++k) {
                    const size_t row = (size_t(k) * KH + g) * D;
                    a += s[k] * double(f.vc[row + d]) * double(__half2float(__ushort_as_half(f.vs[(row + d) / 32])));
                }
                out[(size_t(t) * H + h) * D + d] = float(a / sum);
            }
        }
    return out;
}

void case_check(Fixture& f, int keys, int rows, bool with_reference) {
    const long long pos = keys - rows;
    float largest = 0.f;
    flag("0");
    f.attend_rows(pos, rows);
    const auto old = f.out(rows);
    flag("1");
    f.attend_rows(pos, rows);
    const auto m1 = f.out(rows);
    flag("2");
    f.attend_rows(pos, rows);
    const auto m2 = f.out(rows);
    const float d1 = lo_hi(m1, old, largest), d2 = lo_hi(m2, old, largest);
    require(d1 <= 1e-4f * largest, "mode 1 differs from the old kernels: keys " + std::to_string(keys) + " rows " + std::to_string(rows));
    require(d2 <= 1e-2f * largest, "mode 2 differs from the old kernels: keys " + std::to_string(keys) + " rows " + std::to_string(rows));
    if (std::memcmp(m1.data(), old.data(), m1.size() * sizeof(float)) != 0) ++engaged;
    if (with_reference) {
        const auto ref = reference(f, pos, rows);
        float lg = 0.f;
        const float e0 = lo_hi(old, ref, lg), e1 = lo_hi(m1, ref, lg), e2 = lo_hi(m2, ref, lg);
        require(e1 <= 1e-4f * lg, "mode 1 against FP64: keys " + std::to_string(keys) + " rows " + std::to_string(rows));
        require(e2 <= 1e-2f * lg, "mode 2 against FP64: keys " + std::to_string(keys) + " rows " + std::to_string(rows));
        std::printf("keys %6d rows %d: against FP64 old %.2e, mode 1 %.2e, mode 2 %.2e (largest %.3g)\n", keys, rows, e0, e1, e2, lg);
    }
    // the position on the card, room for the whole cache: the same sums as the position as an argument
    const long long p = pos;
    ck(hipMemcpy(f.dat.p, &p, sizeof p, hipMemcpyHostToDevice));
    flag("1");
    ck(hipMemset(f.dout.p, 0, size_t(rows) * H * D * 4));
    f.attend_rows(0, rows, f.dat.as<long long>(), MAXC - 16);
    const auto at1 = f.out(rows);
    require(!std::memcmp(at1.data(), m1.data(), m1.size() * sizeof(float)), "position on the card differs: keys " + std::to_string(keys));
    // the room for its partial sums short: the old kernels' answer, untouched
    flag("1");
    f.attend_rows(pos, rows, nullptr, 0, 64);
    const auto fall = f.out(rows);
    require(!std::memcmp(fall.data(), old.data(), old.size() * sizeof(float)), "fallback is not the old path: keys " + std::to_string(keys));
    checks += 4;
}

void graph_check(Fixture& f) {
    flag("2");
    hipStream_t stream;
    ck(hipStreamCreate(&stream));
    long long p = 20000 - 3;
    ck(hipMemcpy(f.dat.p, &p, sizeof p, hipMemcpyHostToDevice));
    hipGraph_t graph;
    hipGraphExec_t exec;
    ck(hipStreamBeginCapture(stream, hipStreamCaptureModeGlobal));
    f.attend_rows(0, 3, f.dat.as<long long>(), MAXC - 16, ~size_t(0), stream);
    ck(hipStreamEndCapture(stream, &graph));
    ck(hipGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    for (long long pos : {20000LL - 3, 33333LL - 3, 70000LL - 3}) {
        ck(hipMemcpy(f.dat.p, &pos, sizeof pos, hipMemcpyHostToDevice));
        ck(hipGraphLaunch(exec, stream));
        ck(hipStreamSynchronize(stream));
        const auto replay = f.out(3);
        f.attend_rows(pos, 3);   // eagerly, the position as an argument
        const auto eager = f.out(3);
        require(!std::memcmp(replay.data(), eager.data(), eager.size() * sizeof(float)), "graph replay differs at position " + std::to_string(pos));
        ++checks;
    }
    ck(hipGraphExecDestroy(exec)); ck(hipGraphDestroy(graph)); ck(hipStreamDestroy(stream));
}
}  // namespace
#endif

int main() {
#if defined(STRATA_USE_HIP)
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count < 1) return 77;
    hipDeviceProp_t prop;
    if (hipGetDeviceProperties(&prop, 0) != hipSuccess || prop.warpSize != 32) {
        std::puts("poly_hip_split_attend_test: SKIP (needs 32-lane waves)");
        return 77;
    }
    Restore restore;
    try {
        Fixture f;
        // sizes around the 4,096 keys it starts at and its blocks of 512, every row count up to its 4 a launch and beyond, FP64 where it is cheap
        case_check(f, 4096, 1, true);
        case_check(f, 4097, 2, true);
        case_check(f, 5000, 3, true);
        case_check(f, 8191, 8, true);
        case_check(f, 12345, 4, true);
        case_check(f, 33333, 1, true);
        case_check(f, 100001, 1, true);
        for (int rows : {2, 3, 4, 5, 6, 7}) case_check(f, 20001, rows, false);
        for (int keys : {4098, 4607, 4608, 4609, 65537, 100352}) case_check(f, keys, 4, false);
        graph_check(f);
        require(engaged > 0, "the opt-in path was not exercised");
        ck(hipDeviceSynchronize()); ck(hipGetLastError());
        std::printf("poly_hip_split_attend_test: PASS checks=%d\n", checks);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
    return 0;
#else
    std::puts("poly_hip_split_attend_test: SKIP (HIP only)");
    return 77;
#endif
}
