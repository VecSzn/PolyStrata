// HIP prompt attention on FP32 inputs (STRATA_HIP_PREFILL_FP32=1): attend_batch() over q8_0 caches against an independent FP64 reference and the
// default path (FP16 inputs on the products), the fallbacks (a workspace too short for FP32 rows, an FP16 cache), and a ragged chunk.
// No GGUF or model is needed; needs a card (exit 77 without a HIP device).
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
constexpr const char* key = "STRATA_HIP_PREFILL_FP32";
int checks = 0;
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
template <class T> struct Dev {
    T* p = nullptr;
    explicit Dev(size_t n) { ck(hipMalloc((void**) &p, n * sizeof(T))); }
    ~Dev() { if (p) hipFree(p); }
    void up(const std::vector<T>& x) { ck(hipMemcpy(p, x.data(), x.size() * sizeof(T), hipMemcpyHostToDevice)); }
};

struct Geometry { int dim, heads, kv; };

struct Case {
    Geometry g;
    int ctx, rows;
    size_t score_floats, work_halves;
};

// the caches and queries for a case, FP64 reference from the dequantized caches
struct Fixture {
    Geometry g;
    int ctx, rows, width, per;
    std::vector<int8_t> kc, vc;
    std::vector<uint16_t> ks, vs;
    std::vector<float> q;
    explicit Fixture(Geometry geo, int c, int r) : g(geo), ctx(c), rows(r), width(geo.kv * geo.dim), per(geo.heads / geo.kv) {
        std::mt19937 rng(20261106 + c);
        std::uniform_int_distribution<int> code(-127, 127), sc(0, 7);
        std::uniform_real_distribution<float> qv(-2.0f, 2.0f);
        kc.resize(size_t(ctx) * width); vc.resize(kc.size());
        ks.resize(kc.size() / 32); vs.resize(ks.size());
        q.resize(size_t(rows) * g.heads * g.dim);
        for (auto& x : kc) x = int8_t(code(rng));
        for (auto& x : vc) x = int8_t(code(rng));
        for (auto& s : ks) s = __half_as_ushort(__float2half(0.004f + 0.0005f * sc(rng)));
        for (auto& s : vs) s = __half_as_ushort(__float2half(0.004f + 0.0005f * sc(rng)));
        for (auto& x : q) x = qv(rng);
    }
    void reference(int t, int h, std::vector<double>& out) const {
        const int grp = h / per, keys = ctx - rows + t + 1;
        std::vector<double> s(keys);
        double top = -1e300;
        for (int k = 0; k < keys; ++k) {
            const size_t r = size_t(k) * width + size_t(grp) * g.dim;
            double a = 0;
            for (int d = 0; d < g.dim; ++d)
                a += double(q[(size_t(t) * g.heads + h) * g.dim + d]) * kc[r + d] * double(__half2float(__ushort_as_half(ks[(r + d) / 32])));
            s[k] = a / std::sqrt(double(g.dim));
            top = std::max(top, s[k]);
        }
        double sum = 0;
        for (int k = 0; k < keys; ++k) { s[k] = std::exp(s[k] - top); sum += s[k]; }
        out.assign(g.dim, 0.0);
        for (int k = 0; k < keys; ++k) {
            const size_t r = size_t(k) * width + size_t(grp) * g.dim;
            for (int d = 0; d < g.dim; ++d) out[d] += s[k] * vc[r + d] * double(__half2float(__ushort_as_half(vs[(r + d) / 32])));
        }
        for (auto& x : out) x /= sum;
    }
};

std::vector<float> run(const Fixture& f, const Case& c, const char* mode, bool q8, void* blas) {
    const size_t hq = size_t(f.rows) * f.g.heads * f.g.dim;
    Dev<int8_t> kc(f.kc.size()), vc(f.vc.size());
    Dev<uint16_t> ks(f.ks.size()), vs(f.vs.size());
    Dev<float> q(hq), out(hq), score(c.score_floats), work(hq + hq / 2 + 1 + 3 * size_t(f.rows) * f.g.heads);
    Dev<float> w8(c.work_halves + 1);   // two workspaces of work_halves halves
    Dev<uint16_t> kf(q8 ? 1 : f.kc.size()), vf(q8 ? 1 : f.kc.size());
    kc.up(f.kc); vc.up(f.vc); ks.up(f.ks); vs.up(f.vs); q.up(f.q);
    if (!q8) {   // the same caches as halves
        std::vector<uint16_t> kh(f.kc.size()), vh(f.kc.size());
        for (size_t i = 0; i < kh.size(); ++i) {
            kh[i] = __half_as_ushort(__float2half(float(f.kc[i]) * __half2float(__ushort_as_half(f.ks[i / 32]))));
            vh[i] = __half_as_ushort(__float2half(float(f.vc[i]) * __half2float(__ushort_as_half(f.vs[i / 32]))));
        }
        kf.up(kh); vf.up(vh);
    }
    flag(mode);
    uint16_t* kw = (uint16_t*) w8.p;
    uint16_t* vw = kw + c.work_halves;
    attend_batch(blas, q.p, q8 ? nullptr : kf.p, q8 ? nullptr : vf.p, c.ctx - c.rows, c.rows, score.p, c.score_floats, work.p, out.p, f.g.dim, f.g.heads,
                 f.g.kv, 1.0f / std::sqrt(float(f.g.dim)), nullptr, 0, 0, false, q8 ? Kvq8{kc.p, ks.p} : Kvq8{}, q8 ? Kvq8{vc.p, vs.p} : Kvq8{}, q8 ? kw : nullptr,
                 q8 ? vw : nullptr, q8 ? c.work_halves : 0);
    ck(hipDeviceSynchronize()); ck(hipGetLastError());
    std::vector<float> o(hq);
    ck(hipMemcpy(o.data(), out.p, hq * 4, hipMemcpyDeviceToHost));
    return o;
}

double against(const Fixture& f, const std::vector<float>& o, double& largest) {
    double e = 0;
    for (int t : {0, f.rows / 2, f.rows - 1})
        for (int h = 0; h < f.g.heads; ++h) {
            std::vector<double> r;
            f.reference(t, h, r);
            for (int d = 0; d < f.g.dim; ++d) {
                largest = std::max(largest, std::fabs(r[d]));
                e = std::max(e, std::fabs(double(o[(size_t(t) * f.g.heads + h) * f.g.dim + d]) - r[d]));
            }
        }
    return e;
}

void one(const Case& c, void* blas) {
    const Fixture f(c.g, c.ctx, c.rows);
    const std::string what = "dim " + std::to_string(c.g.dim) + " ctx " + std::to_string(c.ctx) + " rows " + std::to_string(c.rows) + " scratch " + std::to_string(c.score_floats);
    const auto old = run(f, c, "0", true, blas), fp32 = run(f, c, "1", true, blas);
    double lg = 0;
    const double e0 = against(f, old, lg), e1 = against(f, fp32, lg);
    std::printf("%-58s against FP64: default %.2e, FP32 %.2e (largest %.3g)\n", what.c_str(), e0, e1, lg);
    require(e1 <= 1e-5 * lg, "FP32 path against FP64: " + what);
    require(e0 <= 1e-3 * lg, "default path against FP64: " + what);
    ++checks;
    ++checks;
    require(std::memcmp(old.data(), fp32.data(), old.size() * 4) != 0, "the FP32 path was not taken: " + what);
    // a workspace too short for 256 FP32 rows: the default path, bitwise
    Case shorter = c;
    shorter.work_halves = size_t(200) * f.width;
    const auto fall = run(f, shorter, "1", true, blas), plain = run(f, shorter, "0", true, blas);
    require(!std::memcmp(fall.data(), plain.data(), plain.size() * 4), "short workspace is not the default path: " + what);
    ++checks;
}
}  // namespace
#endif

int main() {
#if defined(STRATA_USE_HIP)
    int count = 0;
    if (hipGetDeviceCount(&count) != hipSuccess || count < 1) return 77;
    Restore restore;
    try {
        void* blas = blas_create(nullptr);
        const size_t big = size_t(16) << 20, wh2048 = size_t(2048) * 512;
        // the engine's buffers (16M floats of scores, 2048 rows of halves a workspace): blocks of a thousand keys and more, and several chunks of tokens
        one({{256, 16, 2}, 2047, 3, big, wh2048}, blas);
        one({{256, 16, 2}, 5000, 70, big, wh2048}, blas);
        one({{256, 16, 2}, 9000, 600, big, wh2048}, blas);   // two chunks of 512 and 88
        one({{256, 16, 2}, 12000, 500, size_t(9) << 20, wh2048}, blas);   // little scratch: blocks of a few hundred keys, many of them
        one({{128, 32, 4}, 6000, 33, big, size_t(2048) * 512}, blas);   // another geometry: heads of 128, 8 to a group
        // an FP16 cache is not this path's: the same bits with the flag set
        {
            const Case c{{256, 16, 2}, 4000, 20, big, wh2048};
            const Fixture f(c.g, c.ctx, c.rows);
            const auto a = run(f, c, "0", false, blas), b = run(f, c, "1", false, blas);
            require(!std::memcmp(a.data(), b.data(), a.size() * 4), "FP16 cache differs with the flag");
            ++checks;
        }
        ck(hipDeviceSynchronize()); ck(hipGetLastError());
        std::printf("poly_hip_prefill_fp32_test: PASS checks=%d\n", checks);
        blas_destroy(blas);
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
    return 0;
#else
    std::puts("poly_hip_prefill_fp32_test: SKIP (HIP only)");
    return 77;
#endif
}
