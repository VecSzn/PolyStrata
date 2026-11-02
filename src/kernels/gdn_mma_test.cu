// Standalone consumer: prepared scalar-gated GDN, without model/engine/Poly objects.
#include "strata/kernels/gdn_mma.hpp"
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <limits>
#include <random>
#include <stdexcept>
#include <vector>

using namespace strata::kernels;
namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void ck(cudaError_t e) { if (e != cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    float* p = nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p, n * sizeof(float))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<float>& x) { ck(cudaMemcpy(p, x.data(), x.size()*sizeof(float), cudaMemcpyHostToDevice)); }
    std::vector<float> get(size_t n) const {
        std::vector<float> x(n); ck(cudaMemcpy(x.data(), p, n*sizeof(float), cudaMemcpyDeviceToHost)); return x;
    }
};
void compare(const std::vector<float>& actual, const std::vector<float>& expected) {
    require(actual.size() == expected.size(), "result size");
    double error = 0, base = 0, maximum = 0;
    for (size_t i = 0; i < actual.size(); ++i) {
        require(std::isfinite(actual[i]), "nonfinite output");
        double delta = double(actual[i]) - expected[i];
        error += delta * delta; base += double(expected[i]) * expected[i];
        maximum = std::max(maximum, std::abs(delta));
    }
    double relative = std::sqrt(error / std::max(base, 1e-30));
    std::printf("L2 %.9g max %.9g\n", relative, maximum);
    require(relative < 5e-4 && maximum < .001, "shared kernel vs double recurrence");
}
void run(int heads, int rows, bool interleaved, bool separate_state) {
    constexpr int dim = 128, keys = 16;
    const int qpitch = interleaved ? (2*keys+heads)*dim : keys*dim;
    const int vpitch = interleaved ? qpitch : heads*dim;
    const size_t qcount = size_t(rows)*qpitch, vcount = size_t(rows)*vpitch;
    const size_t states = size_t(heads)*dim*dim, outputs = size_t(rows)*heads*dim;
    std::vector<float> q(qcount), k(qcount), v(vcount), g(rows*heads), beta(rows*heads), initial(states);
    std::mt19937 rng(41+heads+rows); std::uniform_real_distribution<float> u(-1.f, 1.f);
    for (int t=0;t<rows;++t) {
        for (int i=0;i<keys*dim;++i) { q[t*qpitch+i]=u(rng)*.01f; k[t*qpitch+i]=u(rng)*.1f; }
        for (int i=0;i<heads*dim;++i) v[t*vpitch+i]=u(rng)*.2f;
    }
    for (float& x:g) x=-.05f-.01f*std::abs(u(rng));
    for (float& x:beta) x=.5f+.2f*u(rng);
    for (float& x:initial) x=u(rng)*.1f;
    constexpr float scale = .7f;
    std::vector<double> reference(initial.begin(), initial.end());
    std::vector<float> expected(outputs);
    for (int t=0;t<rows;++t) for (int h=0;h<heads;++h) for (int j=0;j<dim;++j) {
        double* col=reference.data()+(h*dim+j)*dim;
        double decay=std::exp(double(g[t*heads+h])), sk=0, out=0;
        for (int i=0;i<dim;++i) sk+=col[i]*decay*k[t*qpitch+(h%keys)*dim+i];
        double delta=beta[t*heads+h]*(v[t*vpitch+h*dim+j]-sk);
        for (int i=0;i<dim;++i) {
            col[i]=col[i]*decay+k[t*qpitch+(h%keys)*dim+i]*delta;
            out+=col[i]*q[t*qpitch+(h%keys)*dim+i]*scale;
        }
        expected[(t*heads+h)*dim+j]=float(out);
    }
    Device dq(qcount), dk(qcount), dv(vcount), dg(g.size()), db(beta.size()), ds(states), next(states), out(outputs);
    // A second layout uses separate tensors; no Qwen projections, activation or normalization is involved.
    if (interleaved) {
        for (int t=0;t<rows;++t) {
            std::copy_n(k.data()+t*qpitch, keys*dim, q.data()+t*qpitch+keys*dim);
            std::copy_n(v.data()+t*vpitch, heads*dim, q.data()+t*qpitch+2*keys*dim);
        }
    }
    dq.put(q); dk.put(k); dv.put(v); dg.put(g); db.put(beta); ds.put(initial);
    next.put(std::vector<float>(states, -9)); out.put(std::vector<float>(outputs, -7));
    GdnMmaArgs a{};
    a.q=dq.p; a.k=interleaved ? dq.p+keys*dim : dk.p;
    a.v=interleaved ? dq.p+2*keys*dim : dv.p;
    a.log_decay=dg.p; a.beta=db.p; a.state=ds.p;
    a.state_out=separate_state ? next.p : ds.p; a.output=out.p;
    a.dim=dim; a.heads=heads; a.key_heads=keys; a.rows=rows;
    a.qk_row_stride=qpitch; a.v_row_stride=vpitch; a.gate_row_stride=heads; a.query_scale=scale;
    require(gdn_mma_supported(a), "prepared input contract");
    auto reject = [&](GdnMmaArgs bad) {
        require(!gdn_mma_supported(bad), "unsupported descriptor accepted");
        require(!gdn_mma_try(bad, nullptr), "unsupported call executed");
        require(out.get(outputs)==std::vector<float>(outputs,-7), "rejection changed output");
        require(ds.get(states)==initial, "rejection changed input state");
        require(next.get(states)==std::vector<float>(states,-9), "rejection changed output state");
        require(dq.get(qcount)==q && dg.get(g.size())==g && db.get(beta.size())==beta,
                "rejection changed prepared inputs");
    };
    auto bad=a; bad.rows=63; reject(bad);
    bad=a; bad.dim=64; reject(bad);
    bad=a; bad.key_heads=8; reject(bad);
    bad=a; bad.heads=24; reject(bad);
    bad=a; bad.q=nullptr; reject(bad);
    bad=a; bad.qk_row_stride=keys*dim-1; reject(bad);
    bad=a; bad.v_row_stride=-1; reject(bad);
    bad=a; bad.gate_row_stride=heads-1; reject(bad);
    bad=a; bad.qk_row_stride=std::numeric_limits<int64_t>::max(); reject(bad);
    bad=a; bad.query_scale=std::numeric_limits<float>::infinity(); reject(bad);
    require(gdn_mma_try(a, nullptr), "eligible independent consumer did not execute");
    ck(cudaDeviceSynchronize());
    compare(out.get(outputs), expected);
    compare((separate_state ? next : ds).get(states), {reference.begin(),reference.end()});
    if (separate_state) require(ds.get(states)==initial, "out-of-place execution modified input state");
    require(dq.get(qcount)==q && dk.get(qcount)==k && dv.get(vcount)==v &&
            dg.get(g.size())==g && db.get(beta.size())==beta, "execution modified prepared inputs");
    std::printf("PASS heads=%d rows=%d interleaved=%d separate_state=%d\n",heads,rows,interleaved,separate_state);
}
}
int main() {
    try {
        int devices=0;
        if (cudaGetDeviceCount(&devices)!=cudaSuccess || !devices || !gdn_mma_available(128,32,16)) return 77;
        for (int heads : {16,32,48,64}) for (bool interleaved : {false,true})
            run(heads, interleaved ? 65 : 64, interleaved, !interleaved);
        std::puts("PASS shared GDN: independent consumer, layouts, scale, state aliasing and rejection without mutation");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
