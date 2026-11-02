// Opt-in GDN: independent double recurrence, final state, continuation and snapshot rollback.
#include "strata/poly/kernels.hpp"
#ifdef STRATA_POLY_GDN_MMA
#include "gdn_mma.hpp"
#include "../kernels/cuda/gdn_mma.cuh"
using namespace strata::kernels;
#endif
#include <cuda_runtime.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <random>
#include <stdexcept>
#include <vector>
using namespace strata::poly;
namespace {
constexpr int D=128, H=32, K=16, Q=(2*K+H)*D, O=H*D, S=H*D*D;
constexpr float eps=1e-6f;
void ck(cudaError_t e) { if(e!=cudaSuccess) throw std::runtime_error(cudaGetErrorString(e)); }
struct Device {
    float* p=nullptr;
    explicit Device(size_t n) { ck(cudaMalloc(&p,n*sizeof(float))); }
    ~Device() { cudaFree(p); }
    void put(const std::vector<float>& v) { ck(cudaMemcpy(p,v.data(),v.size()*4,cudaMemcpyHostToDevice)); }
    std::vector<float> get(size_t n) { std::vector<float> v(n); ck(cudaMemcpy(v.data(),p,n*4,cudaMemcpyDeviceToHost)); return v; }
};
struct Data {
    int n; std::vector<float> q,a,b,z,dt,dec,w,state;
    Data(int rows, bool nonzero, bool extreme): n(rows),q(n*Q),a(n*H),b(n*H),z(n*O),dt(H),dec(H),w(D),state(S) {
        std::mt19937 rng(87123); std::uniform_real_distribution<float> u(-.7f,.7f);
        for(auto* v:{&q,&a,&b,&z,&dt,&w}) for(float& x:*v) x=u(rng);
        for(float& x:w) x+=1;
        for(float& x:dec) x=-.1f-std::abs(u(rng));
        if(nonzero) for(float& x:state) x=u(rng)*.2f;
        if(extreme) for(int t=0;t<n;++t) for(int h=0;h<H;++h) {
            a[t*H+h]=(t%5==0 ? 1000.f : (t%5==1 ? -80.f : .1f));
            b[t*H+h]=h%2 ? 30.f : -30.f;
        }
    }
};
struct Result { std::vector<float> out,state,snaps; bool used; };
Result run(const Data& d,bool enabled,int start,int n,const std::vector<float>& initial,unsigned snaps=0) {
    Device q(n*Q),a(n*H),b(n*H),z(n*O),dt(H),dec(H),w(D),s(S),out(n*O),snap(3*S);
    q.put({d.q.begin()+start*Q,d.q.begin()+(start+n)*Q});
    a.put({d.a.begin()+start*H,d.a.begin()+(start+n)*H});
    b.put({d.b.begin()+start*H,d.b.begin()+(start+n)*H});
    z.put({d.z.begin()+start*O,d.z.begin()+(start+n)*O});
    dt.put(d.dt); dec.put(d.dec); w.put(d.w); s.put(initial);
    bool used=false;
#ifdef STRATA_POLY_GDN_MMA
    used=enabled && gdn_mma_step(q.p,a.p,b.p,dt.p,dec.p,s.p,z.p,w.p,out.p,D,H,K,n,eps,nullptr,snaps!=0);
#endif
    if(!used) gdn_step(q.p,a.p,b.p,dt.p,dec.p,s.p,z.p,w.p,out.p,D,H,K,n,eps,nullptr,
                       snaps?snap.p:nullptr,S,snaps);
    ck(cudaGetLastError()); ck(cudaDeviceSynchronize());
    return {out.get(n*O),s.get(S),snaps?snap.get(3*S):std::vector<float>{},used};
}
void compare(const char* label,const std::vector<float>& x,const std::vector<float>& y,double absolute,bool exact=false) {
    if(x.size()!=y.size()) throw std::runtime_error("size");
    double err=0,base=0,mx=0;
    for(size_t i=0;i<x.size();++i) {
        if(!std::isfinite(x[i])||!std::isfinite(y[i])) throw std::runtime_error("nonfinite");
        double e=double(x[i])-y[i]; err+=e*e; base+=double(y[i])*y[i]; mx=std::max(mx,std::abs(e));
    }
    double rel=std::sqrt(err/std::max(base,1e-30));
    std::printf("%s L2=%.9g max=%.9g\n",label,rel,mx);
    if((exact && mx!=0)||rel>5e-4||mx>absolute) throw std::runtime_error(label);
}
Result reference(const Data& d) {
    std::vector<double> state(d.state.begin(),d.state.end());
    std::vector<float> out(d.n*O);
    double q[K][D],k[K][D];
    for(int t=0;t<d.n;++t) {
        for(int h=0;h<K;++h) {
            double qs=eps,ks=eps;
            for(int i=0;i<D;++i) { double a=d.q[t*Q+h*D+i],b=d.q[t*Q+K*D+h*D+i]; qs+=a*a; ks+=b*b; }
            for(int i=0;i<D;++i) { q[h][i]=d.q[t*Q+h*D+i]/std::sqrt(qs*D); k[h][i]=d.q[t*Q+K*D+h*D+i]/std::sqrt(ks); }
        }
        for(int h=0;h<H;++h) {
            double gate=double(d.a[t*H+h])+d.dt[h];
            double dc=std::exp((gate>20 ? gate : std::log1p(std::exp(gate)))*d.dec[h]);
            double beta=1/(1+std::exp(-double(d.b[t*H+h]))), sum=eps;
            double row[D];
            for(int j=0;j<D;++j) {
                double* col=state.data()+(h*D+j)*D; double sk=0,o=0;
                for(int i=0;i<D;++i) sk+=col[i]*dc*k[h%K][i];
                double delta=(d.q[t*Q+2*K*D+h*D+j]-sk)*beta;
                for(int i=0;i<D;++i) { col[i]=col[i]*dc+k[h%K][i]*delta; o+=col[i]*q[h%K][i]; }
                row[j]=o; sum+=o*o/D;
            }
            for(int j=0;j<D;++j) { double z=d.z[t*O+h*D+j]; out[t*O+h*D+j]=row[j]/std::sqrt(sum)*d.w[j]*(z/(1+std::exp(-z))); }
        }
    }
    return {out,{state.begin(),state.end()},{},false};
}
void parity(int n,bool nonzero,bool extreme) {
    std::printf("CASE n=%d state=%d extreme=%d\n",n,nonzero,extreme); std::fflush(stdout);
    Data d(n,nonzero,extreme); auto stock=run(d,false,0,n,d.state),fast=run(d,true,0,n,d.state);
    if(stock.used || fast.used!=(n>=64)) throw std::runtime_error("dispatch");
    compare("output stock",fast.out,stock.out,.003,n<64);
    compare("state stock",fast.state,stock.state,.001,n<64);
    auto ref=reference(d);
    compare("stock/reference output",stock.out,ref.out,.003);
    compare("stock/reference state",stock.state,ref.state,.001);
    compare("MMA/reference output",fast.out,ref.out,.003);
    compare("MMA/reference state",fast.state,ref.state,.001);
    if(n>=65) {
        int at=n>2048?2048:64;
        auto first=run(d,true,0,at,d.state);
        auto second=run(d,true,at,n-at,first.state); // also checkpoint restore into a fresh state allocation
        auto combined=first.out; combined.insert(combined.end(),second.out.begin(),second.out.end());
        compare("split/checkpoint output",combined,stock.out,.003);
        compare("split/checkpoint state",second.state,stock.state,.001);
    }
}
#ifdef STRATA_POLY_GDN_MMA
// Exercise every upstream CUDA schedule on this device, even when its SM count selects only one.
// The raw kernel sees already normalized Q/K, log-decay and activated beta; check an independent recurrence.
void schedules(int heads, int n, bool nonzero) {
    const int width=(2*K+heads)*D, outputs=heads*D, states=heads*D*D;
    std::mt19937 rng(417+heads+n);std::uniform_real_distribution<float> u(-1.f,1.f);
    std::vector<float> qkv(n*width),g(n*heads),beta(n*heads),initial(states);
    for(int t=0;t<n;++t) {
        for(int i=0;i<K*D;++i){qkv[t*width+i]=u(rng)*.01f;qkv[t*width+K*D+i]=u(rng)*.1f;}
        for(int i=0;i<outputs;++i)qkv[t*width+2*K*D+i]=u(rng)*.2f;
    }
    for(float& x:g)x=-.05f-.01f*std::abs(u(rng));
    for(float& x:beta)x=.5f+.2f*u(rng);
    if(nonzero)for(float& x:initial)x=u(rng)*.1f;
    std::vector<double> state(initial.begin(),initial.end());std::vector<float> reference(n*outputs);
    for(int t=0;t<n;++t)for(int h=0;h<heads;++h)for(int v=0;v<D;++v) {
        const float* key=qkv.data()+t*width+K*D+(h%K)*D;
        const float* query=qkv.data()+t*width+(h%K)*D;
        double* column=state.data()+(h*D+v)*D;
        const double decay=std::exp(double(g[t*heads+h]));double sk=0;
        for(int i=0;i<D;++i)sk+=column[i]*decay*key[i];
        const double delta=beta[t*heads+h]*(qkv[t*width+2*K*D+h*D+v]-sk);double out=0;
        for(int i=0;i<D;++i){column[i]=column[i]*decay+key[i]*delta;out+=column[i]*query[i];}
        reference[t*outputs+h*D+v]=float(out);
    }
    const std::vector<float> expected_state(state.begin(),state.end());
    Device dq(qkv.size()),dg(g.size()),db(beta.size()),ds(states),out(n*outputs);
    dq.put(qkv);dg.put(g);db.put(beta);
    ggml_cuda_gdn_mma_args a{};a.q=dq.p;a.k=dq.p+K*D;a.v=dq.p+2*K*D;
    a.g=dg.p;a.beta=db.p;a.state=a.state_out=ds.p;a.dst=out.p;
    a.H=heads;a.H_k=K;a.n_tokens=n;a.n_seqs=a.rq3=1;
    a.sq1=a.sv1=D;a.sq2=a.sv2=width;a.sb1=1;a.sb2=heads;a.scale=1.f;a.eligible=true;
    int device=0;ck(cudaGetDevice(&device));
    for(auto path:{GdnMmaPath::whole_head,GdnMmaPath::slice32_one_block,GdnMmaPath::slice32_two_blocks}) {
        std::printf("SCHEDULE heads=%d rows=%d state=%d path=%d\n",heads,n,nonzero,int(path));
        if(!gdn_mma_prepare(device,path))throw std::runtime_error("schedule unavailable on an eligible device");
        ds.put(initial);gdn_mma_launch(a,path,nullptr);ck(cudaDeviceSynchronize());
        compare("raw schedule/double output",out.get(n*outputs),reference,.003);
        compare("raw schedule/double state",ds.get(states),expected_state,.001);
    }
}
#endif

}
int main() {
    try {
#ifdef STRATA_POLY_GDN_MMA
        int devices=0;
        if(cudaGetDeviceCount(&devices)!=cudaSuccess || devices==0) {
            std::puts("SKIP GDN MMA: no CUDA device");return 77;
        }
        if(!gdn_mma_available(D,H,K)) {
            parity(1,true,false);parity(63,true,false);
            std::puts("SKIP GDN MMA fast path: unsupported device; stock short-call checks passed");return 77;
        }
#endif
#ifdef STRATA_POLY_GDN_MMA
        for(int heads:{16,32,48,64})schedules(heads,65,true);
        schedules(32,97,false);
#endif
        for(int n:{1,63,64,65,97,2048,2049}) parity(n,true,false);
        parity(65,false,false); parity(65,true,true);
        Data d(101,true,false);
        auto prefix=run(d,true,0,97,d.state);
        Data guessed=d;
        for(int t=98;t<101;++t) {
            for(int i=0;i<Q;++i) guessed.q[t*Q+i]*=-2;
            for(int h=0;h<H;++h) guessed.a[t*H+h]+=2;
        }
        auto wrong=run(guessed,true,97,4,prefix.state,7),stock=run(guessed,false,97,4,prefix.state,7);
        if(wrong.used) throw std::runtime_error("MTP snapshot dispatch");
        compare("MTP output",wrong.out,stock.out,0,true);
        compare("MTP snapshots",wrong.snaps,stock.snaps,0,true);
        for(int keep=1;keep<=3;++keep) {
            std::vector<float> restored(wrong.snaps.begin()+(keep-1)*S,wrong.snaps.begin()+keep*S);
            auto resumed=run(d,true,97+keep,4-keep,restored);
            Data expected=d;
            std::copy(guessed.q.begin()+97*Q,guessed.q.begin()+(97+keep)*Q,expected.q.begin()+97*Q);
            std::copy(guessed.a.begin()+97*H,guessed.a.begin()+(97+keep)*H,expected.a.begin()+97*H);
            auto straight=run(expected,false,97,4,prefix.state);
            compare("wrong-guess rollback state",resumed.state,straight.state,0,true);
        }
        std::puts("PASS GDN outputs/state/reference/threshold/continuation/checkpoint/MTP rollback");
        return 0;
    } catch(const std::exception& e) { std::fprintf(stderr,"FAIL %s\n",e.what()); return 1; }
}
