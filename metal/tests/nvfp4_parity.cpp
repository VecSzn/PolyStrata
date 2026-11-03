// Ready-pack NVFP4 expert blocks and independent projection scales, against ggml's decoder.
#include "mtl_rt.h"
#include "ggml.h"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

struct Args {
    uint64_t gate_off, up_off, down_off, gu_stride, d_stride;
    uint32_t gu_row, d_row, n_embd, n_ff, n_tok, topk, n_expert, gx_gu, gx_dn, pad;
};
int main() {
    constexpr int N=256, F=128, E=16, K=10, P=16+128+128+4+128+128+128;
    constexpr int gs=N*F/64*36, ds=F*N/64*36, blob=2*gs+ds+16;
    constexpr int slot=(blob+16383)/16384*16384;
    mtl::Context cx;
    int lib=cx.compile({"quant.metal","units.metal","moe.metal"});
    auto plan=cx.pipeline(lib,"moe_plan"),gu=cx.pipeline(lib,"moe_gu_40_32_1_4",mtl::Constants().b(20,true)),
         dn=cx.pipeline(lib,"moe_down_40_4_2_4",mtl::Constants().b(20,true));
    auto w=cx.alloc(E*slot),map=cx.alloc(E*4),x=cx.alloc(8*N*4),h=cx.alloc(8*K*F*4),
         y=cx.alloc(8*K*N*4),ids=cx.alloc(8*K*4),weights=cx.alloc(8*K*4),p=cx.alloc(P*4);
    for(int e=0;e<E;++e) {
        map.as<uint32_t>()[e]=e*slot/16384;
        auto* data=w.as<uint8_t>()+e*slot;
        for(int b=0;b<(2*gs+ds)/36;++b) {
            for(int j=0;j<4;++j)data[36*b+j]=uint8_t(8+(b+j+e)%24);
            for(int j=4;j<36;++j)data[36*b+j]=uint8_t(b*13+j*37+e*17);
        }
        float scales[]={.25f+.05f*e,1.7f-.05f*e,.003f+.001f*e,0};
        std::memcpy(data+blob-16,scales,16);
    }
    const auto dequant=ggml_get_type_traits(GGML_TYPE_NVFP4)->to_float;
    double worst=0;
    for(int nt:{1,4,8}) {
        for(int i=0;i<nt*N;++i)x.as<float>()[i]=std::sin(float(i)*.07f);
        for(int t=0;t<nt;++t)for(int k=0;k<K;++k){ids.as<int>()[t*K+k]=(k+t*3)%E;weights.as<float>()[t*K+k]=.1f;}
        Args a{0,gs,2*gs,gs,ds,N/64*36,F/64*36,N,F,uint32_t(nt),K,E,F/4,N/64,16};
        mtl::Dispatch rp;rp.p=&plan;rp.args={mtl::Arg::B(ids),mtl::Arg::B(weights),mtl::Arg::B(p),mtl::Arg::V(a)};rp.tg[0]=512;
        mtl::Dispatch g;g.p=&gu;g.args={mtl::Arg::B(w),mtl::Arg::B(x),mtl::Arg::B(h),mtl::Arg::B(p),mtl::Arg::V(a),mtl::Arg::B(x),mtl::Arg::B(map),mtl::Arg::B(w),mtl::Arg::B(w),mtl::Arg::B(w)};g.tg[0]=128;g.indirect=&p;g.ind_off=16;
        auto d=g;d.p=&dn;d.args[1]=mtl::Arg::B(h);d.args[2]=mtl::Arg::B(y);d.ind_off=32;
        cx.run({rp,g,d});
        std::vector<float> gate(F*N),up(F*N),down(N*F),hidden(F);
        double peak=0,error=0;
        for(int t=0;t<nt;++t)for(int k=0;k<K;++k) {
            int e=ids.as<int>()[t*K+k];auto* data=w.as<uint8_t>()+e*slot;
            dequant(data,gate.data(),F*N);dequant(data+gs,up.data(),F*N);dequant(data+2*gs,down.data(),N*F);
            float scales[4];std::memcpy(scales,data+blob-16,16);
            int entry=p.as<int>()[16+128+128+4+128+128+t*K+k];
            for(int r=0;r<F;++r){double g0=0,u0=0;for(int j=0;j<N;++j){g0+=double(gate[r*N+j])*x.as<float>()[t*N+j];u0+=double(up[r*N+j])*x.as<float>()[t*N+j];}
                g0*=scales[0];u0*=scales[1];hidden[r]=float(g0/(1+std::exp(-g0))*u0);
                error=std::max(error,std::abs(double(h.as<float>()[entry*F+r])-hidden[r]));peak=std::max(peak,std::abs(double(hidden[r])));}
            for(int r=0;r<N;++r){double sum=0;for(int j=0;j<F;++j)sum+=double(down[r*F+j])*hidden[j];sum*=scales[2]*.1f;
                error=std::max(error,std::abs(double(y.as<float>()[entry*N+r])-sum));peak=std::max(peak,std::abs(sum));}
        }
        double relative=error/std::max(peak,1e-20);worst=std::max(worst,relative);
        std::printf("NVFP4 %d tokens, independent gate/up/down scales: max relative error %.3g\n",nt,relative);
    }
    return worst<2e-5?0:1;
}
