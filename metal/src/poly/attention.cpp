#include "strata/poly/kernels.hpp"
#include "launch.hpp"
#include <algorithm>
namespace strata::poly {
namespace {
struct Args {
    const void* p[10]{};
    int64_t pos=0;
    int dim=0,heads=0,kv=0,rows=0,window=0,ring=0,whole=0,stride=0,rot=0,sections=1,kind=0;
    float scale=0,eps=0;
};
static_assert(sizeof(Args)==144);
void run(const char* name,const Args& a,void* stream,gpu::metal::Grid groups,uint32_t threads=32) {
    static const auto source=metal_detail::read_source(STRATA_POLY_ATTENTION_SOURCE);
    metal_detail::dispatch("poly_attention",source,name,a,stream,groups,threads);
}
void store(const float* x,void* cache,void* scales,int n,int64_t pos,const long long* at,int rows,int ring,int kind,void* s) {
    if(n<=0 || rows<=0) return;
    if(kind && n%32) throw std::invalid_argument("Q8 KV rows must be a multiple of 32");
    Args a; a.p[0]=x;a.p[1]=cache;a.p[2]=scales;a.p[3]=at;a.dim=n;a.pos=pos;a.rows=rows;a.ring=ring;a.kind=kind;
    run("poly_store",a,s,{uint32_t((uint64_t(n)*rows+31)/32),1,1});
}
}
void head_norm_rope(const float* x,int stride,const float* w,float* y,int dim,int heads,int rot,float theta,
                    int64_t pos,int rows,float eps,void* s,const float* freq,const int32_t* axes,int sections,const long long* at) {
    if(dim<=0 || dim>1024 || rot<0 || rot>dim || rot%2 || sections<=0) throw std::invalid_argument("Metal head norm/rope shape");
    Args a; a.p[0]=x;a.p[1]=w;a.p[2]=y;a.p[3]=freq;a.p[4]=axes;a.p[5]=at;
    a.dim=dim;a.stride=stride;a.heads=heads;a.rot=rot;a.scale=theta;a.pos=pos;a.rows=rows;a.eps=eps;a.sections=sections;
    run("poly_rope",a,s,{uint32_t(rows*heads),1,1});
}
void store_f16(const float* x,uint16_t* cache,int n,int64_t pos,int rows,void* s) { store(x,cache,nullptr,n,pos,nullptr,rows,0,0,s); }
void store_f16_at(const float* x,uint16_t* cache,int n,const long long* at,int rows,int ring,void* s) { store(x,cache,nullptr,n,0,at,rows,ring,0,s); }
void store_q8(const float* x,int8_t* codes,uint16_t* scales,int n,int64_t pos,int rows,void* s) { store(x,codes,scales,n,pos,nullptr,rows,0,1,s); }
void store_q8_at(const float* x,int8_t* codes,uint16_t* scales,int n,const long long* at,int rows,int ring,void* s) { store(x,codes,scales,n,0,at,rows,ring,1,s); }
void attend(const float* q,const uint16_t* k,const uint16_t* v,int64_t pos,int rows,float*,int,float* out,
            int dim,int heads,int kv,float scale,void* s,int window,int ring,bool whole,const long long* at,int,size_t,Kvq8 kq,Kvq8 vq) {
    if(dim<=0 || dim>1024 || dim%4 || kv<=0 || heads%kv) throw std::invalid_argument("Metal attention shape");
    if(rows<=0) return;
    Args a;a.p[0]=q;a.p[1]=k;a.p[2]=v;a.p[3]=out;a.p[4]=at;
    a.p[5]=kq.codes;a.p[6]=kq.scales;a.p[7]=vq.codes;a.p[8]=vq.scales;
    a.dim=dim;a.heads=heads;a.kv=kv;a.pos=pos;a.rows=rows;a.scale=scale;a.window=window;a.ring=ring;a.whole=whole;
    run("poly_attention",a,s,{uint32_t(heads),uint32_t(rows),1});
}
size_t attend_split_floats(int,int,int,int) { return 0; } // online softmax has no score workspace
void attend_batch(void*,const float* q,const uint16_t* k,const uint16_t* v,int64_t pos,int rows,float* score,
                  size_t score_n,float*,float* out,int dim,int heads,int kv,float scale,void* s,int window,int ring,
                  bool whole,Kvq8 kq,Kvq8 vq,uint16_t*,uint16_t*,size_t) {
    attend(q,k,v,pos,rows,score,rows,out,dim,heads,kv,scale,s,window,ring,whole,nullptr,0,score_n,kq,vq);
}
}
