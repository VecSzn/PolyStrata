// Native Metal operations for GLM and DeepSeek. The shared host engines own all caches and scheduling.
#include "strata/poly/kernels.hpp"
#include "launch.hpp"
#include <algorithm>
#include <cmath>
namespace strata::poly {
namespace {
struct Args { const void* p[12]{}; int64_t pos=0; uint64_t stride=0; int n[16]{}; float f[8]{}; };
static_assert(sizeof(Args)==208);
void run(const char* name,const Args& a,void* stream,int x,int y=1,int threads=256) {
    if(x<=0 || y<=0) return;
    static const auto source=metal_detail::read_source(STRATA_POLY_ADVANCED_SOURCE);
    metal_detail::dispatch("poly_advanced",source,name,a,stream,{uint32_t(x),uint32_t(y),1},threads);
}
int grid(int n){return (n+255)/256;}
void rope_args(Args& a,const Rope& r){a.f[0]=r.theta_scale;a.f[1]=r.freq_scale;a.f[2]=r.ext_factor;a.f[3]=r.attn_factor;a.f[4]=r.corr0;a.f[5]=r.corr1;}
}
void hc_pre(const float* r,const float* fn,const float* scale,const float* base,float* mix,float* hcw,float* cur,
            int n,int rows,float eps,float heps,int iters,void* s) {
    Args a;a.p[0]=r;a.p[1]=fn;a.p[2]=scale;a.p[3]=base;a.p[4]=mix;a.p[5]=hcw;a.p[6]=cur;
    a.n[0]=n;a.n[1]=24;a.n[2]=iters;a.f[0]=eps;a.f[1]=heps;
    run("poly_hc_mix",a,s,25,rows,32);run("poly_hc_weights",a,s,rows,1,32);run("poly_hc_read",a,s,grid(n),rows);
}
void hc_post(const float* x,const float* w,float* r,int n,int rows,void* s){Args a;a.p[0]=x;a.p[1]=w;a.p[2]=r;a.n[0]=n;run("poly_hc_post",a,s,grid(n),rows);}
void hc_mean(const float* r,float* y,int n,int rows,void* s){Args a;a.p[0]=r;a.p[1]=y;a.n[0]=n;run("poly_hc_mean",a,s,grid(n),rows);}
void hc_head(const float* r,const float* fn,const float* scale,const float* base,float* mix,float* y,int n,int rows,float eps,float heps,void* s){
    Args a;a.p[0]=r;a.p[1]=fn;a.p[2]=scale;a.p[3]=base;a.p[4]=mix;a.p[6]=y;a.n[0]=n;a.n[1]=4;a.f[0]=eps;a.f[1]=heps;
    run("poly_hc_mix",a,s,5,rows,32);run("poly_hc_read",a,s,grid(n),rows);
}
void kda_conv(const float* x,float* state,const float* wq,const float* wk,const float* wv,float* out,int n,int rows,void* s,float* snap,size_t stride,int snaps){
    Args a;a.p[0]=x;a.p[1]=state;a.p[2]=wq;a.p[3]=wk;a.p[4]=wv;a.p[5]=out;a.p[6]=snap;a.stride=stride;a.n[0]=n;a.n[1]=rows;a.n[2]=snap?snaps:0;
    run("poly_kda_conv",a,s,grid(3*n));
}
void kda_step(float* qkv,float* g1,const float* decay,const float* beta,float* state,const float* g2,const float* norm,float* out,int dim,int heads,int rows,float lower,float eps,void* s,float* snap,size_t stride,int snaps){
    if(dim<32 || dim>256 || dim%32) throw std::invalid_argument("Metal KDA head dimension must be 32..256 in 32s");
    Args a;a.p[0]=qkv;a.p[1]=g1;a.p[2]=decay;a.p[3]=beta;a.p[4]=state;a.p[5]=g2;a.p[6]=norm;a.p[7]=out;a.p[8]=snap;
    a.stride=stride;a.n[0]=dim;a.n[1]=heads;a.n[2]=rows;a.n[3]=snap?snaps:0;a.f[0]=lower;a.f[1]=eps;
    run("poly_kda_step",a,s,heads,1,dim);
}
void idx_store(const float* key,const float* gate,float* ring,int ring_rows,int dim,int64_t pos,int rows,void* s){
    Args a;a.p[0]=key;a.p[1]=gate;a.p[2]=ring;a.pos=pos;a.n[0]=dim;a.n[1]=ring_rows;run("poly_idx_store",a,s,grid(dim),rows);
}
void idx_pool(const float* ring,int ring_rows,float* pool,const float* ape,int dim,int kpool,int64_t pos,int rows,void* s){
    Args a;a.p[0]=ring;a.p[1]=pool;a.p[2]=ape;a.pos=pos;a.n[0]=dim;a.n[1]=ring_rows;a.n[2]=kpool;run("poly_idx_pool",a,s,grid(dim),rows);
}
void idx_score(const float* q,const float* w,const float* pool,float* score,int stride,int dim,int heads,int kpool,int64_t pos,int rows,void* s){
    if(dim<1 || dim>256 || heads<=0) throw std::invalid_argument("Metal GLM indexer shape");
    Args a;a.p[0]=q;a.p[1]=w;a.p[2]=pool;a.p[3]=score;a.pos=pos;a.n[0]=dim;a.n[1]=heads;a.n[2]=kpool;a.n[3]=stride;
    a.f[0]=1.f/std::sqrt(float(dim*heads));run("poly_idx_score",a,s,int((pos+rows)/kpool),rows,32);
}
void idx_score_batch(void*,const float* q,const float* w,const float* pool,float* score,int stride,int dim,int heads,int kpool,int64_t pos,int rows,float*,size_t,void* s){idx_score(q,w,pool,score,stride,dim,heads,kpool,pos,rows,s);}
void idx_select(const float* score,int score_stride,int32_t* sel,int sel_stride,int top,int kpool,int64_t pos,int rows,void* s){
    Args a;a.p[0]=score;a.p[1]=sel;a.pos=pos;a.n[0]=score_stride;a.n[1]=sel_stride;a.n[2]=top;a.n[3]=kpool;run("poly_select",a,s,rows,1,32);
}
void dsa_attend(void*,const float* q,const uint16_t* cache,const int32_t* sel,int sel_stride,int top,int kpool,int64_t pos,int rows,int dense,float*,float*,int,float* out,int dim,int heads,float scale,void* s){
    if(dim<1 || dim>2048) throw std::invalid_argument("Metal latent attention dimension");
    Args a;a.p[0]=q;a.p[1]=cache;a.p[2]=sel;a.p[3]=out;a.pos=pos;a.n[0]=dim;a.n[1]=heads;a.n[2]=sel_stride;a.n[3]=top;a.n[4]=kpool;a.n[5]=dense;a.f[0]=scale;
    run("poly_dsa_attend",a,s,heads,rows,32);
}
void dsv4_rope(const float* x,float* y,int dim,int heads,int off,int n_rot,const Rope& r,int64_t pos,int step,int rows,bool norm,float eps,bool back,void* s,const long long* at){
    if(dim<2 || dim>2048 || dim%2) throw std::invalid_argument("Metal DeepSeek rotary shape");
    Args a;a.p[0]=x;a.p[1]=y;a.p[2]=at;a.pos=pos;a.n[0]=dim;a.n[1]=heads;a.n[2]=off;a.n[3]=n_rot;a.n[4]=step;a.n[5]=norm;a.n[6]=back;rope_args(a,r);a.f[6]=eps;run("poly_dsv4_rope",a,s,heads,rows,256);
}
void dsv4_state(const float* kv,const float* score,const float* ape,float* ring_kv,float* ring_score,int ring,int width,int ratio,int64_t pos,int rows,void* s,const long long* at){
    Args a;a.p[0]=kv;a.p[1]=score;a.p[2]=ape;a.p[3]=ring_kv;a.p[4]=ring_score;a.p[5]=at;a.pos=pos;a.n[0]=width;a.n[1]=ring;a.n[2]=ratio;run("poly_dsv4_state",a,s,grid(width),rows);
}
void dsv4_squeeze(const float* kv,const float* score,int ring,int dim,int ratio,bool overlap,const float* norm,float eps,const Rope& r,int n_rot,int64_t pos,int rows,uint16_t* cache,void* s,const long long* at){
    if(dim<32 || dim>1024 || dim%32) throw std::invalid_argument("Metal DeepSeek compressor shape");
    Args a;a.p[0]=kv;a.p[1]=score;a.p[2]=norm;a.p[3]=cache;a.p[4]=at;a.pos=pos;a.n[0]=dim;a.n[1]=ring;a.n[2]=ratio;a.n[3]=overlap;a.n[4]=n_rot;rope_args(a,r);a.f[6]=eps;run("poly_dsv4_squeeze",a,s,rows,1,256);
}
void dsv4_idx_score(const float* q,const float* w,const uint16_t* keys,float* score,int stride,int dim,int heads,int ratio,int64_t pos,int rows,float scale,void* s,const long long* at,int room){
    if(dim!=128 || heads<1) throw std::invalid_argument("Metal DeepSeek indexer shape");
    Args a;a.p[0]=q;a.p[1]=w;a.p[2]=keys;a.p[3]=score;a.p[4]=at;a.pos=pos;a.n[0]=dim;a.n[1]=heads;a.n[2]=ratio;a.n[3]=stride;a.n[4]=1;a.f[0]=scale;run("poly_idx_score",a,s,int((at?room:pos+rows)/ratio),rows,32);
}
void dsv4_select(const float* score,int stride,int32_t* sel,int sel_stride,int top,int ratio,int64_t pos,int rows,void* s,const long long* at){
    Args a;a.p[0]=score;a.p[1]=sel;a.p[2]=at;a.pos=pos;a.n[0]=stride;a.n[1]=sel_stride;a.n[2]=top;a.n[3]=ratio;a.n[4]=1;run("poly_select",a,s,rows,1,32);
}
void dsv4_attend(const float* q,const uint16_t* raw,int ring,int window,const uint16_t* comp,int ratio,const int32_t* sel,int sel_stride,int top,const float* sinks,float* out,int dim,int heads,float scale,int64_t pos,int rows,void* s,const long long* at,int){
    if(dim<1 || dim>2048) throw std::invalid_argument("Metal DeepSeek attention dimension");
    Args a;a.p[0]=q;a.p[1]=raw;a.p[2]=comp;a.p[3]=sel;a.p[4]=sinks;a.p[5]=out;a.p[6]=at;a.pos=pos;a.n[0]=dim;a.n[1]=heads;a.n[2]=ring;a.n[3]=window;a.n[4]=ratio;a.n[5]=sel_stride;a.n[6]=top;a.f[0]=scale;run("poly_dsv4_attend",a,s,heads,rows,32);
}
void permute_groups(const float* x,float* y,int n,int groups,int rows,bool back,void* s){Args a;a.p[0]=x;a.p[1]=y;a.n[0]=n;a.n[1]=groups;a.n[2]=rows;a.n[3]=back;run("poly_permute",a,s,grid(n*groups*rows));}
}
