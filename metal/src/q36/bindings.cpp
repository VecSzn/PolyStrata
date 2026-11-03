// Native Metal operations for the separate strata-q36 host driver.
#include "strata/kernels/iq_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/native_router.hpp"
#include "strata/prefill/gemm.hpp"
#include "strata/prefill/moe_mmq.hpp"
#include "strata/poly/kernels.hpp"
#include "../poly/quantized.hpp"
#include "../poly/launch.hpp"
#include "mmq.hpp"
#include "ggml.h"
#include <stdexcept>
namespace strata::kernels {
using namespace poly::metal_detail;
bool native_q8_0_packed_enabled() { return false; }
bool native_q8_0_packed_eligible(int,int) { return false; }
void native_q8_0_pack_host(const void*,void*,int,int) { throw std::logic_error("CUDA Q8 packing is not used on Metal"); }
void native_q8_0_packed_register(const void*,const void*,int,int) { throw std::logic_error("CUDA Q8 packing is not used on Metal"); }
void native_q8_0_packed_unregister(const void*) {}
void native_router_topk(const float* x,int32_t* ids,float* w,int ne,int k,int rows,void* s) {
    poly::route_rows(x,ne,k,rows,0,1,nullptr,nullptr,true,1,ids,w,s);
}
void native_swiglu_quantize_q8_1(const float* g,const float* u,void* q,int n,int rows,void* s) {
    QArgs a;a.p[0]=g;a.p[1]=q;a.p[3]=u;a.ni=n;a.rows=rows;a.xstride=n;a.kind=1;
    qrun("poly_quantize",a,s,{uint32_t(n/32),uint32_t(rows),1});
}
bool native_expert_supported(int gu,int down,int64_t n,int64_t f) noexcept {
    return quant_name(gu)&&quant_name(down)&&n>0&&f>0&&n%ggml_blck_size(ggml_type(gu))==0&&f%ggml_blck_size(ggml_type(down))==0;
}
NativeExpertLayout native_expert_layout(int gu,int down,int64_t n,int64_t f) {
    if(!native_expert_supported(gu,down,n,f))throw std::invalid_argument("unsupported Metal expert layout");
    NativeExpertLayout l;l.gu_type=gu;l.d_type=down;l.n_embd=n;l.n_ff=f;
    l.gu_row=ggml_row_size(ggml_type(gu),n);l.d_row=ggml_row_size(ggml_type(down),f);
    l.up_off=l.gu_row*f;l.down_off=2*l.up_off;l.bytes=l.down_off+l.d_row*n;return l;
}
size_t native_expert_scratch_bytes(int64_t rows,int64_t f) {return size_t(rows*f)*12+native_q8_1_bytes(int(f),int(rows));}
void native_expert_grouped(const NativeExpertLayout& l,const unsigned long long* ptr,const int32_t* bounds,
    const int32_t* count,const int32_t* dst,const int32_t* tok,int64_t groups,int64_t rows,const void* x,
    void* scratch,float* out,void* s,int64_t) {
    float* gu=static_cast<float*>(scratch);float* h=gu+2*rows*l.n_ff;void* hq=h+rows*l.n_ff;
    QArgs a;a.p[0]=ptr;a.p[1]=x;a.p[2]=gu;a.p[3]=tok;a.p[4]=bounds;a.p[5]=count;
    a.ni=l.n_embd;a.no=2*l.n_ff;a.rows=rows;a.xstride=native_q8_1_bytes(a.ni);a.ystride=a.no;a.kind=4;
    qrun(std::string("poly_mul_")+quant_name(l.gu_type),a,s,{uint32_t(a.no),uint32_t(rows),uint32_t(groups)});
    poly::swiglu_clamp_rows(gu,h,l.n_ff,rows,0,s);
    native_quantize_q8_1(h,hq,l.n_ff,rows,s);
    a.p[1]=hq;a.p[2]=out;a.p[3]=dst;a.ni=l.n_ff;a.no=l.n_embd;a.xstride=native_q8_1_bytes(a.ni);
    a.ystride=a.no;a.offset=l.down_off;a.kind=5;
    qrun(std::string("poly_mul_")+quant_name(l.d_type),a,s,{uint32_t(a.no),uint32_t(rows),uint32_t(groups)});
}
void iq_embed_rows(int type,const void* table,size_t stride,const int32_t* tokens,int64_t rows,int64_t n,float* out,void* s) {
    const char* name=type==0?"f32":type==1?"f16":type==30?"bf16":quant_name(type);
    if(!name)throw std::invalid_argument("unsupported Metal embedding type");
    QArgs a;a.p[0]=table;a.p[2]=out;a.p[3]=tokens;a.stride=stride;a.ni=n;a.rows=rows;
    qrun(std::string("q36_embed_")+name,a,s,{uint32_t((n+(type==30?31:511))/(type==30?32:512)),uint32_t(rows),1});
}
}
namespace strata::prefill {
Gemm::~Gemm()=default;
bool Gemm::init(void* s,int64_t n,std::string&) {stream_=s;scratch_elems_=n;return true;}
void Gemm::native(const uint16_t* x,int t,const void* w,float* y,int64_t rows,int64_t no,int64_t ni,int64_t ldy,float beta,int64_t ldx) {
    using namespace poly::metal_detail;
    if(beta!=0)throw std::invalid_argument("q36 Metal GEMM beta must be zero");
    kernels::native_mmvq_weight_bytes(t,ni,no);
    QArgs a;a.p[0]=w;a.p[1]=x;a.p[2]=y;a.ni=ni;a.no=no;a.rows=rows;a.xstride=(ldx?ldx:ni)*2;a.ystride=ldy?ldy:no;a.kind=3;
    qrun(std::string("poly_mul_")+quant_name(t),a,stream_,{uint32_t(no),uint32_t(rows),1});
}
namespace mmq {
bool built(){return true;}
bool supported(int t){return kernels::native_mmvq_supported(t);}
bool fits(int t,int64_t rows){return supported(t)&&rows>0;}
size_t q8_bytes(int64_t rows,int64_t cols){return poly::Mmq::q8_bytes(rows,cols);}
void quantize(const float* x,const int32_t* ids,void* q,int t,int64_t cols,int64_t ld,int64_t rows,void* s,float* scale){
    if(scale)throw std::invalid_argument("q36 does not use NVFP4 activation scales");
    poly::Mmq::quantize(x,ids,q,t,cols,ld,rows,s);
}
Context::Context(){ctx_=new poly::Mmq;}
Context::~Context(){delete static_cast<poly::Mmq*>(ctx_);}
void Context::run(const Product& p,void* s){
    if(p.tab||p.y_scale)throw std::invalid_argument("unsupported q36 product pointer table / scales");
    static_cast<poly::Mmq*>(ctx_)->mul(p.w,p.type,p.w_rows,p.w_cols,p.expert_bytes,p.n,p.xq,p.bounds,p.ids,p.total_rows,p.max_rows,p.dst,p.ld_dst,s);
}
void swiglu(const float* gu,float* h,int64_t rows,int64_t f,bool interleaved,void* s){
    if(interleaved)throw std::invalid_argument("q36 GGUF experts use split gate/up rows");
    poly::swiglu_clamp_rows(gu,h,f,rows,0,s);
}
void iota(int32_t* dst,int64_t n,void* s){
    struct Args{int32_t* dst;int64_t n;}a{dst,n};
    static const std::string source=R"(#include <metal_stdlib>
using namespace metal;struct Args{device int* dst;long n;};
kernel void q36_iota(constant Args& a [[buffer(0)]],uint i [[thread_position_in_grid]]){if(i<a.n)a.dst[i]=i;})";
    poly::metal_detail::dispatch("q36_iota",source,"q36_iota",a,s,{uint32_t((n+255)/256),1,1},256);
}
}
}
