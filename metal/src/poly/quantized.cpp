// PolyStrata projections using the pinned ggml Metal block decoders. All dispatch
// and graph ownership stays in PolyStrata's Metal runtime.
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"
#include "launch.hpp"
#include "quantized.hpp"
#include "mmq.hpp"
#include "ggml.h"
#include <limits>
namespace strata::poly::metal_detail {
namespace {
std::string without(std::string s,const std::string& line) {
    auto p=s.find(line); if(p!=std::string::npos) s.erase(p,line.size());return s;
}
std::string quant_source() {
    const std::string root=STRATA_POLY_GGML_SOURCE;
    auto common=read_source((root+"/ggml-metal/kernels/common.h").c_str());
    common=without(common,"#include \"ggml-metal-impl.h\"");
    auto deq=read_source((root+"/ggml-metal/kernels/dequantize.h").c_str());
    deq=without(deq,"#include \"common.h\"");
    auto pos=deq.find("#include \"ggml-common.h\"");
    deq.replace(pos,std::string("#include \"ggml-common.h\"").size(),read_source((root+"/ggml-common.h").c_str()));
    return read_source((root+"/ggml-metal/ggml-metal-impl.h").c_str())+common+deq+read_source(STRATA_POLY_QUANT_SOURCE);
}
}
void qrun(const std::string& name,const QArgs& a,void* s,gpu::metal::Grid groups) {
    static const auto source=quant_source();
    const bool indirect = name.starts_with("poly_mul_") && (a.kind==1 || a.kind==4 || a.kind==5);
    dispatch("poly_quantized",source,name.c_str(),a,s,groups,32,{a.p,a.kind==8?size_t(3*a.count):6,true,indirect?a.p[0]:nullptr});
}
const char* quant_name(int t) {
    switch(t) {
#define Q(id,name) case id:return #name;
    Q(2,q4_0) Q(3,q4_1) Q(6,q5_0) Q(7,q5_1) Q(8,q8_0) Q(10,q2_K) Q(11,q3_K) Q(12,q4_K) Q(13,q5_K)
    Q(14,q6_K) Q(16,iq2_xxs) Q(17,iq2_xs) Q(18,iq3_xxs) Q(19,iq1_s) Q(20,iq4_nl) Q(21,iq3_s)
    Q(22,iq2_s) Q(23,iq4_xs) Q(29,iq1_m) Q(39,mxfp4) Q(42,q2_0)
#undef Q
    default:return nullptr;
    }
}
}
namespace strata::kernels {
using namespace poly::metal_detail;
bool native_mmvq_supported(int t) noexcept { return quant_name(t); }
bool native_mmvq_ids_supported(int t) { return native_mmvq_supported(t); }
size_t native_q8_1_bytes(int n,int rows) {
    if(n<=0 || n%32 || rows<=0) throw std::invalid_argument("Q8_1 shape");
    return size_t(n/32)*36*rows;
}
size_t native_mmvq_weight_bytes(int t,int ni,int no) {
    if(!native_mmvq_supported(t) || ni<=0 || no<=0 || ni%ggml_blck_size(ggml_type(t))) throw std::invalid_argument("Metal quantized matrix shape");
    return ggml_row_size(ggml_type(t),ni)*no;
}
void native_quantize_q8_1(const float* x,void* q,int n,int rows,void* s) {
    native_q8_1_bytes(n,rows);
    QArgs a;a.p[0]=x;a.p[1]=q;a.ni=n;a.rows=rows;a.xstride=n;
    qrun("poly_quantize",a,s,{uint32_t(n/32),uint32_t(rows),1});
}
void native_add_rms_norm_q8_1(float* x,const float* add,const float* w,float* y,void* q,int n,int rows,float eps,int,void* s) {
    if(add) poly::axpy(add,1,x,n*rows,true,s);
    poly::rms_norm(x,w,y,n,rows,eps,s);
    native_quantize_q8_1(y,q,n,rows,s);
}
void native_mmvq(int type,const void* w,const void* x,float* y,int ni,int no,int rows,void* s) {
    native_mmvq_weight_bytes(type,ni,no);
    QArgs a;a.p[0]=w;a.p[1]=x;a.p[2]=y;a.ni=ni;a.no=no;a.rows=rows;a.xstride=native_q8_1_bytes(ni);a.ystride=no;
    qrun(std::string("poly_mul_")+quant_name(type),a,s,{uint32_t(no),uint32_t(rows),1});
}
bool native_mmvq_slots(int t,const MmvqSlots& a,int count,int ni,int no,void* s) {
    if(!native_mmvq_supported(t) || count<0 || count>8) return false;
    if(!count) return true;
    if(!gpu::metal::optimize()) {
        for(int i=0;i<count;++i) native_mmvq(t,a.w[i],a.x[i],a.y[i],ni,no,1,s);
        return true;
    }
    native_mmvq_weight_bytes(t,ni,no);
    QArgs q; q.ni=ni; q.no=no; q.rows=count; q.count=count; q.ystride=1; q.kind=8;
    for(int i=0;i<count;++i) { q.p[3*i]=a.w[i]; q.p[3*i+1]=a.x[i]; q.p[3*i+2]=a.y[i]; }
    qrun(std::string("poly_mul_")+quant_name(t),q,s,{uint32_t(no),uint32_t(count),1});
    return true;
}
bool native_mmvq_ids(int t,const MmvqIds& a,int count,int ni,int no,void* s) {
    if(!native_mmvq_supported(t)) return false;
    native_mmvq_weight_bytes(t,ni,no);
    QArgs q;q.p[0]=a.table;q.p[1]=a.x;q.p[2]=a.y;q.p[3]=a.ids;q.ni=ni;q.no=no;q.rows=count;
    q.xstride=a.x_stride;q.ystride=a.y_stride;q.offset=a.offset;q.every=a.x_every;q.kind=1;
    qrun(std::string("poly_mul_")+quant_name(t),q,s,{uint32_t(no),uint32_t(count),1});return true;
}
}
namespace strata::poly {
using namespace metal_detail;
Mmq::Mmq() = default;
Mmq::~Mmq() = default;
bool Mmq::fits(int t,int64_t rows,int64_t cols) {
    return kernels::native_mmvq_supported(t) && rows>0 && rows<=INT32_MAX && cols>0 && cols<=INT32_MAX && cols%ggml_blck_size(ggml_type(t))==0;
}
size_t Mmq::q8_bytes(int64_t rows,int64_t cols) { return kernels::native_q8_1_bytes(int(cols),int(rows)); }
void Mmq::quantize(const float* x,const int32_t* ids,void* q,int,int64_t cols,int64_t ld,int64_t rows,void* s) {
    QArgs a;a.p[0]=x;a.p[1]=q;a.p[2]=ids;a.ni=int(cols);a.rows=int(rows);a.xstride=ld;
    qrun("poly_quantize",a,s,{uint32_t(cols/32),uint32_t(rows),1});
}
void Mmq::mul(const void* w,int type,int64_t nr,int64_t nc,size_t stride,int n,const void* x,
              const int32_t* bounds,const int32_t* ids,int64_t total,int64_t max_rows,float* y,int64_t ld,void* s) {
    if(n<=0 || total<=0 || max_rows<=0) return;
    if(!fits(type,nr,nc)) throw std::invalid_argument("Metal MMQ shape");
    const bool opt=gpu::metal::optimize(), flat=opt && bounds && total<max_rows*n;
    QArgs a;a.p[0]=w;a.p[1]=x;a.p[2]=y;a.p[3]=ids;a.p[4]=bounds;a.ni=int(nc);a.no=int(nr);a.rows=int(total);a.count=n;
    a.xstride=kernels::native_q8_1_bytes(int(nc));a.ystride=ld;a.stride=stride;a.kind=flat?7:(opt?6:2);
    qrun(std::string("poly_mul_")+quant_name(type),a,s,{uint32_t(nr),uint32_t(flat?total:(opt?max_rows:total)),uint32_t(flat?1:n)});
}
}
