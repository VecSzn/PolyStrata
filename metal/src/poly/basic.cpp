// strata-poly's basic forward-pass operations on the native Metal stream runtime.
#include "strata/poly/kernels.hpp"
#include "strata/gpu/metal.hpp"
#include <algorithm>
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
#include <type_traits>
#include <vector>

namespace strata::poly {
namespace {
struct BasicArgs {
    const void* p[4]{};
    uint64_t count = 0;
    int n = 0, rows = 0, stride = 0, kind = 0;
    float scale = 0, eps = 0;
};
static_assert(sizeof(BasicArgs) == 64);
template<class Args>
void launch(const char* name, const Args& input, void* stream, gpu::metal::Grid groups, uint32_t threads, gpu::metal::Resources resources = {}) {
    Args args = input;
    if constexpr (std::is_same_v<Args, BasicArgs>) {
        for (auto& ptr : args.p) if (ptr) {
            void* mapped = nullptr;
            auto rc = gpu::host_device_pointer_raw(&mapped, const_cast<void*>(ptr));
            if (rc != gpu::kSuccess) throw std::runtime_error(gpu::error_string(rc));
            ptr = mapped;
        }
    }
    static const std::string source = [] {
        std::ifstream f(STRATA_POLY_BASIC_SOURCE);
        if (!f) throw std::runtime_error("cannot read strata-poly Metal kernels");
        return std::string(std::istreambuf_iterator<char>(f), {});
    }();
    std::string error;
    auto* pipeline = gpu::metal::pipeline("poly_basic", source.c_str(), name, error);
    if (!pipeline) throw std::runtime_error(error);
    if constexpr (std::is_same_v<Args, BasicArgs>) resources = {args.p,4,true};
    auto rc = gpu::metal::dispatch(static_cast<gpu::Stream>(stream), pipeline, &args, sizeof(args), groups,
                                   {threads, 1, 1},0,resources);
    if (rc != gpu::kSuccess) throw std::runtime_error(gpu::error_string(rc));
}
void element(const BasicArgs& a, void* stream) {
    if (!a.count) return;
    launch("poly_elementwise", a, stream, {uint32_t((a.count + 255) / 256), 1, 1}, 256);
}
}

int sum_threads(int n) {
    int threads = 128;
    while (threads < 1024 && threads * 4 < n) threads *= 2;
    return threads;
}
void rms_norm(const float* x, const float* w, float* y, int n, int rows, float eps, void* stream, int stride) {
    if (n <= 0 || rows <= 0) return;
    BasicArgs a{{x, w, y, nullptr}};
    a.n = n; a.stride = stride; a.eps = eps;
    launch("poly_norm", a, stream, {uint32_t(rows), 1, 1}, sum_threads(n));
}
void layer_norm(const float* x, const float* w, const float* b, float* y, int n, int rows, float eps, void* stream) {
    if (n <= 0 || rows <= 0) return;
    BasicArgs a{{x, w, y, b}};
    a.n = n; a.eps = eps; a.kind = 1;
    launch("poly_norm", a, stream, {uint32_t(rows), 1, 1}, sum_threads(n));
}
static void gemv(const void* w, const float* x, float* y, int ni, int no, int group, int rows, int kind, void* stream) {
    if (ni <= 0 || no <= 0 || rows <= 0 || group <= 0 || no % group) throw std::invalid_argument("invalid Metal GEMV shape");
    BasicArgs a{{w, x, y, nullptr}};
    a.n = ni; a.rows = no; a.stride = group; a.kind = kind;
    launch("poly_gemv", a, stream, {uint32_t(no), uint32_t(rows), 1}, ni < 128 ? 32 : sum_threads(ni));
}
void gemv_f32(const float* w, const float* x, float* y, int ni, int no, int group, int rows, void* stream) {
    gemv(w, x, y, ni, no, group, rows, 0, stream);
}
void gemv_bf16(const uint16_t* w, const float* x, float* y, int ni, int no, int group, int rows, void* stream) {
    gemv(w, x, y, ni, no, group, rows, 1, stream);
}
void bf16_to_f32(const uint16_t* w, float* y, size_t n, void* stream) {
    BasicArgs a{{w, y}}; a.count = n; element(a, stream);
}
void axpy(const float* x, float scale, float* y, int n, bool add, void* stream) {
    BasicArgs a{{x, y}}; a.count = std::max(n, 0); a.kind = add ? 2 : 1; a.scale = scale; element(a, stream);
}
void softcap(float* x, size_t n, float cap, void* stream) {
    BasicArgs a{{x}}; a.count = n; a.kind = 3; a.scale = cap; element(a, stream);
}
void add_bias(const float* b, float* y, int n, int rows, void* stream) {
    if (n <= 0 || rows <= 0) return;
    BasicArgs a{{b, y}}; a.count = uint64_t(n) * rows; a.n = n; a.kind = 4; element(a, stream);
}
namespace {
struct ConvArgs {
    const void* x; void* state; const void* w; void* out; void* snap;
    uint64_t stride;
    int channels, rows, kind;
    uint32_t snaps;
};
static_assert(sizeof(ConvArgs) == 64);
void convolution(const float* x, float* state, const float* w, float* out, int channels, int rows,
                 void* stream, float* snap, size_t stride, unsigned snaps, int kind) {
    if (rows <= 0 || channels <= 0) return;
    ConvArgs a{x,state,w,out,snap,stride,channels,rows,kind,snaps};
    const void* refs[]={x,state,w,out,snap};
    launch("poly_conv", a, stream, {uint32_t((channels+255)/256),1,1}, 256,{refs,5,true});
}
}
void shortconv(const float* x, float* state, const float* w, float* out, int channels, int rows,
               void* stream, float* snap, size_t stride, unsigned snaps) {
    convolution(x,state,w,out,channels,rows,stream,snap,stride,snaps,0);
}
void conv4_silu(const float* x, float* state, const float* w, float* out, int channels, int rows,
               void* stream, float* snap, size_t stride, unsigned snaps) {
    convolution(x,state,w,out,channels,rows,stream,snap,stride,snaps,1);
}
void route_rows(const float* logits, int experts, int used, int rows, int gate, float in_scale,
                const float* bias, const float* escale, bool norm, float w_scale, int32_t* ids, float* weights, void* stream) {
    if (rows <= 0) return;
    if (experts < 1 || experts > 512 || used < 1 || used > 32 || used > experts || gate < 0 || gate > 2)
        throw std::invalid_argument("invalid Metal expert routing shape");
    struct Args {
        const void* logits; const void* bias; const void* escale; void* ids; void* weights;
        int experts, used, rows, gate;
        float in_scale, w_scale;
        int norm;
    } a{logits,bias,escale,ids,weights,experts,used,rows,gate,in_scale,w_scale,norm};
    static_assert(sizeof(Args) == 72);
    const void* refs[]={logits,bias,escale,ids,weights};
    launch("poly_route",a,stream,{uint32_t(rows),1,1},32,{refs,5,true});
}
// The Metal batch path uses the same arithmetic as one-token GEMV.
void* blas_create(void* stream) { return new void*(stream); }
void blas_destroy(void* blas) { delete static_cast<void**>(blas); }
void gemm_f32(void* blas, const float* w, const float* x, float* y, int ni, int no, int group, int rows) {
    gemv_f32(w,x,y,ni,no,group,rows,blas ? *static_cast<void**>(blas) : nullptr);
}
static void activation(const float* gate, const float* up, float* out, int n, int rows, float limit,
                       int kind, void* stream) {
    BasicArgs a{{gate,up,out}}; a.n=n; a.count=uint64_t(n)*rows; a.scale=limit; a.kind=kind;
    element(a,stream);
}
void swiglu_clamp(const float* g,const float* u,float* o,int n,float limit,void* s) { activation(g,u,o,n,1,limit,5,s); }
void swiglu_clamp_rows(const float* gu,float* o,int n,int rows,float limit,void* s) { activation(gu,nullptr,o,n,rows,limit,5,s); }
void geglu(const float* g,const float* u,float* o,int n,void* s) { activation(g,u,o,n,1,0,6,s); }
void geglu_rows(const float* gu,float* o,int n,int rows,void* s) { activation(gu,nullptr,o,n,rows,0,6,s); }
void moe_sum(const float* x,const float* w,float* acc,int n,int used,int rows,void* s) {
    BasicArgs a{{x,w,acc}}; a.n=n; a.stride=used; a.count=uint64_t(n)*rows; a.kind=7; element(a,s);
}
void mul_sigmoid(float* acc,const float* g,int n,int rows,int stride,bool one,void* s) {
    BasicArgs a{{acc,g}}; a.n=n; a.rows=one; a.stride=stride; a.count=uint64_t(n)*rows; a.kind=8; element(a,s);
}
void copy2(const void* x0,void* y0,int n0,const void* x1,void* y1,int n1,void* s) {
    BasicArgs a{{x0,y0,x1,y1}}; a.n=n0; a.count=uint64_t(n0)+n1; a.kind=9; element(a,s);
}
void add_scaled(const ScaledRows& rows,int count,int n,void* s) {
    if (count < 0 || count > 32) throw std::invalid_argument("add_scaled: count outside 0..32");
    struct Args { ScaledRows rows; int count,n; } a{rows,count,n};
    for (int i=0;i<count;++i) {
        void* p=nullptr;
        if (gpu::host_device_pointer_raw(&p,const_cast<float*>(a.rows.x[i])) != gpu::kSuccess) throw std::runtime_error("unmapped scaled input");
        a.rows.x[i]=static_cast<float*>(p);
        if (gpu::host_device_pointer_raw(&p,a.rows.acc[i]) != gpu::kSuccess) throw std::runtime_error("unmapped scaled output");
        a.rows.acc[i]=static_cast<float*>(p);
    }
    std::vector<const void*> refs;
    for(int i=0;i<count;++i){refs.push_back(a.rows.x[i]);refs.push_back(a.rows.acc[i]);}
    launch("poly_scaled",a,s,{uint32_t((n+255)/256),1,1},256,{refs.data(),refs.size(),true});
}
void likeliest(const float* logits,int n,Likeliest* out,void* s) {
    BasicArgs a{{logits,out}}; a.n=n;
    launch("poly_likeliest",a,s,{1,1,1},32);
}
} // namespace strata::poly
