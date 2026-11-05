#include "strata/poly/kernels.hpp"
#include "strata/gpu/runtime.hpp"
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
namespace g = strata::gpu;
namespace p = strata::poly;
void ck(g::Error e) { if(e != g::kSuccess) throw std::runtime_error(g::error_string(e)); }
struct Dev {
    void* ptr;
    explicit Dev(size_t n) { ck(g::malloc_device_raw(&ptr, n)); }
    ~Dev() { g::free_device(ptr); }
    float* f() { return static_cast<float*>(ptr); }
    void put(const void* x,size_t n) { ck(g::memcpy_sync(ptr,x,n,g::kHostToDevice)); }
    std::vector<float> read(size_t n) { std::vector<float> v(n); ck(g::memcpy_sync(v.data(),ptr,n*4,g::kDeviceToHost)); return v; }
};
bool near(float a, double b) { return std::isfinite(a) && std::abs(a-b) <= 3e-5 * (1+std::abs(b)); }
int main() {
    int fails = 0;
    g::Stream stream; ck(g::stream_create(&stream));
    for(int n : {17,256,769,4096}) {
        const int rows=3, outputs=12, group=4, stride=n+7;
        std::vector<float> x(rows*3*n), w(outputs*n), scale(n), bias(n);
        for(size_t i=0;i<x.size();++i) x[i]=std::sin(double(i)*.37f);
        for(size_t i=0;i<w.size();++i) w[i]=std::cos(double(i)*.21f)*.03f;
        for(int i=0;i<n;++i) { scale[i]=.7f+.01f*(i%7); bias[i]=.01f*(i%3); }
        Dev dx(x.size()*4), dw(w.size()*4), ds(n*4), db(n*4), dy(rows*stride*4);
        dx.put(x.data(),x.size()*4); dw.put(w.data(),w.size()*4); ds.put(scale.data(),n*4); db.put(bias.data(),n*4);
        for(int kind=0;kind<2;++kind) {
            if(kind) p::layer_norm(dx.f(),ds.f(),db.f(),dy.f(),n,rows,1e-5,stream);
            else p::rms_norm(dx.f(),ds.f(),dy.f(),n,rows,1e-5,stream,stride);
            ck(g::stream_synchronize(stream)); auto y=dy.read(rows*stride);
            for(int r=0;r<rows;++r) {
                double mean=0,ss=0;
                if(kind) { for(int i=0;i<n;++i) mean+=x[r*n+i]; mean/=n; }
                for(int i=0;i<n;++i) { double z=x[r*n+i]-mean; ss+=z*z; }
                for(int i=0;i<n;++i) fails+=!near(y[r*(kind?n:stride)+i],(x[r*n+i]-mean)/std::sqrt(ss/n+1e-5)*scale[i]+(kind?bias[i]:0));
            }
        }
        for(int bf=0;bf<2;++bf) {
            std::vector<uint16_t> bits(w.size()); std::vector<float> weights=w;
            if(bf) {
                for(size_t i=0;i<w.size();++i) { uint32_t v; std::memcpy(&v,&w[i],4); bits[i]=v>>16; v &= 0xffff0000; std::memcpy(&weights[i],&v,4); }
                dw.put(bits.data(),bits.size()*2);
                p::gemv_bf16(static_cast<uint16_t*>(dw.ptr),dx.f(),dy.f(),n,outputs,group,rows,stream);
            } else p::gemv_f32(dw.f(),dx.f(),dy.f(),n,outputs,group,rows,stream);
            ck(g::stream_synchronize(stream)); auto y=dy.read(rows*outputs);
            for(int r=0;r<rows;++r) for(int o=0;o<outputs;++o) {
                double sum=0; for(int i=0;i<n;++i) sum+=double(weights[o*n+i])*x[(r*3+o/group)*n+i];
                fails+=!near(y[r*outputs+o],sum);
            }
        }
        p::axpy(dx.f(),.25f,dy.f(),n,false,stream); p::axpy(dx.f(),.5f,dy.f(),n,true,stream);
        p::add_bias(db.f(),dy.f(),n,1,stream); p::softcap(dy.f(),n,2,stream);
        ck(g::stream_synchronize(stream)); auto y=dy.read(n);
        for(int i=0;i<n;++i) fails+=!near(y[i],2*std::tanh((.75*x[i]+bias[i])/2));
    }
    ck(g::stream_destroy(stream));
    std::printf("poly basic CPU-reference mismatches: %d\n",fails);
    return fails ? 1 : 0;
}
