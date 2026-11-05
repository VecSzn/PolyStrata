#pragma once
#include "strata/gpu/runtime.hpp"
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
namespace g=strata::gpu;
inline void ck(g::Error e) { if(e!=g::kSuccess) throw std::runtime_error(g::error_string(e)); }
struct Buffer {
    void* ptr=nullptr;
    explicit Buffer(size_t bytes) { ck(g::malloc_device_raw(&ptr,bytes)); }
    explicit Buffer(const std::vector<float>& v):Buffer(v.size()*4) {put(v.data(),v.size()*4);}
    ~Buffer(){g::free_device(ptr);}
    Buffer(const Buffer&)=delete;
    template<class T=float> T* as()const{return static_cast<T*>(ptr);}
    void put(const void* p,size_t n){ck(g::memcpy_sync(ptr,p,n,g::kHostToDevice));}
    template<class T=float> std::vector<T> read(size_t n)const{std::vector<T> v(n);ck(g::memcpy_sync(v.data(),ptr,n*sizeof(T),g::kDeviceToHost));return v;}
};
inline std::vector<float> values(size_t n,float scale=1){std::vector<float> v(n);for(size_t i=0;i<n;++i)v[i]=scale*std::sin(float(i)*.37f+.17f);return v;}
inline int compare(const std::vector<float>& got,const std::vector<float>& ref,const char* label,float eps=3e-5f){
    int bad=0;float most=0;for(size_t i=0;i<ref.size();++i){float d=std::abs(got[i]-ref[i]);most=std::max(most,d);bad+=!std::isfinite(got[i])||d>eps*(1+std::abs(ref[i]));}
    std::printf("%s: %d mismatches, max error %g\n",label,bad,most);return bad;
}
