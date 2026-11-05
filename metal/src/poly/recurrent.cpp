#include "strata/poly/kernels.hpp"
#include "launch.hpp"
namespace strata::poly {
void gdn_step(float* qkv,float* alpha,float* beta,const float* dt,const float* a,float* state,const float* z,
              const float* norm,float* out,int dim,int heads,int keys,int rows,float eps,void* s,
              float* snap,size_t stride,unsigned snaps) {
    if(dim<=0 || dim>256 || dim%32 || keys<=0 || heads%keys) throw std::invalid_argument("Metal GDN shape");
    if(rows<=0) return;
    struct Args { const void* p[10]; uint64_t stride; int dim,heads,keys,rows; float eps; unsigned snaps; }
        args{{qkv,alpha,beta,dt,a,state,z,norm,out,snap},stride,dim,heads,keys,rows,eps,snaps};
    static_assert(sizeof(Args)==112);
    static const auto source=metal_detail::read_source(STRATA_POLY_RECURRENT_SOURCE);
    auto run=[&](const char* kernel,uint32_t groups) {
        metal_detail::dispatch("poly_recurrent",source,kernel,args,s,{groups,1,1},32);
    };
    run("poly_gdn_norm",rows*keys);
    run("poly_gdn_gates",(rows*heads+31)/32);
    run("poly_gdn_state",heads*dim);
    run("poly_gdn_out",rows*heads);
}
}
