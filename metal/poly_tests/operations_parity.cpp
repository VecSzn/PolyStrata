#include "test_utils.hpp"
#include "strata/poly/kernels.hpp"
#include <algorithm>
#include <cstring>
namespace p=strata::poly;
int main(){
    int bad=0;g::Stream s;ck(g::stream_create(&s));
    const int n=73,rows=3;auto gu=values(rows*2*n,3);Buffer input(gu),out(rows*n*4);
    for(bool gelu:{false,true})for(bool separate:{false,true}){
        int count=separate?1:rows;std::vector<float> ref(count*n);
        for(int r=0;r<count;++r)for(int i=0;i<n;++i){float g=gu[r*2*n+i],u=gu[r*2*n+n+i];
            if(gelu)ref[r*n+i]=.5f*g*(1+std::tanh(.7978845608028654f*g*(1+.044715f*g*g)))*u;
            else{g=std::min(g,1.2f);u=std::clamp(u,-1.2f,1.2f);ref[r*n+i]=g/(1+std::exp(-g))*u;}}
        if(gelu){if(separate)p::geglu(input.as(),input.as()+n,out.as(),n,s);else p::geglu_rows(input.as(),out.as(),n,rows,s);}
        else{if(separate)p::swiglu_clamp(input.as(),input.as()+n,out.as(),n,1.2f,s);else p::swiglu_clamp_rows(input.as(),out.as(),n,rows,1.2f,s);}
        ck(g::stream_synchronize(s));bad+=compare(out.read(ref.size()),ref,"activation");
    }
    for(bool one:{false,true}){
        auto acc=values(n*rows),gates=values((n+4)*rows);Buffer da(acc),dg(gates);
        for(int r=0;r<rows;++r)for(int i=0;i<n;++i)acc[r*n+i]/=1+std::exp(-gates[r*(n+4)+(one?0:i)]);
        p::mul_sigmoid(da.as(),dg.as(),n,rows,n+4,one,s);ck(g::stream_synchronize(s));bad+=compare(da.read(acc.size()),acc,"gate strides");
    }
    auto x=values(rows*2*n),w=values(rows*2,.3f),acc=values(rows*n);Buffer dx(x),dw(w),da(acc);
    for(int r=0;r<rows;++r)for(int i=0;i<n;++i)for(int k=0;k<2;++k)acc[r*n+i]+=w[r*2+k]*x[(r*2+k)*n+i];
    p::moe_sum(dx.as(),dw.as(),da.as(),n,2,rows,s);ck(g::stream_synchronize(s));bad+=compare(da.read(acc.size()),acc,"weighted experts");
    // Pinned pointers differ from GPU addresses on Apple Silicon. Exercise both
    // directions, a second run, graph replay and overlapping accumulation rows.
    float* pinned=nullptr;ck(g::host_alloc_raw(reinterpret_cast<void**>(&pinned),n*3*4,0));
    for(int i=0;i<n;++i)pinned[i]=x[i];
    ck(g::begin_capture(s));p::copy2(pinned,out.ptr,n,dx.ptr,pinned+n,n,s);
    p::ScaledRows scaled{};scaled.x[0]=out.as();scaled.x[1]=pinned+n;scaled.acc[0]=pinned+2*n;scaled.acc[1]=pinned+2*n;scaled.w[0]=.5f;scaled.w[1]=.25f;
    p::add_scaled(scaled,2,n,s);g::Graph graph;ck(g::end_capture(s,&graph));g::GraphExec exec;ck(g::graph_instantiate(&exec,graph));
    for(int run=0;run<2;++run){std::fill(pinned+2*n,pinned+3*n,0);ck(g::graph_launch(exec,s));ck(g::stream_synchronize(s));std::vector<float> ref(n);for(int i=0;i<n;++i)ref[i]=.75f*x[i];bad+=compare(std::vector<float>(pinned+2*n,pinned+3*n),ref,"mapped copy and add");}
    g::graph_exec_destroy(exec);g::graph_destroy(graph);g::free_host(pinned);
    auto logits=values(113,4);logits[7]=8;logits[92]=8;logits[110]=-40;Buffer dl(logits);p::Likeliest* result=nullptr;ck(g::host_alloc_raw(reinterpret_cast<void**>(&result),sizeof(*result),0));
    p::likeliest(dl.as(),logits.size(),result,s);ck(g::stream_synchronize(s));double denom=0;for(float v:logits)if(v>8-20)denom+=std::exp(double(v)-8);
    bad+=result->token!=7||std::abs(result->prob-1/denom)>1e-6;g::free_host(result);
    ck(g::stream_destroy(s));return bad?1:0;
}
