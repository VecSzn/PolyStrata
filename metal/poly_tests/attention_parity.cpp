#include "test_utils.hpp"
#include "strata/poly/kernels.hpp"
#include <algorithm>
namespace p=strata::poly;
int main(){
    int bad=0; g::Stream s;ck(g::stream_create(&s));
    for(int dim:{32,64,128})for(bool quant:{false,true})for(int window:{0,7})for(bool whole:{false,true}){
        const int heads=4,kv=2,rows=3,total=19,pos=total-rows,ring=window?11:0,n=kv*dim;
        auto keys=values(total*n),vals=values(total*n,.63f),query=values(rows*heads*dim,.42f);
        Buffer dk(keys),dv(vals),dq(query),kcache(total*n*2),vcache(total*n*2),kscale(total*n/32*2),vscale(total*n/32*2),out(rows*heads*dim*4),at(8);
        long long position=0;at.put(&position,8);
        if(quant){p::store_q8_at(dk.as(),kcache.as<int8_t>(),kscale.as<uint16_t>(),n,at.as<long long>(),total,0,s);p::store_q8_at(dv.as(),vcache.as<int8_t>(),vscale.as<uint16_t>(),n,at.as<long long>(),total,0,s);}
        else {p::store_f16(dk.as(),kcache.as<uint16_t>(),n,0,total,s);p::store_f16_at(dv.as(),vcache.as<uint16_t>(),n,at.as<long long>(),total,0,s);}
        ck(g::stream_synchronize(s));
        // Independent CPU decoding of what store wrote, checked against FP16/Q8 rounding.
        auto decode=[&](Buffer& c,Buffer& scale,const std::vector<float>& original){
            auto half=c.read<_Float16>(total*n);auto codes=c.read<int8_t>(total*n);auto scales=scale.read<_Float16>(total*n/32);
            std::vector<float> result(total*n);
            for(int i=0;i<total*n;++i){
                if(!quant){bad+=float(half[i])!=float(_Float16(original[i]));result[i]=float(half[i]);}
                else {float mx=0;for(int j=i/32*32;j<i/32*32+32;++j)mx=std::max(mx,std::abs(original[j]));float d=mx/127;
                    bad+=codes[i]!=int(std::round(original[i]/d));bad+=float(scales[i/32])!=float(_Float16(d));result[i]=codes[i]*float(scales[i/32]);}
            }return result;
        };
        auto k=decode(kcache,kscale,keys),v=decode(vcache,vscale,vals);
        // Ring reads are tested against explicitly wrapped physical cache indices.
        std::vector<float> ref(rows*heads*dim);
        for(int t=0;t<rows;++t)for(int h=0;h<heads;++h){
            int lo=window?std::max(0,pos+t-window+1):0,hi=whole?total-1:pos+t;
            std::vector<double> scores;double mx=-1e100;
            for(int j=lo;j<=hi;++j){int off=((ring?j%ring:j)*kv+h/(heads/kv))*dim;double dot=0;for(int d=0;d<dim;++d)dot+=double(query[(t*heads+h)*dim+d])*k[off+d];scores.push_back(dot*.125);mx=std::max(mx,dot*.125);}
            double denom=0;for(auto& a:scores){a=std::exp(a-mx);denom+=a;}
            for(int d=0;d<dim;++d){double sum=0;for(int j=lo;j<=hi;++j)sum+=scores[j-lo]*v[((ring?j%ring:j)*kv+h/(heads/kv))*dim+d];ref[(t*heads+h)*dim+d]=sum/denom;}
        }
        p::Kvq8 kq{quant?kcache.as<int8_t>():nullptr,kscale.as<uint16_t>()},vq{quant?vcache.as<int8_t>():nullptr,vscale.as<uint16_t>()};
        position=pos;at.put(&position,8);
        ck(g::begin_capture(s));p::attend(dq.as(),kcache.as<uint16_t>(),vcache.as<uint16_t>(),0,rows,nullptr,2,out.as(),dim,heads,kv,.125f,s,window,ring,whole,at.as<long long>(),total,0,kq,vq);
        g::Graph graph;ck(g::end_capture(s,&graph));g::GraphExec exec;ck(g::graph_instantiate(&exec,graph));ck(g::graph_launch(exec,s));ck(g::stream_synchronize(s));
        bad+=compare(out.read(rows*heads*dim),ref,"attention");g::graph_exec_destroy(exec);g::graph_destroy(graph);
    }
    // Per-head RMS, NeoX pairs, frequencies, image axes, and an in-place output.
    for(bool axes:{false,true}){
        int dim=64,heads=2,rows=3,rot=32;auto x=values(dim*heads*rows),w=values(dim,.2f);for(auto& v:w)v+=1;
        std::vector<float> freq(rot/2,1.3f),ref(x.size());std::vector<int32_t> axis={3,2,1,4,2,2,5,3,2};
        Buffer dx(x),dw(w),df(freq),da(axis.size()*4);da.put(axis.data(),axis.size()*4);
        for(int h=0;h<heads*rows;++h){double ss=0;for(int i=0;i<dim;++i)ss+=double(x[h*dim+i])*x[h*dim+i];double inv=1/std::sqrt(ss/dim+1e-5);
            for(int i=0;i<dim;++i){double value=x[h*dim+i]*inv*w[i];if(i<rot){int j=i%(rot/2);double theta=(axes?axis[(h/heads)*3+(j%6)%3]:7+h/heads)*std::pow(.91,j)/freq[j];double a=x[h*dim+j]*inv*w[j],b=x[h*dim+j+rot/2]*inv*w[j+rot/2];value=i<rot/2?a*std::cos(theta)-b*std::sin(theta):a*std::sin(theta)+b*std::cos(theta);}ref[h*dim+i]=value;}}
        p::head_norm_rope(dx.as(),dim,dw.as(),dx.as(),dim,heads,rot,.91f,7,rows,1e-5f,s,df.as(),axes?da.as<int32_t>():nullptr,6);ck(g::stream_synchronize(s));bad+=compare(dx.read(x.size()),ref,"rope");
    }
    ck(g::stream_destroy(s));return bad?1:0;
}
