#include "test_utils.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/poly/kernels.hpp"
#include "mmq.hpp"
#include "ggml.h"
#include "strata/gpu/metal.hpp"
#include <cstring>
namespace k=strata::kernels;
namespace p=strata::poly;
int main(){
    int bad=0;g::Stream stream;ck(g::stream_create(&stream));
    const int ni=256,no=5,rows=3,matrices=2;auto wf=values(matrices*no*ni),x=values(rows*ni,.8f);std::vector<float> importance(ni,1);
    Buffer dx(x),q(k::native_q8_1_bytes(ni,rows)),y(rows*no*4);k::native_quantize_q8_1(dx.as(),q.ptr,ni,rows,stream);ck(g::stream_synchronize(stream));
    auto qb=q.read<unsigned char>(k::native_q8_1_bytes(ni,rows));std::vector<float> xq(x.size());
    for(size_t i=0;i<x.size();++i){_Float16 d;std::memcpy(&d,qb.data()+(i/32)*36,2);xq[i]=float(d)*reinterpret_cast<int8_t*>(qb.data())[(i/32)*36+4+i%32];}
    for(bool optimized:{false,true}) { g::metal::set_optimize(optimized);
    for(int type:{2,3,6,7,8,10,11,12,13,14,16,17,18,19,20,21,22,23,29,39,42}){
        auto t=ggml_type(type);ggml_quantize_init(t);size_t size=k::native_mmvq_weight_bytes(type,ni,no);
        std::vector<unsigned char> weights(size*matrices);
        ggml_quantize_chunk(t,wf.data(),weights.data(),0,matrices*no,ni,importance.data());
        auto traits=ggml_get_type_traits(t);std::vector<float> dwf(wf.size());traits->to_float(weights.data(),dwf.data(),dwf.size());
        Buffer dw(weights.size());dw.put(weights.data(),weights.size());
        auto reference=[&](const std::vector<int>& experts,const std::vector<int>& inputs,const std::vector<int>& outputs){
            std::vector<float> ref(rows*no);
            for(int r=0;r<rows;++r)for(int o=0;o<no;++o){double sum=0;for(int i=0;i<ni;++i)sum+=double(dwf[(experts[r]*no+o)*ni+i])*xq[inputs[r]*ni+i];ref[outputs[r]*no+o]=sum;}return ref;
        };
        k::native_mmvq(type,dw.ptr,q.ptr,y.as(),ni,no,rows,stream);ck(g::stream_synchronize(stream));
        bad+=compare(y.read(rows*no),reference({0,0,0},{0,1,2},{0,1,2}),ggml_type_name(t),1e-4f);
        std::vector<const void*> table={dw.ptr,static_cast<unsigned char*>(dw.ptr)+size};std::vector<int32_t> ids={1,0,1};
        Buffer tab(table.size()*sizeof(void*)),di(ids.size()*4);tab.put(table.data(),table.size()*sizeof(void*));di.put(ids.data(),ids.size()*4);
        k::MmvqIds selected{tab.as<const void*>(),di.as<int32_t>(),0,q.ptr,k::native_q8_1_bytes(ni),1,y.as(),no};
        k::native_mmvq_ids(type,selected,rows,ni,no,stream);ck(g::stream_synchronize(stream));
        bad+=compare(y.read(rows*no),reference({1,0,1},{0,1,2},{0,1,2}),"expert ids",1e-4f);
        k::MmvqSlots slots{};
        for(int r=0;r<rows;++r){slots.w[r]=table[ids[r]];slots.x[r]=static_cast<unsigned char*>(q.ptr)+r*k::native_q8_1_bytes(ni);slots.y[r]=y.as()+r*no;}
        k::native_mmvq_slots(type,slots,rows,ni,no,stream);ck(g::stream_synchronize(stream));
        bad+=compare(y.read(rows*no),reference({1,0,1},{0,1,2},{0,1,2}),"expert slots",1e-4f);
        std::vector<int32_t> gather={2,0,1},bounds={0,1,3},scatter={1,2,0};
        Buffer dg(12),db(12),ds(12);dg.put(gather.data(),12);db.put(bounds.data(),12);ds.put(scatter.data(),12);
        p::Mmq mmq;p::Mmq::quantize(dx.as(),dg.as<int32_t>(),q.ptr,type,ni,ni,rows,stream);
        mmq.mul(dw.ptr,type,no,ni,size,matrices,q.ptr,db.as<int32_t>(),ds.as<int32_t>(),rows,2,y.as(),no,stream);ck(g::stream_synchronize(stream));
        bad+=compare(y.read(rows*no),reference({0,1,1},{2,0,1},{1,2,0}),"batch gather/scatter",1e-4f);
        // Restore the input ordering for the next weight format.
        k::native_quantize_q8_1(dx.as(),q.ptr,ni,rows,stream);
    }
    }
    // Empty first expert and nonzero bounds; compact dispatch must preserve the
    // untouched row and global gather/scatter indices, with output padding.
    for(int type:{12,13,14}) {
        const int ni2=512,no2=7,rows2=5,stride=11;
        auto w=values(2*no2*ni2), x2=values(rows2*ni2);std::vector<float> imp(ni2,1);
        auto ty=ggml_type(type);ggml_quantize_init(ty);
        size_t wb=k::native_mmvq_weight_bytes(type,ni2,no2);
        std::vector<unsigned char> packed(wb*2);ggml_quantize_chunk(ty,w.data(),packed.data(),0,2*no2,ni2,imp.data());
        Buffer dw(packed.size()),dx2(x2),qx(k::native_q8_1_bytes(ni2,rows2)),out(rows2*stride*4),bd(12),id(20);
        dw.put(packed.data(),packed.size());int32_t bounds[]={1,1,5},ids[]={3,4,2,1,0};bd.put(bounds,12);id.put(ids,20);
        k::native_quantize_q8_1(dx2.as(),qx.ptr,ni2,rows2,stream);
        std::vector<float> old;
        for(bool optimized:{false,true}) {
            g::metal::set_optimize(optimized);ck(g::memset_async(out.ptr,0,rows2*stride*4,stream));
            p::Mmq mmq;mmq.mul(dw.ptr,type,no2,ni2,wb,2,qx.ptr,bd.as<int32_t>(),id.as<int32_t>(),rows2,4,out.as(),stride,stream);
            ck(g::stream_synchronize(stream));auto got=out.read(rows2*stride);
            if(!optimized)old=got;else bad+=compare(got,old,"compact empty/nonzero/stride",0);
        }
    }
    ck(g::stream_synchronize(stream));ck(g::stream_destroy(stream));ggml_quantize_free();return bad?1:0;
}
