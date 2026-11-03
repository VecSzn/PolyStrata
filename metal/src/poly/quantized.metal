// ggml's pinned dequantize.h is prepended by quantized.cpp (upstream MIT license).
struct QArgs {
    device void* p[24]; ulong stride,xstride,ystride,offset;
    int ni,no,rows,count,every,kind;
};
kernel void poly_quantize(constant QArgs& a [[buffer(0)]],uint2 group [[threadgroup_position_in_grid]],
                          uint lane [[thread_index_in_threadgroup]],uint2 shape [[threads_per_threadgroup]]) {
    int row=a.p[2]?static_cast<device int*>(a.p[2])[group.y]:int(group.y);
    float v=static_cast<device float*>(a.p[0])[ulong(row)*a.xstride+group.x*32+lane];
    if(a.kind==1) {float u=static_cast<device float*>(a.p[3])[ulong(row)*a.xstride+group.x*32+lane];v=v/(1+exp(-v))*u;}
    float d=simd_max(abs(v))/127.0f,sum=v;
    for(uint k=16;k;k>>=1) sum+=simd_shuffle_xor(sum,k);
    device uchar* dst=static_cast<device uchar*>(a.p[1])+(ulong(group.y)*(a.ni/32)+group.x)*36;
    reinterpret_cast<device char*>(dst+4)[lane]=char(d?round(v/d):0);
    if(!lane) { reinterpret_cast<device half*>(dst)[0]=half(d);reinterpret_cast<device half*>(dst)[1]=half(sum); }
}
template<typename Block,int QK,void Deq(device const Block*,short,thread float4x4&)>
kernel void poly_mul(constant QArgs& a [[buffer(0)]],uint3 group [[threadgroup_position_in_grid]],
                     uint lane [[thread_index_in_threadgroup]],uint3 shape [[threads_per_threadgroup]]) {
    uint row=group.y; ulong yrow=row;
    device const uchar* w=static_cast<device uchar*>(a.p[0]);
    device float* out_y=static_cast<device float*>(a.p[2]);
    if(a.kind==1) { w=static_cast<device const uchar* const*>(a.p[0])[static_cast<device int*>(a.p[3])[row]]+a.offset; }
    else if(a.kind==2 || a.kind==6) {
        if(a.kind==6 && a.p[4]) row+=static_cast<device int*>(a.p[4])[group.z];
        if(row>=uint(a.rows)) return;
        yrow=row;
        if(a.p[4]) { device const int* bounds=static_cast<device int*>(a.p[4]); if(row<uint(bounds[group.z]) || row>=uint(bounds[group.z+1])) return; }
        w+=ulong(group.z)*a.stride;
        if(a.p[3]) yrow=static_cast<device int*>(a.p[3])[row];
    }
    else if(a.kind==7) {
        device const int* bounds=static_cast<device int*>(a.p[4]);
        if(row<uint(bounds[0]) || row>=uint(bounds[a.count])) return;
        uint lo=0,hi=uint(a.count);
        while(lo+1<hi) {
            uint mid=(lo+hi)>>1;
            if(row<uint(bounds[mid])) hi=mid;
            else lo=mid;
        }
        w+=ulong(lo)*a.stride;
        if(a.p[3]) yrow=static_cast<device int*>(a.p[3])[row];
    }
    else if(a.kind==8) {
        w=static_cast<device const uchar*>(a.p[3*row]);
        out_y=static_cast<device float*>(a.p[3*row+2]);
        yrow=0;
    }
    ulong xrow=row;
    if(a.kind==4 || a.kind==5){
        if(group.z>=uint(*static_cast<device int*>(a.p[5])))return;
        device int* bounds=static_cast<device int*>(a.p[4]);
        if(row<uint(bounds[group.z]) || row>=uint(bounds[group.z+1]))return;
        w=static_cast<device const uchar* const*>(a.p[0])[group.z]+a.offset;
        if(a.kind==4)xrow=static_cast<device int*>(a.p[3])[row];
        else yrow=static_cast<device int*>(a.p[3])[row];
    }
    device const Block* weights=reinterpret_cast<device const Block*>(w)+ulong(group.x)*(a.ni/QK);
    device const uchar* x=a.kind==8
        ? static_cast<device const uchar*>(a.p[3*row+1])
        : static_cast<device uchar*>(a.p[1])+ulong(xrow/a.every)*a.xstride;
    float acc=0;
    for(int start=lane*16;start<a.ni;start+=32*16) {
        float4x4 deq; Deq(weights+start/QK,short((start%QK)/16),deq);
        if(a.kind==3){
            for(int j=0;j<16;++j) acc+=float(half(deq[j/4][j%4]))*float(reinterpret_cast<device const half*>(x)[start+j]);
            continue;
        }
        device const uchar* block=x+(start/32)*36;
        float d=float(*reinterpret_cast<device const half*>(block));
        for(int j=0;j<16;++j) acc+=deq[j/4][j%4]*(d*float(reinterpret_cast<device const char*>(block+4)[start%32+j]));
    }
    acc=simd_sum(acc);
    if(!lane) out_y[yrow*a.ystride+group.x]=acc;
}
#define MUL(name,QK) template [[host_name("poly_mul_" #name)]] kernel decltype(poly_mul<block_##name,QK,dequantize_##name<float4x4>>) poly_mul<block_##name,QK,dequantize_##name<float4x4>>;
MUL(q4_0,32) MUL(q4_1,32) MUL(q5_0,32) MUL(q5_1,32) MUL(q8_0,32) MUL(q2_0,64) MUL(mxfp4,32)
MUL(q2_K,256) MUL(q3_K,256) MUL(q4_K,256) MUL(q5_K,256) MUL(q6_K,256)
MUL(iq2_xxs,256) MUL(iq2_xs,256) MUL(iq3_xxs,256) MUL(iq1_s,256) MUL(iq4_nl,32)
MUL(iq3_s,256) MUL(iq2_s,256) MUL(iq4_xs,256) MUL(iq1_m,256)

template<typename Block,int QK,void Deq(device const Block*,short,thread float4x4&)>
kernel void q36_embed(constant QArgs& a [[buffer(0)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    uint col=(group.x*32+lane)*16;if(col>=uint(a.ni))return;
    int id=static_cast<device int*>(a.p[3])[group.y];
    device const Block* w=reinterpret_cast<device const Block*>(static_cast<device uchar*>(a.p[0])+ulong(id)*a.stride);
    float4x4 d;Deq(w+col/QK,short(col%QK/16),d);
    for(int j=0;j<16;++j)static_cast<device float*>(a.p[2])[ulong(group.y)*a.ni+col+j]=d[j/4][j%4];
}
#define EMB(name,QK) template [[host_name("q36_embed_" #name)]] kernel decltype(q36_embed<block_##name,QK,dequantize_##name<float4x4>>) q36_embed<block_##name,QK,dequantize_##name<float4x4>>;
EMB(q4_0,32) EMB(q4_1,32) EMB(q5_0,32) EMB(q5_1,32) EMB(q8_0,32) EMB(q2_0,64) EMB(iq4_nl,32) EMB(iq4_xs,256)
EMB(iq2_xxs,256) EMB(iq2_xs,256) EMB(iq3_xxs,256) EMB(iq1_s,256) EMB(iq3_s,256) EMB(iq2_s,256) EMB(iq1_m,256) EMB(mxfp4,32)
EMB(q2_K,256) EMB(q3_K,256) EMB(q4_K,256) EMB(q5_K,256) EMB(q6_K,256)
template [[host_name("q36_embed_f32")]] kernel decltype(q36_embed<float4x4,16,dequantize_f32<float4x4>>) q36_embed<float4x4,16,dequantize_f32<float4x4>>;
template [[host_name("q36_embed_f16")]] kernel decltype(q36_embed<half4x4,16,dequantize_f16<float4x4>>) q36_embed<half4x4,16,dequantize_f16<float4x4>>;
// bfloat vectors require Metal 3.1; decode BF16 without that device feature.
kernel void q36_embed_bf16(constant QArgs& a [[buffer(0)]],uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    uint col=group.x*32+lane;if(col>=uint(a.ni))return;int id=static_cast<device int*>(a.p[3])[group.y];
    device const ushort* w=reinterpret_cast<device const ushort*>(static_cast<device uchar*>(a.p[0])+ulong(id)*a.stride);
    static_cast<device float*>(a.p[2])[ulong(group.y)*a.ni+col]=as_type<float>(uint(w[col])<<16);
}
