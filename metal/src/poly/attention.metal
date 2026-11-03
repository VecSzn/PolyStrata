#include <metal_stdlib>
using namespace metal;
struct Args {
    device void* p[10]; long pos;
    int dim,heads,kv,rows,window,ring,whole,stride,rot,sections,kind;
    float scale,eps;
};
kernel void poly_store(constant Args& a [[buffer(0)]],uint i [[thread_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    if(i>=uint(a.dim)*a.rows) return;
    long pos=(a.p[3]?*static_cast<device long*>(a.p[3]):a.pos)+i/a.dim;
    if(a.ring) pos%=a.ring;
    ulong j=ulong(pos)*a.dim+i%a.dim;
    float x=static_cast<device float*>(a.p[0])[i];
    if(!a.kind) static_cast<device half*>(a.p[1])[j]=half(x);
    else {
        float d=simd_max(abs(x))/127.0f;
        static_cast<device char*>(a.p[1])[j]=char(d?round(x/d):0);
        if(!lane) static_cast<device half*>(a.p[2])[j/32]=half(d);
    }
}
kernel void poly_rope(constant Args& a [[buffer(0)]],uint head [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    device const float* x=static_cast<device float*>(a.p[0])+ulong(head)*a.stride;
    device const float* w=static_cast<device float*>(a.p[1]);
    device float* y=static_cast<device float*>(a.p[2])+ulong(head)*a.dim;
    float values[32]; float sum=0;
    for(int i=lane;i<a.dim;i+=32) { float v=x[i];sum+=v*v; }
    float inv=rsqrt(simd_sum(sum)/a.dim+a.eps);
    int token=head/a.heads,half_rot=a.rot/2;
    long own=(a.p[5]?*static_cast<device long*>(a.p[5]):a.pos)+token;
    for(int i=lane;i<a.dim;i+=32) {
        float value=x[i]*inv*(w?w[i]:1.0f);
        if(i<a.rot) {
            int j=i%half_rot;
            float position=a.p[4]?float(static_cast<device int*>(a.p[4])[token*3+(j%a.sections)%3]):float(own);
            float theta=position*pow(a.scale,float(j));
            if(a.p[3]) theta/=static_cast<device float*>(a.p[3])[j];
            float x0=x[j]*inv*(w?w[j]:1.0f),x1=x[j+half_rot]*inv*(w?w[j+half_rot]:1.0f);
            value=i<half_rot?x0*cos(theta)-x1*sin(theta):x0*sin(theta)+x1*cos(theta);
        }
        values[i/32]=value;
    }
    threadgroup_barrier(mem_flags::mem_device); // x and y may alias
    for(int i=lane;i<a.dim;i+=32) y[i]=values[i/32];
}
inline float cache_value(device void* raw,device void* codes,device void* scales,ulong at) {
    return codes?float(static_cast<device char*>(codes)[at])*float(static_cast<device half*>(scales)[at/32]):float(static_cast<device half*>(raw)[at]);
}
// One SIMD group per query head. Online softmax bounds scratch independently of context length.
kernel void poly_attention(constant Args& a [[buffer(0)]],uint2 group [[threadgroup_position_in_grid]],
                           uint lane [[thread_index_in_threadgroup]],uint2 shape [[threads_per_threadgroup]]) {
    int h=group.x,t=group.y,kh=h/(a.heads/a.kv);
    device const float* q=static_cast<device float*>(a.p[0])+(ulong(t)*a.heads+h)*a.dim;
    device float* out=static_cast<device float*>(a.p[3])+(ulong(t)*a.heads+h)*a.dim;
    long start=a.p[4]?*static_cast<device long*>(a.p[4]):a.pos,own=start+t;
    long lo=a.window>0?max(0L,own-a.window+1):0,hi=a.whole?start+a.rows-1:own;
    float acc[32]={0},query[32];
    for(int d=lane;d<a.dim;d+=32) query[d/32]=q[d];
    float maximum=-INFINITY,denom=0;
    for(long p=lo;p<=hi;++p) {
        ulong offset=(ulong(a.ring?p%a.ring:p)*a.kv+kh)*a.dim;
        float dot=0;
        for(int d=lane;d<a.dim;d+=32) dot+=query[d/32]*cache_value(a.p[1],a.p[5],a.p[6],offset+d);
        dot=simd_sum(dot)*a.scale;
        float next=max(maximum,dot),keep=exp(maximum-next),weight=exp(dot-next);
        denom=denom*keep+weight; maximum=next;
        for(int d=lane;d<a.dim;d+=32) acc[d/32]=acc[d/32]*keep+weight*cache_value(a.p[2],a.p[7],a.p[8],offset+d);
    }
    for(int d=lane;d<a.dim;d+=32) out[d]=acc[d/32]/denom;
}
