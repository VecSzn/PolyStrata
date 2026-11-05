#include <metal_stdlib>
using namespace metal;

// Mirrors BasicArgs in basic.cpp; device pointers are Metal gpuAddress values.
struct BasicArgs {
    device float* p[4];
    ulong count;
    int n, rows, stride, kind;
    float scale, eps;
};

inline float block_sum(float v, threadgroup float* partial, uint tid, uint nt) {
    for (uint k = 16; k; k >>= 1) v += simd_shuffle_xor(v, k);
    if ((tid & 31) == 0) partial[tid >> 5] = v;
    threadgroup_barrier(mem_flags::mem_threadgroup);
    float sum = 0;
    for (uint i = 0; i < nt / 32; ++i) sum += partial[i];
    threadgroup_barrier(mem_flags::mem_threadgroup);
    return sum;
}

kernel void poly_norm(constant BasicArgs& a [[buffer(0)]], uint row [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint nt [[threads_per_threadgroup]]) {
    threadgroup float partial[32];
    device const float* x = a.p[0] + ulong(row) * a.n;
    device float* y = a.p[2] + ulong(row) * (a.stride ? a.stride : a.n);
    float mean = 0;
    if (a.kind) {
        float s = 0;
        for (int i = tid; i < a.n; i += nt) s += x[i];
        mean = block_sum(s, partial, tid, nt) / a.n;
    }
    float ss = 0;
    for (int i = tid; i < a.n; i += nt) { float z = x[i] - mean; ss += z * z; }
    float inv = rsqrt(block_sum(ss, partial, tid, nt) / a.n + a.eps);
    for (int i = tid; i < a.n; i += nt)
        y[i] = (x[i] - mean) * inv * (a.p[1] ? a.p[1][i] : 1.0f) + (a.p[3] ? a.p[3][i] : 0.0f);
}

kernel void poly_gemv(constant BasicArgs& a [[buffer(0)]], uint2 group [[threadgroup_position_in_grid]],
                      uint tid [[thread_index_in_threadgroup]], uint2 shape [[threads_per_threadgroup]]) {
    const uint nt = shape.x;
    threadgroup float partial[32];
    const ulong o = group.x, token = group.y;
    device const float* x = a.p[1] + (token * (a.rows / a.stride) + o / a.stride) * a.n;
    float acc = 0;
    for (int i = tid; i < a.n; i += nt) {
        ulong j = o * a.n + i;
        float w = a.kind ? as_type<float>(uint(reinterpret_cast<device ushort*>(a.p[0])[j]) << 16) : a.p[0][j];
        acc += w * x[i];
    }
    float sum = block_sum(acc, partial, tid, nt);
    if (tid == 0) a.p[2][token * a.rows + o] = sum;
}

kernel void poly_elementwise(constant BasicArgs& a [[buffer(0)]], uint i [[thread_position_in_grid]]) {
    if (i >= a.count) return;
    if (a.kind == 0) a.p[1][i] = as_type<float>(uint(reinterpret_cast<device ushort*>(a.p[0])[i]) << 16);
    else if (a.kind == 1) a.p[1][i] = a.p[0][i] * a.scale;
    else if (a.kind == 2) a.p[1][i] += a.p[0][i] * a.scale;
    else if (a.kind == 3) a.p[0][i] = a.scale * tanh(a.p[0][i] / a.scale);
    else if (a.kind == 4) a.p[1][i] += a.p[0][i % a.n];
    else if (a.kind == 5 || a.kind == 6) {
        ulong at = a.p[1] ? i : (i / a.n) * 2 * a.n + i % a.n;
        float g = a.p[0][at], u = a.p[1] ? a.p[1][i] : a.p[0][at+a.n];
        if (a.kind == 5) { g=min(g,a.scale); u=clamp(u,-a.scale,a.scale); a.p[2][i]=g/(1.0f+exp(-g))*u; }
        else a.p[2][i]=0.5f*g*(1.0f+tanh(0.7978845608028654f*g*(1.0f+0.044715f*g*g)))*u;
    } else if (a.kind == 7) {
        ulong row=i/a.n, col=i%a.n; float sum=0;
        for (int k=0;k<a.stride;++k) sum += a.p[1][row*a.stride+k]*a.p[0][(row*a.stride+k)*a.n+col];
        a.p[2][i]+=sum;
    } else if (a.kind == 8) a.p[0][i] /= 1.0f+exp(-a.p[1][(i/a.n)*a.stride+(a.rows?0:i%a.n)]);
    else if (a.kind == 9) {
        int pair = i < uint(a.n) ? 0 : 2; ulong j = i < uint(a.n) ? i : i-a.n;
        reinterpret_cast<device uint*>(a.p[pair+1])[j]=reinterpret_cast<device uint*>(a.p[pair])[j];
    }

}

struct ConvArgs {
    device const float* x;
    device float* state;
    device const float* w;
    device float* out;
    device float* snap;
    ulong stride;
    int channels, rows, kind;
    uint snaps;
};
kernel void poly_conv(constant ConvArgs& a [[buffer(0)]], uint c [[thread_position_in_grid]]) {
    if (c >= uint(a.channels)) return;
    const int past = a.kind ? 3 : 2;
    device float* state = a.state + ulong(c) * past;
    device const float* w = a.w + ulong(c) * (past + 1);
    float s0 = state[0], s1 = state[1], s2 = a.kind ? state[2] : 0;
    for (int row = 0; row < a.rows; ++row) {
        device const float* x = a.x + ulong(row) * a.channels * (a.kind ? 1 : 3);
        float v = a.kind ? x[c] : x[c] * x[2 * a.channels + c];
        float y;
        if (a.kind) {
            y = s0*w[0] + s1*w[1] + s2*w[2] + v*w[3];
            s0=s1; s1=s2; s2=v;
            y = y / (1.0f + exp(-y));
        } else {
            y = (s0*w[0] + s1*w[1] + v*w[2]) * x[a.channels+c];
            s0=s1; s1=v;
        }
        a.out[ulong(row)*a.channels+c]=y;
        if (a.snap && row < 32 && ((a.snaps >> row) & 1u)) {
            device float* dst = a.snap + ulong(popcount(a.snaps & ((1u << row)-1u))) * a.stride + ulong(c)*past;
            dst[0]=s0; dst[1]=s1; if (a.kind) dst[2]=s2;
        }
    }
    state[0]=s0; state[1]=s1; if (a.kind) state[2]=s2;
}

struct RouteArgs {
    device const float* logits;
    device const float* bias;
    device const float* escale;
    device int* ids;
    device float* weights;
    int experts, used, rows, gate;
    float in_scale, w_scale;
    int norm;
};
kernel void poly_route(constant RouteArgs& a [[buffer(0)]], uint row [[threadgroup_position_in_grid]],
                       uint lane [[thread_index_in_simdgroup]]) {
    float p[16], pick[16], mx=-INFINITY;
    const int per = a.experts <= 64 ? 2 : a.experts <= 128 ? 4 : a.experts <= 256 ? 8 : 16;
    for(int k=0;k<per;++k) {
        int e=lane+32*k;
        p[k]=e<a.experts ? a.logits[ulong(row)*a.experts+e]*a.in_scale : -INFINITY;
        mx=max(mx,p[k]);
    }
    if(a.gate==0) {
        for(int o=16;o;o>>=1) mx=max(mx,simd_shuffle_xor(mx,o));
        float sum=0;
        for(int k=0;k<per;++k) { p[k]=int(lane)+32*k<a.experts ? exp(p[k]-mx):0; sum+=p[k]; }
        for(int o=16;o;o>>=1) sum+=simd_shuffle_xor(sum,o);
        const float inv=1.0f/sum;
        for(int k=0;k<per;++k) p[k]*=inv;
    } else {
        for(int k=0;k<per;++k) { float x=p[k]; p[k]=a.gate==1 ? 1.0f/(1.0f+exp(-x)) : sqrt(x>20 ? x:log(1.0f+exp(x))); }
    }
    for(int k=0;k<per;++k) {
        int e=lane+32*k;
        pick[k]=e<a.experts ? p[k]+(a.bias?a.bias[e]:0): -INFINITY;
    }
    float sum=0, mine=0; int my_id=0;
    for(int j=0;j<a.used;++j) {
        float best=-INFINITY; int at=a.experts;
        for(int k=0;k<per;++k) if(pick[k]>best) {best=pick[k];at=lane+32*k;}
        for(int o=16;o;o>>=1) {
            float b=simd_shuffle_xor(best,o); int id=simd_shuffle_xor(at,o);
            if(b>best || (b==best && id<at)) {best=b;at=id;}
        }
        float pw=0;
        for(int k=0;k<per;++k) if(int(lane)+32*k==at) {pw=p[k];pick[k]=-INFINITY;}
        pw=simd_shuffle(pw,at%32); sum+=pw;
        if(int(lane)==j) {mine=pw;my_id=at;}
    }
    if(int(lane)>=a.used) return;
    if(a.norm) mine/=max(sum,6.103515625e-5f);
    if(a.escale) mine*=a.escale[my_id];
    if(a.w_scale!=0) mine*=a.w_scale;
    a.ids[ulong(row)*a.used+lane]=my_id;
    a.weights[ulong(row)*a.used+lane]=mine;
}

struct ScaledArgs { device const float* x[32]; device float* acc[32]; float w[32]; int count,n; };
kernel void poly_scaled(constant ScaledArgs& a [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    if(i>=uint(a.n)) return;
    for(int k=0;k<a.count;++k) a.acc[k][i]+=a.x[k][i]*a.w[k];
}
kernel void poly_likeliest(constant BasicArgs& a [[buffer(0)]],uint lane [[thread_index_in_threadgroup]]) {
    float m=-INFINITY;
    for(int i=lane;i<a.n;i+=32) m=max(m,a.p[0][i]);
    m=simd_max(m); float sum=0; int at=a.n;
    for(int i=lane;i<a.n;i+=32) { float v=a.p[0][i]; if(v==m) at=min(at,i); if(v>m-20) sum+=exp(v-m); }
    sum=simd_sum(sum); at=simd_min(at);
    if(!lane) { reinterpret_cast<device int*>(a.p[1])[0]=at<a.n?at:0; a.p[1][1]=1.0f/sum; }
}
