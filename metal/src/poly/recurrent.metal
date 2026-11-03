#include <metal_stdlib>
using namespace metal;
struct Args { device float* p[10]; ulong stride; int dim,heads,keys,rows; float eps; uint snaps; };
inline float warp_sum(float v) { for(uint k=16;k;k>>=1) v+=simd_shuffle_xor(v,k);return v; }
kernel void poly_gdn_norm(constant Args& a [[buffer(0)]],uint head [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    ulong row_n=ulong(2*a.keys+a.heads)*a.dim;
    device float* q=a.p[0]+(head/a.keys)*row_n+(head%a.keys)*a.dim;
    device float* k=q+a.keys*a.dim;
    float qs=0,ks=0;
    for(int i=lane;i<a.dim;i+=32) { qs+=q[i]*q[i];ks+=k[i]*k[i]; }
    float qi=rsqrt(warp_sum(qs)+a.eps)*rsqrt(float(a.dim)),ki=rsqrt(warp_sum(ks)+a.eps);
    for(int i=lane;i<a.dim;i+=32) { q[i]*=qi;k[i]*=ki; }
}
kernel void poly_gdn_gates(constant Args& a [[buffer(0)]],uint i [[thread_position_in_grid]]) {
    if(i>=uint(a.rows*a.heads)) return;
    float g=a.p[1][i]+a.p[3][i%a.heads];
    a.p[1][i]=exp((g>20?g:log(1.0f+exp(g)))*a.p[4][i%a.heads]);
    a.p[2][i]=1.0f/(1.0f+exp(-a.p[2][i]));
}
kernel void poly_gdn_state(constant Args& a [[buffer(0)]],uint col [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    int h=col/a.dim,j=col%a.dim,hk=h%a.keys;
    ulong keys=ulong(a.keys)*a.dim,inner=ulong(a.heads)*a.dim,mine=ulong(col)*a.dim;
    float state[8];
    for(int i=lane;i<a.dim;i+=32) state[i/32]=a.p[5][mine+i];
    for(int row=0;row<a.rows;++row) {
        device const float* x=a.p[0]+ulong(row)*(2*keys+inner);
        device const float* q=x+hk*a.dim; device const float* k=q+keys;
        float decay=a.p[1][row*a.heads+h],sk=0;
        for(int i=lane;i<a.dim;i+=32) sk+=state[i/32]*decay*k[i];
        float delta=(x[2*keys+h*a.dim+j]-warp_sum(sk))*a.p[2][row*a.heads+h],o=0;
        bool save=a.p[9] && row<32 && ((a.snaps>>row)&1u);
        ulong snap_at=save?ulong(popcount(a.snaps&((1u<<row)-1)))*a.stride+mine:0;
        for(int i=lane;i<a.dim;i+=32) {
            float v=state[i/32]*decay+k[i]*delta; state[i/32]=v;o+=v*q[i];
            if(save) a.p[9][snap_at+i]=v;
        }
        o=warp_sum(o); if(!lane) a.p[8][ulong(row)*inner+h*a.dim+j]=o;
    }
    for(int i=lane;i<a.dim;i+=32) a.p[5][mine+i]=state[i/32];
}
kernel void poly_gdn_out(constant Args& a [[buffer(0)]],uint head [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    ulong base=ulong(head)*a.dim;float ss=0;
    for(int i=lane;i<a.dim;i+=32) {float v=a.p[8][base+i];ss+=v*v;}
    float inv=rsqrt(warp_sum(ss)/a.dim+a.eps);
    for(int i=lane;i<a.dim;i+=32) {float z=a.p[6][base+i];a.p[8][base+i]*=inv*a.p[7][i]*(z/(1.0f+exp(-z)));}
}
