#include <metal_stdlib>
using namespace metal;
struct Args { device void* p[12]; long pos; ulong stride; int n[16]; float f[8]; };
#define F(i) static_cast<device float*>(a.p[i])
#define I(i) static_cast<device int*>(a.p[i])
#define H(i) static_cast<device half*>(a.p[i])
#define KERNEL(name) kernel void name(constant Args& a [[buffer(0)]], uint2 g [[threadgroup_position_in_grid]], uint t [[thread_index_in_threadgroup]])
inline float sig(float x){return 1.f/(1.f+exp(-x));}
inline long position(device void* at,long p,int row,int step=1){return at?static_cast<device long*>(at)[2*row]:p+long(row)*step;}
inline float sum256(float v,threadgroup float* part,uint t){
    v=simd_sum(v);if(!(t%32))part[t/32]=v;threadgroup_barrier(mem_flags::mem_threadgroup);
    float s=0;for(int k=0;k<8;++k)s+=part[k];threadgroup_barrier(mem_flags::mem_threadgroup);return s;
}
// Rotary exponents are integers. A compensated product avoids libm pow's
// phase error amplification at long positions, without requiring GPU fp64.
inline float2 product_pair(float2 x,float2 y){
    float hi=x.x*y.x;
    float lo=fma(x.x,y.x,-hi)+x.x*y.y+x.y*y.x;
    float sum=hi+lo;return float2(sum,lo-(sum-hi));
}
inline float rotary_power(float x,int exponent){
    float2 value(1.f,0.f),power(x,0.f);
    while(exponent){if(exponent&1)value=product_pair(value,power);exponent>>=1;if(exponent)power=product_pair(power,power);}
    return value.x+value.y;
}
inline float2 yarn(float theta,constant Args& a,int pair,bool back=false){
    float ti=theta*a.f[1],ms=a.f[3];
    if(a.f[2]!=0){float y=(float(pair)-a.f[4])/max(.001f,a.f[5]-a.f[4]);float m=(1.f-clamp(y,0.f,1.f))*a.f[2];ti=ti*(1-m)+theta*m;ms*=1+.1f*log(1.f/a.f[1]);}
    return float2(precise::cos(ti)*ms,precise::sin(ti)*ms*(back?-1.f:1.f));
}
KERNEL(poly_hc_mix){
    int n=4*a.n[0], nm=a.n[1];device float* r=F(0)+ulong(g.y)*n;
    float v=0;for(int i=t;i<n;i+=32)v+=g.x<uint(nm)?r[i]*F(1)[ulong(g.x)*n+i]:r[i]*r[i];
    v=simd_sum(v);if(!t)F(4)[ulong(g.y)*(nm+1)+g.x]=v;
}
KERNEL(poly_hc_weights){
    if(t)return;device float* m=F(4)+ulong(g.x)*25;device float* w=F(5)+ulong(g.x)*20;
    float rs=rsqrt(m[24]/float(4*a.n[0])+a.f[0]),pre[4],comb[16];
    for(int h=0;h<4;++h){pre[h]=sig(m[h]*rs*F(2)[0]+F(3)[h])+a.f[1];w[h]=2*sig(m[4+h]*rs*F(2)[1]+F(3)[4+h]);}
    for(int src=0;src<4;++src){float mx=-INFINITY,sum=0;for(int dst=0;dst<4;++dst){int k=dst+4*src;comb[k]=m[8+k]*rs*F(2)[2]+F(3)[8+k];mx=max(mx,comb[k]);}
        for(int dst=0;dst<4;++dst){int k=dst+4*src;comb[k]=exp(comb[k]-mx);sum+=comb[k];}
        for(int dst=0;dst<4;++dst)comb[dst+4*src]=comb[dst+4*src]/sum+a.f[1];}
    for(int iter=0;iter<max(1,a.n[2]);++iter){
        if(iter)for(int src=0;src<4;++src){float sum=a.f[1];for(int dst=0;dst<4;++dst)sum+=comb[dst+4*src];for(int dst=0;dst<4;++dst)comb[dst+4*src]/=sum;}
        for(int dst=0;dst<4;++dst){float sum=a.f[1];for(int src=0;src<4;++src)sum+=comb[dst+4*src];for(int src=0;src<4;++src)comb[dst+4*src]/=sum;}}
    for(int k=0;k<16;++k)w[4+k]=comb[k];for(int k=0;k<4;++k)m[k]=pre[k];
}
KERNEL(poly_hc_read){
    int i=g.x*256+t,n=a.n[0],nm=a.n[1];if(i>=n)return;
    device float* m=F(4)+ulong(g.y)*(nm+1);float out=0,rs=nm==4?rsqrt(m[4]/float(4*n)+a.f[0]):1;
    for(int h=0;h<4;++h){float w=nm==4?sig(m[h]*rs*F(2)[0]+F(3)[h])+a.f[1]:m[h];out+=F(0)[(ulong(g.y)*4+h)*n+i]*w;}F(6)[ulong(g.y)*n+i]=out;
}
KERNEL(poly_hc_post){
    int i=g.x*256+t,n=a.n[0];if(i>=n)return;device float* r=F(2)+ulong(g.y)*4*n;device float* w=F(1)+ulong(g.y)*20;float in[4];for(int h=0;h<4;++h)in[h]=r[h*n+i];
    for(int d=0;d<4;++d){float v=F(0)[ulong(g.y)*n+i]*w[d];for(int h=0;h<4;++h)v+=in[h]*w[4+d+4*h];r[d*n+i]=v;}
}
KERNEL(poly_hc_mean){int i=g.x*256+t,n=a.n[0];if(i>=n)return;float v=0;for(int h=0;h<4;++h)v+=F(0)[(ulong(g.y)*4+h)*n+i];F(1)[ulong(g.y)*n+i]=v*.25f;}
KERNEL(poly_kda_conv){
    int c=g.x*256+t,n=a.n[0],rows=a.n[1];if(c>=3*n)return;int m=c/n,d=c%n;device float* w=static_cast<device float*>(a.p[2+m])+4*d;device float* st=F(1)+ulong(c)*3;
    float s0=st[0],s1=st[1],s2=st[2];for(int r=0;r<rows;++r){ulong at=(ulong(m)*rows+r)*n+d;float x=F(0)[at],v=s0*w[0]+s1*w[1]+s2*w[2]+x*w[3];s0=s1;s1=s2;s2=x;F(5)[at]=v*sig(v);
        if(r<a.n[2]){device float* snap=F(6)+ulong(r)*a.stride+ulong(c)*3;snap[0]=s0;snap[1]=s1;snap[2]=s2;}}
    st[0]=s0;st[1]=s1;st[2]=s2;
}
KERNEL(poly_kda_step){
    threadgroup float q[256],k[256],decay[256],o[256],norm[2];int d=a.n[0],heads=a.n[1],rows=a.n[2];ulong inner=ulong(d)*heads,plane=inner*rows;
    device float* col=F(4)+(ulong(g.x)*d+t)*d;
    for(int row=0;row<rows;++row){ulong at=ulong(row)*inner+g.x*d+t;q[t]=F(0)[at];k[t]=F(0)[plane+at];threadgroup_barrier(mem_flags::mem_threadgroup);
        if(!t){float qs=0,ks=0;for(int i=0;i<d;++i){qs+=q[i]*q[i];ks+=k[i]*k[i];}norm[0]=rsqrt(qs+1e-6f)*rsqrt(float(d));norm[1]=rsqrt(ks+1e-6f);}threadgroup_barrier(mem_flags::mem_threadgroup);
        q[t]*=norm[0];k[t]*=norm[1];decay[t]=exp(a.f[0]/(1.f+exp(F(1)[at]*F(2)[g.x])));threadgroup_barrier(mem_flags::mem_threadgroup);
        float sk=0;for(int i=0;i<d;++i)sk=fma(col[i]*decay[i],k[i],sk);
        float delta=(F(0)[2*plane+at]-sk)*sig(F(3)[ulong(row)*heads+g.x]),v=0;
        for(int i=0;i<d;++i){float next=fma(col[i],decay[i],k[i]*delta);col[i]=next;if(row<a.n[3])F(8)[ulong(row)*a.stride+(ulong(g.x)*d+t)*d+i]=next;v=fma(next,q[i],v);}o[t]=v;
        threadgroup_barrier(mem_flags::mem_threadgroup);if(!t){float ss=0;for(int i=0;i<d;++i)ss+=o[i]*o[i];norm[0]=rsqrt(ss/d+a.f[1]);}threadgroup_barrier(mem_flags::mem_threadgroup);
        F(7)[at]=v*norm[0]*F(6)[t]*sig(F(5)[at]);threadgroup_barrier(mem_flags::mem_threadgroup);
    }
}
KERNEL(poly_idx_store){int d=g.x*256+t,n=a.n[0];if(d>=n)return;ulong dst=ulong((a.pos+g.y)%a.n[1])*2*n;F(2)[dst+d]=F(0)[ulong(g.y)*n+d];F(2)[dst+n+d]=F(1)[ulong(g.y)*n+d];}
KERNEL(poly_idx_pool){
    int d=g.x*256+t,n=a.n[0],count=a.n[2];long last=a.pos+g.y;if(d>=n || (last+1)%count)return;
    float mx=-INFINITY;for(int k=0;k<count;++k){ulong at=ulong((last-count+1+k)%a.n[1])*2*n;mx=max(mx,F(0)[at+n+d]+F(2)[ulong(k)*n+d]);}
    float sum=0,out=0;for(int k=0;k<count;++k){ulong at=ulong((last-count+1+k)%a.n[1])*2*n;float e=exp(F(0)[at+n+d]+F(2)[ulong(k)*n+d]-mx);sum+=e;out+=e*F(0)[at+d];}F(1)[ulong(last/count)*n+d]=out/sum;
}
KERNEL(poly_idx_score){
    int dim=a.n[0],heads=a.n[1];long pos=position(a.p[4],a.pos,g.y);if(g.x>=ulong((pos+1)/a.n[2]))return;
    float v=0;for(int h=t;h<heads;h+=32){float dot=0;for(int d=0;d<dim;++d){float key=a.n[4]?float(H(2)[ulong(g.x)*dim+d]):F(2)[ulong(g.x)*dim+d];dot=fma(F(0)[(ulong(g.y)*heads+h)*dim+d],key,dot);}v+=max(dot,0.f)*F(1)[ulong(g.y)*heads+h]*a.f[0];}
    v=simd_sum(v);if(!t)F(3)[ulong(g.y)*a.n[3]+g.x]=v;
}
KERNEL(poly_select){
    int n=int((position(a.p[2],a.pos,g.x)+1)/a.n[3]),top=min(n,a.n[2]);device int* sel=I(1)+ulong(g.x)*a.n[1];
    if(a.n[4] && n<=top){for(int i=t;i<n;i+=32)sel[i]=i;return;}
    float previous=INFINITY;int before=-1;
    for(int rank=0;rank<top;++rank){float best=-INFINITY;int id=INT_MAX;
        for(int i=t;i<n;i+=32){float v=F(0)[ulong(g.x)*a.n[0]+i];if((v<previous || (v==previous && i>before)) && (v>best || (v==best && i<id))){best=v;id=i;}}
        float mx=simd_max(best);id=simd_min(best==mx?id:INT_MAX);previous=mx;before=id;
        if(!t){if(a.n[4]){int j=rank;while(j && sel[j-1]>id){sel[j]=sel[j-1];--j;}sel[j]=id;}
            else for(int k=0;k<a.n[3];++k)sel[rank*a.n[3]+k]=id*a.n[3]+k;}}
    if(!a.n[4])for(int i=t;i<(int(position(nullptr,a.pos,g.x))+1)%a.n[3];i+=32)sel[top*a.n[3]+i]=n*a.n[3]+i;
}
KERNEL(poly_dsa_attend){
    int dim=a.n[0],heads=a.n[1];long pos=a.pos+g.y;int count=g.y<uint(a.n[5])?int(pos+1):a.n[3]+int((pos+1)%a.n[4]);
    float q[64],acc[64];for(int j=0;j<64;++j){int d=t+j*32;q[j]=d<dim?F(0)[(ulong(g.y)*heads+g.x)*dim+d]:0;acc[j]=0;}
    float mx=-INFINITY,sum=0;for(int row=0;row<count;++row){int idx=g.y<uint(a.n[5])?row:I(2)[ulong(g.y)*a.n[2]+row];float dot=0;for(int j=0;j<64 && int(t+j*32)<dim;++j)dot=fma(q[j],float(H(1)[ulong(idx)*dim+t+j*32]),dot);dot=simd_sum(dot)*a.f[0];
        float next=max(mx,dot),old=exp(mx-next),w=exp(dot-next);sum=sum*old+w;for(int j=0;j<64 && int(t+j*32)<dim;++j)acc[j]=acc[j]*old+w*float(H(1)[ulong(idx)*dim+t+j*32]);mx=next;}
    for(int j=0;j<64 && int(t+j*32)<dim;++j)F(3)[(ulong(g.y)*heads+g.x)*dim+t+j*32]=acc[j]/sum;
}
KERNEL(poly_dsv4_rope){
    threadgroup float parts[8];int dim=a.n[0];ulong base=(ulong(g.y)*a.n[1]+g.x)*dim;float x0[4],x1[4],ss=0;
    for(int j=0;j<4;++j){int d=2*(t+256*j);x0[j]=d<dim?F(0)[base+d]:0;x1[j]=d<dim?F(0)[base+d+1]:0;ss+=x0[j]*x0[j]+x1[j]*x1[j];}
    float scale=a.n[5]?rsqrt(sum256(ss,parts,t)/dim+a.f[6]):1.f;threadgroup_barrier(mem_flags::mem_device);
    for(int j=0;j<4;++j){int d=2*(t+256*j);if(d>=dim)break;float x=x0[j]*scale,y=x1[j]*scale;
        if(d>=a.n[2] && d<a.n[2]+a.n[3]){int pair=(d-a.n[2])/2;float2 rot=yarn(float(position(a.p[2],a.pos,g.y,a.n[4]))*rotary_power(a.f[0],pair),a,pair,a.n[6]);float xx=x;x=x*rot.x-y*rot.y;y=xx*rot.y+y*rot.x;}
        F(1)[base+d]=x;F(1)[base+d+1]=y;}
}
KERNEL(poly_dsv4_state){int d=g.x*256+t,n=a.n[0];if(d>=n)return;long p=position(a.p[5],a.pos,g.y);ulong src=ulong(g.y)*n+d,dst=ulong(p%a.n[1])*n+d;F(3)[dst]=F(0)[src];F(4)[dst]=F(1)[src]+F(2)[ulong(p%a.n[2])*n+d];}
inline long compressed_src(int j,int d,long start,int dim,int ratio,int ring,bool overlap){long row=start+j;int col=d;if(overlap){row=start-ratio+j;if(j>=ratio)col+=dim;}return row<0?-1:(row%ring)*(overlap?2*dim:dim)+col;}
KERNEL(poly_dsv4_squeeze){
    threadgroup float vals[1024],parts[8];int dim=a.n[0],ratio=a.n[2];long p=position(a.p[4],a.pos,g.x);if((p+1)%ratio)return;long start=p+1-ratio;bool overlap=a.n[3];int count=ratio*(overlap?2:1);float local[4],ss=0;
    for(int d=t,jj=0;d<dim;d+=256,++jj){float mx=-INFINITY;for(int j=0;j<count;++j){long idx=compressed_src(j,d,start,dim,ratio,a.n[1],overlap);if(idx>=0)mx=max(mx,F(1)[idx]);}
        float sum=0;for(int j=0;j<count;++j){long idx=compressed_src(j,d,start,dim,ratio,a.n[1],overlap);if(idx>=0)sum+=exp(F(1)[idx]-mx);}
        float out=0;for(int j=0;j<count;++j){long idx=compressed_src(j,d,start,dim,ratio,a.n[1],overlap);if(idx>=0)out+=exp(F(1)[idx]-mx)/sum*F(0)[idx];}local[jj]=out;ss+=out*out;}
    float scale=rsqrt(sum256(ss,parts,t)/dim+a.f[6]);for(int d=t,j=0;d<dim;d+=256,++j)vals[d]=local[j]*scale*F(2)[d];threadgroup_barrier(mem_flags::mem_threadgroup);
    int off=dim-a.n[4];for(int d=t;d<dim;d+=256){float v=vals[d];if(d>=off){int pair=(d-off)/2;float2 rot=yarn(float(start)*rotary_power(a.f[0],pair),a,pair);float x=vals[off+2*pair],y=vals[off+2*pair+1];v=(d-off)%2?x*rot.y+y*rot.x:x*rot.x-y*rot.y;}H(3)[ulong(start/ratio)*dim+d]=half(v);}
}
KERNEL(poly_dsv4_attend){
    int dim=a.n[0],heads=a.n[1];long pos=position(a.p[6],a.pos,g.y);int raw=min(long(a.n[3]),pos+1),comp=a.p[2]?int((pos+1)/a.n[4]):0;if(a.p[3])comp=min(comp,a.n[6]);
    float q[64],acc[64];for(int j=0;j<64;++j){int d=t+j*32;q[j]=d<dim?F(0)[(ulong(g.y)*heads+g.x)*dim+d]:0;acc[j]=0;}
    float mx=F(4)[g.x],sum=1;for(int row=0;row<raw+comp;++row){int idx=row<raw?int((pos+1-raw+row)%a.n[2]):a.p[3]?I(3)[ulong(g.y)*a.n[5]+row-raw]:row-raw;device half* key=(row<raw?H(1):H(2))+ulong(idx)*dim;
        float dot=0;for(int j=0;j<64 && int(t+j*32)<dim;++j)dot=fma(q[j],float(key[t+j*32]),dot);dot=simd_sum(dot)*a.f[0];float next=max(mx,dot),old=exp(mx-next),w=exp(dot-next);sum=sum*old+w;
        for(int j=0;j<64 && int(t+j*32)<dim;++j)acc[j]=acc[j]*old+w*float(key[t+j*32]);mx=next;}
    for(int j=0;j<64 && int(t+j*32)<dim;++j)F(5)[(ulong(g.y)*heads+g.x)*dim+t+j*32]=acc[j]/sum;
}
KERNEL(poly_permute){int i=g.x*256+t,n=a.n[0],groups=a.n[1],rows=a.n[2];if(i>=n*groups*rows)return;int d=i%n,h=(i/n)%groups,r=i/(n*groups);ulong other=(ulong(h)*rows+r)*n+d;if(a.n[3])F(1)[i]=F(0)[other];else F(1)[other]=F(0)[i];}
