#include "test_utils.hpp"
#include "strata/poly/kernels.hpp"
#include <algorithm>
#include <numeric>
using namespace strata::poly;
float sigmoid(float x){return 1/(1+std::exp(-x));}
int hyper(){
    constexpr int n=96,rows=3;auto r=values(rows*4*n),fn=values(24*4*n,.02f),scale=std::vector<float>{.7f,.8f,.9f},base=values(24,.3f);
    std::vector<float> mix(rows*25),w(rows*20),out(rows*n),post=r,mean(rows*n),head(rows*n);
    for(int row=0;row<rows;++row){float rs=0,m[24]{};for(int i=0;i<4*n;++i){float v=r[row*4*n+i];rs+=v*v;for(int k=0;k<24;++k)m[k]+=v*fn[k*4*n+i];}rs=1/std::sqrt(rs/(4*n)+1e-5f);
        float pre[4],comb[16];for(int h=0;h<4;++h){pre[h]=sigmoid(m[h]*rs*scale[0]+base[h])+1e-6f;w[row*20+h]=2*sigmoid(m[4+h]*rs*scale[1]+base[4+h]);}
        for(int h=0;h<4;++h){float mx=-INFINITY,sum=0;for(int d=0;d<4;++d){int k=d+4*h;comb[k]=m[8+k]*rs*scale[2]+base[8+k];mx=std::max(mx,comb[k]);}for(int d=0;d<4;++d){comb[d+4*h]=std::exp(comb[d+4*h]-mx);sum+=comb[d+4*h];}for(int d=0;d<4;++d)comb[d+4*h]=comb[d+4*h]/sum+1e-6f;}
        for(int it=0;it<5;++it){if(it)for(int h=0;h<4;++h){float sum=1e-6f;for(int d=0;d<4;++d)sum+=comb[d+4*h];for(int d=0;d<4;++d)comb[d+4*h]/=sum;}
            for(int d=0;d<4;++d){float sum=1e-6f;for(int h=0;h<4;++h)sum+=comb[d+4*h];for(int h=0;h<4;++h)comb[d+4*h]/=sum;}}
        for(int k=0;k<16;++k)w[row*20+4+k]=comb[k];
        for(int i=0;i<n;++i){for(int h=0;h<4;++h){out[row*n+i]+=r[(row*4+h)*n+i]*pre[h];mean[row*n+i]+=r[(row*4+h)*n+i]*.25f;}head[row*n+i]=out[row*n+i];
            for(int d=0;d<4;++d){float v=out[row*n+i]*w[row*20+d];for(int h=0;h<4;++h)v+=r[(row*4+h)*n+i]*comb[d+4*h];post[(row*4+d)*n+i]=v;}}
    }
    Buffer R(r),FN(fn),SC(scale),BA(base),M(rows*25*4),W(rows*20*4),O(rows*n*4),H(rows*n*4),ME(rows*n*4);
    hc_pre(R.as(),FN.as(),SC.as(),BA.as(),M.as(),W.as(),O.as(),n,rows,1e-5f,1e-6f,5,nullptr);
    int bad=compare(W.read(rows*20),w,"HC weights")+compare(O.read(rows*n),out,"HC read");
    hc_head(R.as(),FN.as(),SC.as(),BA.as(),M.as(),H.as(),n,rows,1e-5f,1e-6f,nullptr);
    hc_mean(R.as(),ME.as(),n,rows,nullptr);
    bad+=compare(H.read(rows*n),head,"HC head")+compare(ME.read(rows*n),mean,"HC mean");
    hc_post(O.as(),W.as(),R.as(),n,rows,nullptr);return bad+compare(R.read(r.size()),post,"HC write");
}
int recurrence(int dim,int rows){
    const int heads=2,n=dim*heads,snaps=std::min(rows,4),state_n=heads*dim*dim;const size_t plane=rows*n;
    auto x=values(3*plane,.2f),state=values(9*n,.1f),wq=values(4*n,.7f),wk=values(4*n,.6f),wv=values(4*n,.8f);auto refstate=state;
    std::vector<float> conv(x.size()),conv_snap(size_t(snaps)*state.size());
    for(int c=0;c<3*n;++c){int m=c/n,d=c%n;const auto& weights=m==0?wq:m==1?wk:wv;float* st=refstate.data()+c*3;
        for(int row=0;row<rows;++row){size_t at=(size_t(m)*rows+row)*n+d;float v=st[0]*weights[4*d]+st[1]*weights[4*d+1]+st[2]*weights[4*d+2]+x[at]*weights[4*d+3];conv[at]=v*sigmoid(v);st[0]=st[1];st[1]=st[2];st[2]=x[at];if(row<snaps)std::copy(st,st+3,conv_snap.data()+size_t(row)*state.size()+c*3);}}
    Buffer X(x),ST(state),WQ(wq),WK(wk),WV(wv),CV(conv.size()*4),CS(conv_snap.size()*4);
    kda_conv(X.as(),ST.as(),WQ.as(),WK.as(),WV.as(),CV.as(),n,rows,nullptr,CS.as(),state.size(),snaps);
    int bad=compare(CV.read(conv.size()),conv,"KDA conv")+compare(ST.read(state.size()),refstate,"KDA conv state")+compare(CS.read(conv_snap.size()),conv_snap,"KDA conv snapshots");
    auto g1=values(plane,.2f),a=values(heads,.2f),beta=values(rows*heads),s=values(state_n,.02f),g2=values(plane),norm=values(dim,.7f);auto expected=s;
    std::vector<float> out(plane),snap(size_t(snaps)*state_n);
    for(int row=0;row<rows;++row)for(int h=0;h<heads;++h){std::vector<float> q(dim),k(dim),decay(dim),o(dim);float qs=0,ks=0;
        for(int j=0;j<dim;++j){size_t at=size_t(row)*n+h*dim+j;q[j]=conv[at];k[j]=conv[plane+at];qs+=q[j]*q[j];ks+=k[j]*k[j];decay[j]=std::exp(-5.f/(1+std::exp(g1[at]*a[h])));}
        for(int j=0;j<dim;++j){q[j]=q[j]/std::sqrt(qs+1e-6f)/std::sqrt(float(dim));k[j]/=std::sqrt(ks+1e-6f);}
        float os=0;for(int j=0;j<dim;++j){float* col=expected.data()+(h*dim+j)*dim;float sk=0;for(int i=0;i<dim;++i)sk=std::fma(col[i]*decay[i],k[i],sk);float delta=(conv[2*plane+size_t(row)*n+h*dim+j]-sk)*sigmoid(beta[row*heads+h]);
            for(int i=0;i<dim;++i){col[i]=std::fma(col[i],decay[i],k[i]*delta);o[j]=std::fma(col[i],q[i],o[j]);}os+=o[j]*o[j];}
        for(int j=0;j<dim;++j){size_t at=size_t(row)*n+h*dim+j;out[at]=o[j]/std::sqrt(os/dim+1e-5f)*norm[j]*sigmoid(g2[at]);}
        if(row<snaps)std::copy(expected.begin()+h*dim*dim,expected.begin()+(h+1)*dim*dim,snap.begin()+size_t(row)*state_n+h*dim*dim);
    }
    Buffer G1(g1),A(a),B(beta),S(s),G2(g2),N(norm),O(out.size()*4),SN(snap.size()*4);
    kda_step(CV.as(),G1.as(),A.as(),B.as(),S.as(),G2.as(),N.as(),O.as(),dim,heads,rows,-5,1e-5,nullptr,SN.as(),state_n,snaps);
    return bad+compare(O.read(out.size()),out,"KDA outputs",2e-4)+compare(S.read(s.size()),expected,"KDA state",2e-4)+compare(SN.read(snap.size()),snap,"KDA snapshots",2e-4);
}
int indexer(int dim,int heads){
    const int rows=5,pos=11,kpool=4,ring=20,npool=4,top=2,stride=12;
    auto keys=values(ring*dim),gates=values(ring*dim,.3f),ape=values(kpool*dim,.2f),q=values(rows*heads*dim,.2f),weight=values(rows*heads);
    std::vector<float> pools(npool*dim),scores(rows*npool),zeros(ring*2*dim),expected(rows*stride,-1);
    for(int pool=0;pool<npool;++pool)for(int d=0;d<dim;++d){float mx=-INFINITY,sum=0,v=0;for(int k=0;k<kpool;++k)mx=std::max(mx,gates[(pool*kpool+k)*dim+d]+ape[k*dim+d]);for(int k=0;k<kpool;++k){float w=std::exp(gates[(pool*kpool+k)*dim+d]+ape[k*dim+d]-mx);sum+=w;v+=w*keys[(pool*kpool+k)*dim+d];}pools[pool*dim+d]=v/sum;}
    for(int row=0;row<rows;++row)for(int p=0;p<(pos+row+1)/kpool;++p){float v=0;for(int h=0;h<heads;++h){float dot=0;for(int d=0;d<dim;++d)dot=std::fma(q[(row*heads+h)*dim+d],pools[p*dim+d],dot);v+=std::max(dot,0.f)*weight[row*heads+h]/std::sqrt(float(dim*heads));}scores[row*npool+p]=v;}
    Buffer K(keys),G(gates),APE(ape),R(zeros),P(pools.size()*4),Q(q),W(weight),SC(std::vector<float>(scores.size(),0)),SEL(rows*stride*4);
    idx_store(K.as(),G.as(),R.as(),ring,dim,0,16,nullptr);idx_pool(R.as(),ring,P.as(),APE.as(),dim,kpool,0,16,nullptr);
    idx_score(Q.as(),W.as(),P.as(),SC.as(),npool,dim,heads,kpool,pos,rows,nullptr);int bad=compare(P.read(pools.size()),pools,"GLM pools")+compare(SC.read(scores.size()),scores,"GLM indexer scores");
    idx_select(SC.as(),npool,SEL.as<int>(),stride,top,kpool,pos,rows,nullptr);auto got=SEL.read<int>(rows*stride);
    for(int row=0;row<rows;++row){std::vector<int> order((pos+row+1)/kpool);std::iota(order.begin(),order.end(),0);std::stable_sort(order.begin(),order.end(),[&](int a,int b){return scores[row*npool+a]>scores[row*npool+b];});
        for(int i=0;i<top;++i)for(int k=0;k<kpool;++k)bad+=got[row*stride+i*kpool+k]!=order[i]*kpool+k;
        for(int k=0;k<(pos+row+1)%kpool;++k)bad+=got[row*stride+top*kpool+k]!=int(order.size())*kpool+k;}
    std::printf("GLM selection: %d total mismatches\n",bad);return bad;
}
int latent_attention(){
 const int dim=256,heads=3,rows=4,pos=15,kpool=4,top=8,stride=12,dense=1,cells=32;
 auto q=values(rows*heads*dim,.2f),cache=values(cells*dim,.4f);std::vector<_Float16> half(cache.begin(),cache.end());
 std::vector<int> sel(rows*stride);for(int r=0;r<rows;++r){for(int i=0;i<top;++i)sel[r*stride+i]=(i*3)%(pos+r+1);for(int i=0;i<(pos+r+1)%kpool;++i)sel[r*stride+top+i]=(pos+r+1)/kpool*kpool+i;}
 std::vector<float> ref(q.size());for(int r=0;r<rows;++r)for(int h=0;h<heads;++h){int count=r<dense?pos+r+1:top+(pos+r+1)%kpool;std::vector<double> sum(dim);double den=0;
  for(int i=0;i<count;++i){int cell=r<dense?i:sel[r*stride+i];double dot=0;for(int d=0;d<dim;++d)dot+=q[(r*heads+h)*dim+d]*float(half[cell*dim+d]);double e=std::exp(dot/16);den+=e;for(int d=0;d<dim;++d)sum[d]+=e*float(half[cell*dim+d]);}
  for(int d=0;d<dim;++d)ref[(r*heads+h)*dim+d]=sum[d]/den;
 }
 Buffer Q(q),C(half.size()*2),SEL(sel.size()*4),O(q.size()*4);C.put(half.data(),half.size()*2);SEL.put(sel.data(),sel.size()*4);
 dsa_attend(nullptr,Q.as(),C.as<uint16_t>(),SEL.as<int>(),stride,top,kpool,pos,rows,dense,nullptr,nullptr,32,O.as(),dim,heads,1.f/16,nullptr);
 return compare(O.read(q.size()),ref,"GLM dense/sparse latent attention");
}
int main(){try{return hyper()+recurrence(32,3)+recurrence(128,33)+recurrence(256,2)+indexer(128,3)+indexer(256,64)+latent_attention()?1:0;}catch(const std::exception& e){std::fprintf(stderr,"%s\n",e.what());return 2;}}
