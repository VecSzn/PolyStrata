#include "test_utils.hpp"
#include "strata/poly/kernels.hpp"
namespace p=strata::poly;
int main(){
    int bad=0;g::Stream stream;ck(g::stream_create(&stream));
    for(int dim:{32,64,128,256}){
        const int heads=4,keys=2,rows=5;const size_t row_n=(2*keys+heads)*dim,state_n=heads*dim*dim,stride=state_n+17;const unsigned snaps=0x15;
        auto qkv=values(rows*row_n,.7f),alpha=values(rows*heads,.4f),beta=values(rows*heads,.2f),dt=values(heads,.1f),decay=values(heads,.1f),state=values(state_n,.03f),z=values(rows*heads*dim,.8f),norm=values(dim,.1f);
        for(auto& v:decay)v=-.4f-std::abs(v);for(auto& v:norm)v+=1;
        Buffer dq(qkv),da(alpha),db(beta),dd(dt),ddec(decay),dstate(state),dz(z),dn(norm),out(rows*heads*dim*4),snap(stride*3*4);
        std::vector<float> ref(rows*heads*dim),snapref(stride*3,0);
        int saved=0;
        for(int r=0;r<rows;++r){
            float* x=qkv.data()+r*row_n;
            for(int h=0;h<keys;++h){double qs=0,ks=0;for(int i=0;i<dim;++i){qs+=double(x[h*dim+i])*x[h*dim+i];ks+=double(x[(keys+h)*dim+i])*x[(keys+h)*dim+i];}
                for(int i=0;i<dim;++i){x[h*dim+i]/=std::sqrt(qs+1e-5)*std::sqrt(double(dim));x[(keys+h)*dim+i]/=std::sqrt(ks+1e-5);}}
            for(int h=0;h<heads;++h){double g=alpha[r*heads+h]+dt[h];alpha[r*heads+h]=std::exp(std::log1p(std::exp(g))*decay[h]);beta[r*heads+h]=1/(1+std::exp(-beta[r*heads+h]));
                for(int j=0;j<dim;++j){float* col=state.data()+(h*dim+j)*dim;double sk=0;for(int i=0;i<dim;++i)sk+=double(col[i])*alpha[r*heads+h]*x[(keys+h%keys)*dim+i];
                    double d=(x[(2*keys+h)*dim+j]-sk)*beta[r*heads+h],o=0;
                    for(int i=0;i<dim;++i){col[i]=col[i]*alpha[r*heads+h]+x[(keys+h%keys)*dim+i]*d;o+=double(col[i])*x[(h%keys)*dim+i];}ref[(r*heads+h)*dim+j]=o;}
                double ss=0;for(int j=0;j<dim;++j){double v=ref[(r*heads+h)*dim+j];ss+=v*v;}
                for(int j=0;j<dim;++j){size_t at=(r*heads+h)*dim+j;ref[at]*=norm[j]/std::sqrt(ss/dim+1e-5)*(z[at]/(1+std::exp(-z[at])));}
            }
            if((snaps>>r)&1u){std::copy(state.begin(),state.end(),snapref.begin()+saved*stride);++saved;}
        }
        ck(g::memset_async(snap.ptr,0,stride*3*4,stream));
        p::gdn_step(dq.as(),da.as(),db.as(),dd.as(),ddec.as(),dstate.as(),dz.as(),dn.as(),out.as(),dim,heads,keys,rows,1e-5f,stream,snap.as(),stride,snaps);ck(g::stream_synchronize(stream));
        bad+=compare(out.read(ref.size()),ref,"GDN output",2e-4f);bad+=compare(dstate.read(state.size()),state,"GDN state");bad+=compare(snap.read(snapref.size()),snapref,"GDN snapshots");
        bad+=compare(dq.read(qkv.size()),qkv,"GDN normalized Q/K");bad+=compare(da.read(alpha.size()),alpha,"GDN decay");
    }
    // The fourth convolution tap and sparse snapshots, used by Qwen's GDN.
    int channels=37,rows=7;auto x=values(channels*rows),w=values(channels*4,.3f),state=values(channels*3,.1f);auto ref=values(x.size(),0),snapref=values(channels*3*3,0);
    Buffer dx(x),dw(w),ds(state),out(x.size()*4),snap(snapref.size()*4);unsigned mask=0x49;int saved=0;
    for(int r=0;r<rows;++r){for(int c=0;c<channels;++c){float* st=state.data()+c*3;float v=st[0]*w[c*4]+st[1]*w[c*4+1]+st[2]*w[c*4+2]+x[r*channels+c]*w[c*4+3];ref[r*channels+c]=v/(1+std::exp(-v));st[0]=st[1];st[1]=st[2];st[2]=x[r*channels+c];}if((mask>>r)&1){std::copy(state.begin(),state.end(),snapref.begin()+saved*state.size());++saved;}}
    p::conv4_silu(dx.as(),ds.as(),dw.as(),out.as(),channels,rows,stream,snap.as(),state.size(),mask);ck(g::stream_synchronize(stream));bad+=compare(out.read(ref.size()),ref,"conv4 output");bad+=compare(snap.read(snapref.size()),snapref,"conv4 snapshots");
    ck(g::stream_destroy(stream));return bad?1:0;
}
