#include "test_utils.hpp"
#include "q36_launch.hpp"
int main(){
 int bad=0;g::Stream s;ck(g::stream_create(&s));
 int n=2048,rows=3;auto x=values(n*rows),w=values(n,.1f);for(auto&v:w)v+=1;
 Buffer dx(x),dw(w),dy(n*rows*4);std::vector<float> ref(x.size());
 for(int r=0;r<rows;++r){double sum=0;for(int i=0;i<n;++i)sum+=double(x[r*n+i])*x[r*n+i];float inv=1/std::sqrt(sum/n+1e-6);for(int i=0;i<n;++i)ref[r*n+i]=x[r*n+i]*w[i]*inv;}
 launch_k_rms(rows,256,s,dx.as(),dw.as(),dy.as(),n,1e-6);ck(g::stream_synchronize(s));
 bad+=compare(dy.read(x.size()),ref,"q36 RMS");

 // State update and gated normalization against a double-precision recurrence.
 const int C=512,HV=2,T=3;auto h=values(C*T,.04f),ga=values(HV*T,.1f),be=values(HV*T,.1f),z=values(HV*128*T,.3f);
 for(auto& v:ga)v=-.2f+v;for(auto& v:be)v+=.5f;
 auto state=values(128*HV*128,.01f);std::vector<float> gamma(128,1),expected(T*HV*128);std::vector<double> st(state.begin(),state.end());
 for(int t=0;t<T;++t)for(int head=0;head<HV;++head){double o[128],ss=0;double decay=std::exp(ga[t*HV+head]);
  for(int col=0;col<128;++col){double kv=0;for(int r=0;r<128;++r)kv+=st[(r*HV+head)*128+col]*h[t*C+128+r];
   double delta=(h[t*C+256+head*128+col]-decay*kv)*be[t*HV+head];o[col]=0;
   for(int r=0;r<128;++r){auto& v=st[(r*HV+head)*128+col];v=decay*v+h[t*C+128+r]*delta;o[col]+=v*h[t*C+r];}
   o[col]/=std::sqrt(128.);ss+=o[col]*o[col];}
  for(int col=0;col<128;++col){int i=(t*HV+head)*128+col;expected[i]=o[col]/std::sqrt(ss/128+1e-6)*z[i]/(1+std::exp(-z[i]));}}
 Buffer dh(h),dg(ga),db(be),dz(z),ds(state),dn(gamma),out(expected.size()*4);
 launch_k_gdn_seq(HV,dim3(128,2),s,ds.as(),dh.as(),C,dg.as(),db.as(),dz.as(),dn.as(),1e-6,out.as(),1,HV,T,true);
 ck(g::stream_synchronize(s));bad+=compare(out.read(expected.size()),expected,"q36 GDN output");
 bad+=compare(ds.read(state.size()),std::vector<float>(st.begin(),st.end()),"q36 GDN state");
 // The prompt column-split path must implement the same recurrence.
 ds.put(state.data(),state.size()*4);Buffer raw(expected.size()*4);
 launch_k_gdn_seq_cols(dim3(HV,4),dim3(32,16),s,ds.as(),dh.as(),C,dg.as(),db.as(),raw.as(),1,HV,T);
 launch_k_gdn_norm(dim3(HV,T),128,s,raw.as(),dz.as(),dn.as(),1e-6,out.as(),HV);
 ck(g::stream_synchronize(s));bad+=compare(out.read(expected.size()),expected,"q36 prompt GDN output");
 bad+=compare(ds.read(state.size()),std::vector<float>(st.begin(),st.end()),"q36 prompt GDN state");
 // Split-K attention across a 256-cell boundary and pointer-table sessions.
 const int ctx=512,P=270;auto q=values(256,.2f),kc=values(ctx*256,.15f),vc=values(ctx*256,.3f),gate=values(256,.4f);
 Buffer dq(q),dk(kc),dv(vc),dgate(gate),dp(4),pa(2*256*4),pm(8),pl(8),ao(256*4);
 int position=P-1;dp.put(&position,4);std::vector<float> ar(256);double den=0;
 for(int t=0;t<P;++t){double dot=0;for(int j=0;j<256;++j)dot+=q[j]*kc[t*256+j];double e=std::exp(dot/16);den+=e;for(int j=0;j<256;++j)ar[j]+=e*vc[t*256+j];}
 for(int j=0;j<256;++j)ar[j]/=den*(1+std::exp(-gate[j]));
 Buffer kt(8),vt(8),slot(4);auto kp=dk.as(),vp=dv.as();int zero=0;kt.put(&kp,8);vt.put(&vp,8);slot.put(&zero,4);
 RowSel rs;rs.slot=slot.as<int32_t>();rs.kc=kt.as<float*>();rs.vc=vt.as<float*>();
 launch_k_attn_part(dim3(2,1),256,s,dq.as(),nullptr,nullptr,dp.as<int32_t>(),0,1,1,ctx,1.f/16,2,pa.as(),pm.as(),pl.as(),rs);
 launch_k_attn_merge(1,256,s,pa.as(),pm.as(),pl.as(),dgate.as(),dp.as<int32_t>(),0,1,2,ao.as(),rs);
 ck(g::stream_synchronize(s));bad+=compare(ao.read(256),ar,"q36 split-K session attention");
 ck(g::stream_destroy(s));return bad?1:0;
}
