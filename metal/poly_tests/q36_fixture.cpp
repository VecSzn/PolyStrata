// Small synthetic topology, including native head/state widths and in-file MTP.
#include "ggml.h"
#include "gguf.h"
#include "strata/core/family_layout.hpp"
#include <cmath>
int main(int argc,char** argv){
 if(argc<2||argc>3)return 2;
 const ggml_type qt=argc==3&&std::string(argv[2])=="q4"?GGML_TYPE_Q4_K:argc==3&&std::string(argv[2])=="q6"?GGML_TYPE_Q6_K:GGML_TYPE_Q8_0;
 auto* ctx=ggml_init({32*1024*1024,nullptr,false});auto* file=gguf_init_empty();
 gguf_set_val_str(file,"general.architecture","qwen35moe");
 auto num=[&](const char* k,int n){gguf_set_val_u32(file,(std::string("qwen35moe.")+k).c_str(),n);};
 for(auto kv:std::vector<std::pair<const char*,int>>{{"block_count",3},{"embedding_length",256},{"attention.head_count",1},
 {"attention.head_count_kv",1},{"attention.key_length",256},{"attention.value_length",256},{"expert_count",4},{"expert_used_count",2},
 {"expert_feed_forward_length",256},{"expert_shared_feed_forward_length",256},{"ssm.conv_kernel",4},{"ssm.state_size",128},
 {"ssm.group_count",1},{"ssm.time_step_rank",2},{"ssm.inner_size",256},{"full_attention_interval",2},{"nextn_predict_layers",1},{"rope.dimension_count",64}})num(kv.first,kv.second);
 std::vector<std::string> names(256);std::vector<const char*> tokens(256);for(int i=0;i<256;++i){names[i]=std::to_string(i);tokens[i]=names[i].c_str();}
 gguf_set_arr_str(file,"tokenizer.ggml.tokens",tokens.data(),tokens.size());
 strata::core::ModelGeometry g;g.family=strata::ModelFamily::Qwen35Moe;g.n_layers=2;g.n_mtp_layers=1;g.n_embd=256;g.n_vocab=256;
 g.n_expert=4;g.n_expert_used=2;g.n_ff=g.n_ff_shexp=256;g.hc=g.hc_lr=0;g.idx_q_heads=g.idx_key_dim=0;
 g.n_head=g.n_head_kv=1;g.head_dim=256;g.ssm_state_size=128;g.ssm_k_heads=1;g.ssm_v_heads=2;g.ssm_value_dim=256;g.ssm_conv_channels=512;g.ssm_d_conv=4;g.qsa_interval=2;
 auto add=[&](std::string name,std::vector<int64_t> shape){
 bool norm=name.find("norm.weight")!=std::string::npos;
 bool dense=name.find("ffn_gate_inp")!=std::string::npos||name.find("ssm_alpha")!=std::string::npos||name.find("ssm_beta")!=std::string::npos||name.find("ssm_conv")!=std::string::npos;
 bool quant=shape.size()>=2&&!dense;
 auto* t=ggml_new_tensor(ctx,quant?qt:GGML_TYPE_F32,shape.size(),shape.data());ggml_set_name(t,name.c_str());
 std::vector<float> v(ggml_nelements(t));for(size_t i=0;i<v.size();++i)v[i]=norm?1:.025f*std::sin(i*.079f+name.size());
 if(quant)ggml_quantize_chunk(qt,v.data(),t->data,0,v.size()/shape[0],shape[0],nullptr);else std::copy(v.begin(),v.end(),static_cast<float*>(t->data));
 gguf_add_tensor(file,t);
 };
 for(auto t:strata::core::global_tensors(g))add(t.name,t.shape);
 for(int l=0;l<3;++l)for(auto t:strata::core::layer_tensors(g,l))add("blk."+std::to_string(l)+"."+t.name,t.shape);
 bool ok=gguf_write_to_file(file,argv[1],false);gguf_free(file);ggml_free(ctx);return ok?0:1;
}
