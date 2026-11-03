// Tiny sparse ready packs exercise actual bounded I/O, including the scale tail.
#include "engine/stream.h"
#include <filesystem>
#include <fstream>
#include <cstring>
#include <fcntl.h>
#include <unistd.h>
#include <cstdio>
int main(){
 char dir[]="/tmp/strata-ready-stream-XXXXXX";if(!mkdtemp(dir))return 2;
 struct Cleanup {const char* p;~Cleanup(){std::filesystem::remove_all(p);}} cleanup{dir};
 const std::string root=dir;std::ofstream(root+"/dense.gguf").put(0);
 constexpr int N=64,F=64,blob=3*N*F/64*36+16;
 mtl::Context cx;
 for(int ne:{256,512}){
  int fd=::open((root+"/experts.bin").c_str(),O_CREAT|O_TRUNC|O_RDWR,0600);
  if(fd<0||ftruncate(fd,int64_t(48)*ne*blob))return 2;
  std::vector<uint8_t> expected(blob);for(int i=0;i<blob;++i)expected[i]=uint8_t(i*31+7);
  float scales[]={.13f,1.7f,.002f,0};std::memcpy(expected.data()+blob-16,scales,16);
  if(pwrite(fd,expected.data(),blob,int64_t(48*ne-1)*blob)!=blob)return 2;
  auto index=[&](int layers){std::ofstream out(root+"/native_experts.txt");for(int l=0;l<layers;++l)out<<l<<" 40 40 "<<int64_t(l)*ne*blob<<" "<<blob<<"\n";};
  index(48);
  {gguf::File f;f.path=root+"/dense.gguf";strata::metal_engine::ExpertStream stream;std::string error;
   if(!stream.open(f,error,root,N,F,ne)||!stream.make_arena(cx,4*16384,"",4,error)){std::fprintf(stderr,"%s\n",error.c_str());return 1;}
   int32_t id=ne-1;stream.at_layer(47,1,1,&id,nullptr);
   uint32_t slot=stream.smap().as<uint32_t>()[47*ne+id];
   const auto* got=stream.chunk(slot>>28).as<uint8_t>()+(slot&0x0fffffff)*16384ull;
   if(std::memcmp(got,expected.data(),blob)||stream.arena_used_bytes()>4*16384)return 1;
  }
  // Reject a truncated final expert and a missing layer, before allocating an arena.
  ftruncate(fd,int64_t(48)*ne*blob-1);
  {gguf::File f;f.path=root+"/dense.gguf";strata::metal_engine::ExpertStream stream;std::string error;if(stream.open(f,error,root,N,F,ne))return 1;}
  ftruncate(fd,int64_t(48)*ne*blob);index(47);
  {gguf::File f;f.path=root+"/dense.gguf";strata::metal_engine::ExpertStream stream;std::string error;if(stream.open(f,error,root,N,F,ne))return 1;}
  close(fd);std::printf("ready pack %d experts: bounded read preserves bytes and scales; malformed spans rejected\n",ne);
 }
 return 0;
}
