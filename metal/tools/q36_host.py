"""Adapt strata-q36's host driver and portable kernels to native Metal at build time.

The original CUDA/HIP source stays authoritative for scheduling, MTP, parallel
sessions and protocol. CUDA-only fused optimizations use that driver's existing
unfused paths. This adapter is intentionally specific to q36_engine.cu; unknown
launches fail the build rather than silently falling back to CPU execution.
"""
from pathlib import Path
import re
import sys


def clean(text):
    return re.sub(r'"(?:\\.|[^"\\])*"|\'(?:\\.|[^\'\\])*\'|//[^\n]*|/\*.*?\*/',
                  lambda m: m[0] if m[0][0] in '\"\'' else '\n'*m[0].count('\n'),text,flags=re.S)


def end(text,start,left='(',right=')'):
    depth=0
    for i in range(start,len(text)):
        if text[i]==left: depth+=1
        if text[i]==right:
            depth-=1
            if depth==0:return i
    raise ValueError('unbalanced source')


def body(text,name,replacement):
    match=re.search(r'\b'+name+r'\s*\([^;{}]*\)\s*(?:const\s*)?\{',text)
    if not match:raise ValueError('missing host method '+name)
    begin=match.end()-1;finish=end(text,begin,'{','}')
    return text[:begin+1]+'\n'+replacement+'\n'+text[finish:]


def split_args(text):
    result=[];start=0;depth=0
    for i,c in enumerate(text):
        if c in '(<[{':depth+=1
        if c in ')>]}':depth-=1
        if c==',' and depth==0:result.append(text[start:i].strip());start=i+1
    result.append(text[start:].strip())
    return result


def metal_type(t):
    t=t.replace('__restrict__','').strip()
    t=re.sub(r'\bint32_t\b','int',t);t=re.sub(r'\buint8_t\b','uchar',t)
    t=re.sub(r'\bint64_t\b','long',t);t=re.sub(r'\buint64_t\b','ulong',t)
    t=t.replace('unsigned long long','ulong').replace('__half','half')
    if t.count('*') == 2: return 'device ulong*'
    if '*' in t:
        t='device '+t
        # The pointer table and its pointees both live in GPU buffers.
        t=re.sub(r'\*\s*(const\s*)?\*',lambda m:'* '+(m[1] or '')+' device *',t)
    return t


def main():
    src=clean(Path(sys.argv[1]).read_text());out=Path(sys.argv[2]);out.mkdir(parents=True,exist_ok=True)
    begin=src.index('__device__');finish=src.index('struct Mat {')
    device=src[begin:finish]
    host=src[:begin]+src[finish:]
    host=host.replace('#include "q36_kernels.cuh"','#include "q36_launch.hpp"').replace('#include <cublas_v2.h>','')
    host=host.replace('cublasHandle_t blas = nullptr;','')
    a=host.index('        if (cublasCreate(');b=end(host,host.index('{',a),'{','}')
    host=host[:a]+host[b+1:]
    host=host.replace('bool fast_k = std::getenv("STRATA_Q36_OLD_KERNELS") == nullptr;','bool fast_k = false;')
    for method in ('q8_small','shared_gu','route_group'):
        host=body(host,method,'return false;')
    host=body(host,'group_entries','group_old(L, slot_of, T, grp_ptr, grp_start, n_groups, ent_dst, ent_tok);')
    host=body(host,'add_rms_post','xq_post=false; k_add_rms<<<T,kThreads,0,st>>>(x,y,L.post_norm,h,(int)m->g.n_embd,m->eps);')
    host=body(host,'moe_end','xq_ready=false; k_moe_out<<<T,kThreads,0,st>>>(x,eparts,prow,ewts,(int)m->top_k,shared,sgate,sgs,wn,hout,(int)m->g.n_embd,m->eps);')
    host=body(host,'f32mm','k_gemv_f32<<<dim3(rows8(n_out),T),kThreads,0,st>>>(W,xin,out,n_in,n_out);')
    # Use the driver's split-K attention for prompt chunks too. It keeps the
    # full FP32 cache and the same positional / multi-session semantics.
    a=host.index('            if (T > kGemmT) {',host.index('            RowSel rs;'))
    b=end(host,host.index('{',a),'{','}');assert host[b+1:].lstrip().startswith('else {')
    c=host.index('{',b+1);d=end(host,c,'{','}')
    host=host[:a]+host[c:d+1]+host[d+1:]
    host=host.replace('prompt experts on MMQ (int8)', 'prompt experts on native Metal quantized products')
    host=body(host,'head_draft',"""
        int count=draft_n>0?std::min<int64_t>(draft_n,m->g.n_vocab):m->g.n_vocab;
        quant(hn,m->g.n_embd,1,q8);
        K::native_mmvq(m->head.type,m->head.d,q8,logits_m,m->g.n_embd,count,1,st);
        k_argmax<<<dim3(kArgBlocks,1),kThreads,0,st>>>(logits_m,count,pick_m,amv,ami,false);
        k_argmax<<<dim3(1,1),kThreads,0,st>>>(logits_m,kArgBlocks,pick_m,amv,ami,true);
    """)
    # Launch dimensions are explicit and retain CUDA's x/y thread ordering.
    host=host.replace('dim3(128, 4)', 'dim3(128, 2)')
    used=set()
    pattern=r'\b(k_\w+)\s*<<<(.*?)>>>\s*\('
    def launch(m):
        used.add(m[1]);g,b,shared,st=split_args(m[2]);assert shared=='0'
        return f'launch_{m[1]}({g},{b},{st},'
    host=re.sub(pattern,launch,host,flags=re.S)
    if '<<<' in host or 'q36k::' in host or 'cublas' in host:raise ValueError('unadapted CUDA host operation: '+str([line for line in host.splitlines() if '<<<' in line or 'q36k::' in line or 'cublas' in line]))
    kernels={}
    for m in re.finditer(r'__global__\s+void\s+(?:__launch_bounds__\(\d+\)\s*)?(k_\w+)\s*\(',device):
        a=m.end()-1;b=end(device,a);c=device.index('{',b);d=end(device,c,'{','}')
        if m[1] in used:kernels[m[1]]=(device[a+1:b],device[c+1:d])
    if used!=kernels.keys():raise ValueError('missing kernel definitions '+str(used-kernels.keys()))
    rowsel=device[device.index('struct RowSel {'):device.index('};',device.index('struct RowSel {'))+2]
    constants='\n'.join(re.findall(r'^constexpr int k(?:ArgBlocks|GdnColGroups) = .*?;',device,re.M))
    header=['#pragma once','#include "q36_runtime.hpp"',constants,rowsel,
            'inline unsigned rows8(int n){return unsigned((n+7)/8);}',
            'inline unsigned grid1(int64_t n,int b=256){return unsigned((n+b-1)/b);}']
    shader=['#include <metal_stdlib>','using namespace metal;','constant int kThreads=256,kChunkCells=256,kConvTile=16;']
    shader.append(constants.replace('constexpr','constant'))
    shader.append('struct RowSel {device const int* cell;device const int* slot;device const int* delta;device ulong* kc;device ulong* vc;};')
    shader.append('''
inline float silu(float x){return x/(1+exp(-x));}
inline float sigm(float x){return 1/(1+exp(-x));}
inline float softplus(float x){return x>20?x:log(1+exp(x));}
inline float block_sum(float v,threadgroup float* sh,uint3 tid,uint3 size){
 sh[tid.x]=v;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint s=size.x/2;s;s>>=1){if(tid.x<s)sh[tid.x]+=sh[tid.x+s];threadgroup_barrier(mem_flags::mem_threadgroup);}
 float r=sh[0];threadgroup_barrier(mem_flags::mem_threadgroup);return r;}
inline float block_max(float v,threadgroup float* sh,uint3 tid,uint3 size){
 sh[tid.x]=v;threadgroup_barrier(mem_flags::mem_threadgroup);
 for(uint s=size.x/2;s;s>>=1){if(tid.x<s)sh[tid.x]=max(sh[tid.x],sh[tid.x+s]);threadgroup_barrier(mem_flags::mem_threadgroup);}
 float r=sh[0];threadgroup_barrier(mem_flags::mem_threadgroup);return r;}
inline int atomicAdd(threadgroup int* p,int v){return atomic_fetch_add_explicit((threadgroup atomic_int*)p,v,memory_order_relaxed);}
inline int atomicAdd(device int* p,int v){return atomic_fetch_add_explicit((device atomic_int*)p,v,memory_order_relaxed);}
''')
    for name,(signature,code) in kernels.items():
        if name == 'k_gdn_seq':
            code=code.replace('RG = 4, RPG = 32','RG = 2, RPG = 64')
            code=code.replace(' + red_kv[2][col] + red_kv[3][col]','').replace(' + red_o[2][col] + red_o[3][col]','')
        args=split_args(signature);fields=[];types=[];names=[]
        for arg in args:
            bare=arg.split('=')[0].strip();m=re.search(r'(\w+)\s*$',bare);names.append(m[1]);types.append(bare[:m.start()].strip());fields.append(bare+';')
        header.append('inline void launch_'+name+'(dim3 grid,dim3 block,void* stream,'+signature+') {\nstruct Args {'+''.join(fields)+'};\nArgs a{'+','.join(names)+'};\nq36_dispatch("'+name+'",a,grid,block,stream);\n}')
        shader.append('struct Args_'+name+' {'+''.join(metal_type(t)+' '+n+';' for t,n in zip(types,names))+'};')
        code=re.sub(r'\bhalf\b','rot_half',code)
        code=code.replace('unsigned long long','ulong').replace('int64_t','long')
        code=code.replace('sincosf(theta, &sn, &cs);','sn=sin(theta);cs=cos(theta);')
        for table,index in [('rs.kc','rs.slot[t]'),('rs.vc','rs.slot[t]'),('hist_tab','slot_rows[g]'),('state_tab','slot_rows[g]')]:
            code=code.replace(table+'['+index+']','reinterpret_cast<device float*>('+table+'['+index+'])')
        code=code.replace('__shared__ const float* pk[16];','device const float* threadgroup pk[16];')
        code=code.replace('__shared__','threadgroup').replace('__restrict__','')
        code=re.sub(r'(?<![\w*])((?:const\s+)?(?:float|int32_t|uint8_t|__half|unsigned long long)\s*\*\s*(?:const\s*\*)?)',lambda m:metal_type(m[1]),code)
        code=code.replace('threadgroup device const float* threadgroup','device const float* threadgroup').replace('device device ','device ')
        for old,new in [('__syncthreads()','threadgroup_barrier(mem_flags::mem_threadgroup | mem_flags::mem_device)'),('__float2half_rn','half'),('__float2half','half'),('__half2float','float'),('__expf','exp'),('expf','exp'),('rsqrtf','rsqrt'),('fmaf','fma'),('fmaxf','max'),('fminf','min'),('powf','pow'),('sqrtf','sqrt'),('sinf','sin'),('cosf','cos'),('fabsf','abs'),('__half','half'),('int32_t','int'),('uint8_t','uchar')]:code=code.replace(old,new)
        code=re.sub(r'__shfl_xor_sync\(0xffffffffu,\s*([^,]+),\s*([^,)]+)(?:,\s*32)?\)',r'simd_shuffle_xor(\1,\2)',code)
        while '__ldg(' in code:
            a=code.index('__ldg(');b=a+5;c=end(code,b)
            code=code[:a]+'(*('+code[b+1:c]+'))'+code[c+1:]
        code=re.sub(r'\b(block_sum|block_max)\(([^,]+),\s*sh\)',r'\1(\2,sh,threadIdx,blockDim)',code)
        shader.append('kernel void '+name+'(constant Args_'+name+'& a [[buffer(0)]],uint3 blockIdx [[threadgroup_position_in_grid]], uint3 threadIdx [[thread_position_in_threadgroup]],uint3 blockDim [[threads_per_threadgroup]],uint3 gridDim [[threadgroups_per_grid]]){\n'+''.join(metal_type(t)+' '+n+'=a.'+n+';\n' for t,n in zip(types,names))+code+'\n}')
    (out/'q36_launch.hpp').write_text('\n'.join(header))
    (out/'q36.metal').write_text('\n'.join(shader))
    (out/'q36_host.cpp').write_text(host)

if __name__=='__main__':main()
