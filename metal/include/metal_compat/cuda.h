// Metal buffers have stable GPU addresses but no CUDA driver VMM. Advertise no
// VMM support so strata-poly uses its existing ordinary-allocation cache path.
#pragma once
#include <cstddef>
using CUresult=int;
using CUdevice=int;
using CUdeviceptr=unsigned long long;
using CUmemGenericAllocationHandle=unsigned long long;
constexpr CUresult CUDA_SUCCESS=0, CUDA_ERROR_NOT_SUPPORTED=801;
constexpr int CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED=102;
constexpr int CU_MEM_ALLOCATION_TYPE_PINNED=1,CU_MEM_LOCATION_TYPE_DEVICE=1;
constexpr int CU_MEM_ACCESS_FLAGS_PROT_READWRITE=3,CU_MEM_ALLOC_GRANULARITY_MINIMUM=0;
struct CUmemLocation { int type,id; };
struct CUmemAllocationProp { int type; CUmemLocation location; };
struct CUmemAccessDesc { CUmemLocation location; unsigned long long flags; };
inline CUresult cuGetErrorString(CUresult,const char** s) { *s="CUDA virtual memory is unavailable on Metal";return CUDA_SUCCESS; }
inline CUresult cuDeviceGet(CUdevice* d,int i) { *d=i;return i==0?CUDA_SUCCESS:CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult cuDeviceGetAttribute(int* v,int,CUdevice) { *v=0;return CUDA_SUCCESS; }
inline CUresult cuMemGetAllocationGranularity(size_t*,const CUmemAllocationProp*,int) { return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult cuMemAddressReserve(CUdeviceptr*,size_t,size_t,CUdeviceptr,unsigned long long) { return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult cuMemAddressFree(CUdeviceptr,size_t) { return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult cuMemCreate(CUmemGenericAllocationHandle*,size_t,const CUmemAllocationProp*,unsigned long long) { return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult cuMemRelease(CUmemGenericAllocationHandle) { return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult cuMemMap(CUdeviceptr,size_t,size_t,CUmemGenericAllocationHandle,unsigned long long) { return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult cuMemUnmap(CUdeviceptr,size_t) { return CUDA_ERROR_NOT_SUPPORTED; }
inline CUresult cuMemSetAccess(CUdeviceptr,size_t,const CUmemAccessDesc*,size_t) { return CUDA_ERROR_NOT_SUPPORTED; }
