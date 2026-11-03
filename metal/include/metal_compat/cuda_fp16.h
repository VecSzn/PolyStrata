#pragma once
// Host-side half conversions for shared reference tests. No CUDA kernel emulation.
#include <cstdint>
#include <cstring>
using __half = _Float16;
using half = _Float16;
inline half __float2half(float v) { return static_cast<half>(v); }
inline half __float2half_rn(float v) { return static_cast<half>(v); }
inline float __half2float(half v) { return static_cast<float>(v); }
inline uint16_t __half_as_ushort(half v) { uint16_t b; std::memcpy(&b,&v,2); return b; }
inline half __ushort_as_half(uint16_t b) { half v; std::memcpy(&v,&b,2); return v; }
