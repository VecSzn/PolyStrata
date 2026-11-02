// Non-overlapping CPU copies into upload-only buffers. The caller publishes task completion after this returns.
// Keep this internal to expert staging: streaming stores help when the CPU will not read the destination next.
#pragma once
#include <algorithm>
#include <cstddef>
#include <cstdint>
#include <cstring>
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
#include <emmintrin.h>
#endif

namespace strata::poly {
inline constexpr bool stage_copy_streaming_available() {
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    return true;
#else
    return false;
#endif
}

inline void stage_copy_streaming(void* destination, const void* source, size_t bytes) {
#if defined(__SSE2__) || defined(_M_X64) || (defined(_M_IX86_FP) && _M_IX86_FP >= 2)
    // Whole 64-byte lines avoid mixing ordinary and streaming stores within a
    // cache line on the measured x86 hosts. There is no size-based performance cutoff.
    // Two lines per iteration is only an unrolling choice; SSE2 stores need 16-byte alignment.
    constexpr size_t kCacheLineBytes = 64;
    constexpr size_t kBlockBytes = 2 * kCacheLineBytes;
    auto* d = static_cast<unsigned char*>(destination);
    auto* s = static_cast<const unsigned char*>(source);
    // Leave partial cache lines to ordinary stores; do not mix them with streaming stores in the same line.
    const size_t misalignment = reinterpret_cast<uintptr_t>(d) & (kCacheLineBytes - 1);
    const size_t head = (std::min)(bytes, (kCacheLineBytes - misalignment) & (kCacheLineBytes - 1));
    // No complete aligned line: copy normally, without issuing a streaming-store fence.
    if (bytes - head < kCacheLineBytes) {
        std::memcpy(d, s, bytes);
        return;
    }
    if (head) {
        std::memcpy(d, s, head);
        d += head;
        s += head;
        bytes -= head;
    }
    while (bytes >= kBlockBytes) {
        const auto v0 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s));
        const auto v1 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 16));
        const auto v2 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 32));
        const auto v3 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 48));
        const auto v4 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 64));
        const auto v5 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 80));
        const auto v6 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 96));
        const auto v7 = _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 112));
        _mm_stream_si128(reinterpret_cast<__m128i*>(d), v0);
        _mm_stream_si128(reinterpret_cast<__m128i*>(d + 16), v1);
        _mm_stream_si128(reinterpret_cast<__m128i*>(d + 32), v2);
        _mm_stream_si128(reinterpret_cast<__m128i*>(d + 48), v3);
        _mm_stream_si128(reinterpret_cast<__m128i*>(d + 64), v4);
        _mm_stream_si128(reinterpret_cast<__m128i*>(d + 80), v5);
        _mm_stream_si128(reinterpret_cast<__m128i*>(d + 96), v6);
        _mm_stream_si128(reinterpret_cast<__m128i*>(d + 112), v7);
        d += kBlockBytes;
        s += kBlockBytes;
        bytes -= kBlockBytes;
    }
    if (bytes >= kCacheLineBytes) {
        _mm_stream_si128(reinterpret_cast<__m128i*>(d), _mm_loadu_si128(reinterpret_cast<const __m128i*>(s)));
        _mm_stream_si128(reinterpret_cast<__m128i*>(d + 16), _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 16)));
        _mm_stream_si128(reinterpret_cast<__m128i*>(d + 32), _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 32)));
        _mm_stream_si128(reinterpret_cast<__m128i*>(d + 48), _mm_loadu_si128(reinterpret_cast<const __m128i*>(s + 48)));
        d += kCacheLineBytes;
        s += kCacheLineBytes;
        bytes -= kCacheLineBytes;
    }
    if (bytes) std::memcpy(d, s, bytes);
    // Drain this worker's streaming stores before it publishes task completion.
    // A fence on the coordinator thread would not drain the other workers' stores.
    _mm_sfence();
#else
    std::memcpy(destination, source, bytes);
#endif
}
} // namespace strata::poly
