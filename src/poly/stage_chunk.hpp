#pragma once

#include <algorithm>
#include <charconv>
#include <cstddef>
#include <cstdio>
#include <string_view>

namespace strata::poly {

// Unset keeps stock grouping. The target never enlarges the pinned buffers.
inline size_t stage_chunk_target(const char* value) {
    if (!value) return 0;
    size_t mib = 0;
    const std::string_view text(value);
    const auto [end, error] = std::from_chars(text.data(), text.data() + text.size(), mib);
    if (error != std::errc{} || end != text.data() + text.size() || mib > 512) {
        std::fprintf(stderr, "strata-poly: warning: STRATA_POLY_CHUNK_STAGE_MIB must be an integer from 0 to 512; using stock staging\n");
        return 0;
    }
    return mib << 20;
}

// Positive expert_bytes must fit the existing pinned buffer. Zero target keeps
// stock grouping, as do experts larger than the requested target.
inline size_t stage_chunk_experts(size_t buffer_bytes, size_t expert_bytes, size_t target_bytes) {
    const size_t bytes = target_bytes && expert_bytes <= target_bytes ? std::min(buffer_bytes, target_bytes) : buffer_bytes;
    return bytes / expert_bytes;
}

}  // namespace strata::poly
