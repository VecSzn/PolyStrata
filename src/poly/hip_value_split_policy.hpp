// Host-only selection for HIP's opt-in split value reduction.
#pragma once
#include <cstddef>

namespace strata::poly::hip_value_split {
// On gfx1201 the split was slower at 8K and neutral-to-faster from 12K; leave
// the shorter path to the existing kernel.
constexpr int kKeys = 1024, kMinKeys = 12288, kDim = 256, kGroup = 8, kShares = 8;

// Explicit opt-in. In particular, an existing variable set to "0" is still off.
constexpr bool requested(const char* value) {
    return value && value[0] == '1' && value[1] == '\0';
}

// scratch_floats is the space before the transposed softmax weights, which must
// remain intact. Dividing the capacity avoids overflowing a product of sizes.
constexpr bool fits(int dim, int heads, int kv_heads, int keys, int rows, int window, int ring, bool whole,
                    size_t scratch_floats) {
    if (dim != kDim || heads <= 0 || kv_heads <= 0 || heads % kGroup || heads / kGroup != kv_heads ||
        keys < kMinKeys || rows <= 0 || window != 0 || ring != 0 || whole)
        return false;
    const size_t parts = size_t((keys - 1) / kKeys) + 1;
    return parts <= scratch_floats / kDim / size_t(heads) / size_t(rows);
}
}  // namespace strata::poly::hip_value_split
