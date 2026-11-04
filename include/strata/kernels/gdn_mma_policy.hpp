// Host-only dispatch policy for the opt-in GDN prefill adapter.
#pragma once
#include <climits>
#include <cstdint>
namespace strata::kernels {
enum class GdnMmaPath { unavailable, whole_head, slice32_one_block, slice32_two_blocks };
// Match the inspected upstream schedule: GA10x, Ada, Blackwell and DGX Spark.
constexpr bool gdn_mma_arch(int major, int minor) {
    return (major == 8 && (minor == 6 || minor == 9)) || (major == 12 && (minor == 0 || minor == 1));
}
constexpr bool gdn_mma_shape(int dim, int heads, int key_heads, int rows, bool snapshots) {
    return dim == 128 && key_heads == 16 && (heads == 16 || heads == 32 || heads == 48 || heads == 64) &&
           rows >= 64 && rows <= INT_MAX / heads && !snapshots;
}
constexpr GdnMmaPath gdn_mma_schedule(int multiprocessors, int heads) {
    if (multiprocessors <= 0 || heads <= 0) return GdnMmaPath::unavailable;
    if (int64_t(heads) * 2 > multiprocessors) return GdnMmaPath::whole_head;
    return int64_t(heads) * 4 <= multiprocessors ? GdnMmaPath::slice32_one_block : GdnMmaPath::slice32_two_blocks;
}
} // namespace strata::kernels
