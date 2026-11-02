// No GPU or model required: explicit opt-in, supported shapes and scratch bounds.
#include "hip_value_split_policy.hpp"
#include <climits>
#include <cstdio>
#include <limits>

using namespace strata::poly::hip_value_split;
int main() {
    int bad = 0;
    const auto check = [&](bool ok, const char* why) {
        if (!ok) { std::fprintf(stderr, "FAIL %s\n", why); ++bad; }
    };
    check(requested("1"), "literal 1 enables");
    const char* disabled[] = {nullptr, "", "0", "2", "true", "01", " 1", "1suffix"};
    for (const char* s : disabled)
        check(!requested(s), "unset, 0 and nonliteral values are disabled");
    constexpr size_t need = size_t(12) * 16 * 4 * 256;
    check(fits(256, 16, 2, 12288, 4, 0, 0, false, need), "exact capacity");
    check(!fits(256, 16, 2, 12288, 4, 0, 0, false, need - 1), "one float short");
    check(!fits(256, 16, 2, 12289, 4, 0, 0, false, need), "partial last segment needs space");
    check(fits(256, 16, 2, 12289, 4, 0, 0, false, need + 16 * 4 * 256), "partial last segment fits");
    check(!fits(256, 16, 2, 12287, 4, 0, 0, false, need), "short context fallback");
    check(!fits(128, 16, 2, 12288, 4, 0, 0, false, need), "other width");
    check(!fits(256, 32, 2, 12288, 4, 0, 0, false, need), "other group");
    check(!fits(256, 17, 2, 12288, 4, 0, 0, false, need), "fractional group");
    check(!fits(256, 16, 0, 12288, 4, 0, 0, false, need), "zero KV heads");
    check(!fits(256, 16, 2, 12288, 0, 0, 0, false, need), "zero rows");
    check(!fits(256, 16, 2, 12288, 4, 1, 0, false, need), "window fallback");
    check(!fits(256, 16, 2, 12288, 4, 0, 1, false, need), "ring fallback");
    check(!fits(256, 16, 2, 12288, 4, 0, 0, true, need), "whole fallback");
    check(!fits(256, INT_MAX - 7, (INT_MAX - 7) / 8, INT_MAX, INT_MAX, 0, 0, false,
                std::numeric_limits<size_t>::max()), "size products cannot wrap into fitting");
    std::printf("poly_hip_value_split_policy_test: %s\n", bad ? "FAIL" : "PASS");
    return bad ? 1 : 0;
}
