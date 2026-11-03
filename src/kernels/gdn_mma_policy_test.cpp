#include "strata/kernels/gdn_mma_policy.hpp"
#include <cstdio>
#include <initializer_list>
#include <stdexcept>
using namespace strata::kernels;
namespace {
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
}
int main() {
    try {
        require(gdn_mma_arch(12, 0), "Blackwell SM120 dispatch");
        require(gdn_mma_arch(12, 1), "DGX Spark SM121 dispatch");
        require(gdn_mma_arch(8, 6) && gdn_mma_arch(8, 9), "SM86/89 dispatch");
        for (int cc : {75, 80, 87, 90, 100, 103, 122, 130})
            require(!gdn_mma_arch(cc / 10, cc % 10), "unsupported architecture must fall back");
        require(gdn_mma_shape(128, 32, 16, 64, false), "first eligible row count");
        for (int heads : {16, 32, 48, 64})
            require(gdn_mma_shape(128, heads, 16, 65, false), "upstream value-head shape");
        require(!gdn_mma_shape(128, 32, 16, 63, false), "short-call fallback");
        require(!gdn_mma_shape(128, 32, 16, 64, true), "intermediate snapshots must fall back");
        require(!gdn_mma_shape(64, 32, 16, 64, false) && !gdn_mma_shape(128, 24, 16, 64, false) &&
                !gdn_mma_shape(128, 32, 8, 64, false), "shape fallback");
        require(!gdn_mma_shape(128, 32, 16, INT_MAX / 32 + 1, false), "launch-index overflow fallback");
        require(gdn_mma_schedule(24, 32) == GdnMmaPath::whole_head, "4060 whole-head schedule");
        require(gdn_mma_schedule(64, 32) == GdnMmaPath::slice32_two_blocks, "two-block threshold");
        require(gdn_mma_schedule(128, 32) == GdnMmaPath::slice32_one_block, "one-block threshold");
        require(gdn_mma_schedule(63, 32) == GdnMmaPath::whole_head &&
                gdn_mma_schedule(127, 32) == GdnMmaPath::slice32_two_blocks, "threshold edges");
        require(gdn_mma_schedule(0, 32) == GdnMmaPath::unavailable, "invalid device fallback");
        std::puts("PASS GDN host dispatch: architecture, shapes, snapshot exclusion and schedule boundaries");
        return 0;
    } catch (const std::exception& error) {
        std::fprintf(stderr, "FAIL %s\n", error.what());
        return 1;
    }
}
