#include "stage_chunk.hpp"

#include <algorithm>
#include <array>
#include <cstdio>
#include <stdexcept>
#include <string>

using strata::poly::stage_chunk_experts;
using strata::poly::stage_chunk_target;

int main() {
    try {
        constexpr size_t mib = size_t(1) << 20;
        const auto check = [](bool ok) { if (!ok) throw std::runtime_error("wrong upload chunk"); };
        check(stage_chunk_target(nullptr) == 0);
        check(stage_chunk_target("0") == 0);
        for (const char* value : {"1", "32", "64", "128", "512"})
            check(stage_chunk_target(value) == size_t(std::stoi(value)) * mib);
        for (const char* value : {"", "-1", "+32", " 32", "32 ", "32MiB", "1.5", "513", "99999999999999999999999999"}) {
            check(stage_chunk_target(value) == 0);
        }
        // Real Tiel Q4_K gate/up + Q5_K down, including the ragged last chunk.
        constexpr size_t expert = 2 * 589824 + 720896;
        check(stage_chunk_experts(512 * mib, expert, 0) == 282);
        check(stage_chunk_experts(512 * mib, expert, 64 * mib) == 35);
        // The same real payload with user-selected transfer targets.
        check(stage_chunk_experts(512 * mib, expert, 32 * mib) == 17);
        check(stage_chunk_experts(512 * mib, expert, 128 * mib) == 70);
        check(stage_chunk_experts(512 * mib, expert, 512 * mib) == 282);
        check(stage_chunk_experts(32 * mib, 2 * mib, 128 * mib) == 16);
        // Gate and up may have different types and byte sizes on current upstream.
        constexpr size_t mixed_expert = 589824 + 720896 + 860160;
        check(stage_chunk_experts(512 * mib, mixed_expert, 0) == 247);
        check(stage_chunk_experts(512 * mib, mixed_expert, 64 * mib) == 30);
        int at = 0;
        for (int expected : std::array{35, 35, 35, 23}) {
            const int n = (int) std::min(size_t(128 - at), stage_chunk_experts(512 * mib, expert, 64 * mib));
            check(n == expected);
            check(size_t(n) * expert <= 64 * mib);
            at += n;
        }
        check(at == 128);
        // Smaller existing allocations and experts at/above the target boundary.
        check(stage_chunk_experts(32 * mib, 2 * mib, 64 * mib) == 16);
        check(stage_chunk_experts(512 * mib, 64 * mib, 64 * mib) == 1);
        check(stage_chunk_experts(512 * mib, 64 * mib + 1, 64 * mib) == 7);
        check(stage_chunk_experts(512 * mib, 512 * mib, 64 * mib) == 1);
        for (size_t bytes : {expert, 64 * mib, 64 * mib + 1, 512 * mib})
            check(stage_chunk_experts(512 * mib, bytes, 0) == 512 * mib / bytes);
        std::puts("PASS upload chunk limits, stock capacity and ragged expert groups");
        return 0;
    } catch (const std::exception& e) {
        std::fprintf(stderr, "FAIL %s\n", e.what());
        return 1;
    }
}
