#include "strata/core/conversation_memory.hpp"
#include <cstdio>
int main() {
    using namespace strata::core;
    auto initial=sample_host_memory();
    if(!initial || !initial->limit || initial->available>initial->limit) return 1;
    set_memory_limit_bytes(1);
    auto capped=sample_host_memory();
    set_memory_limit_bytes(0);
    if(!capped || capped->available>1 || capped->source!=MemorySource::flag) return 1;
    std::printf("macOS memory guard: known physical sample and operator cap applied\n");
    return 0;
}
