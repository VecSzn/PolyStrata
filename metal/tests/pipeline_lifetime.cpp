// Pipeline copies keep the native object alive through recorded dispatches and
// release it when the last owner is discarded (including replacement of a cache).
#include "mtl_rt.h"
#include <cstdio>

int main() {
    mtl::Context cx;
    const int lib = cx.compile({"rt_bench.metal"}, "#define NO_GGML_COMMON 1\n");
    mtl::Buffer out = cx.alloc(4);
    std::weak_ptr<void> weak;
    mtl::Pipeline saved;
    {
        auto original = cx.pipeline(lib, "k_chain");
        weak = original.owner;
        saved = original;
    }
    if (weak.expired()) return 1;
    mtl::Dispatch d;
    d.p = &saved;
    d.args = {mtl::Arg::B(out)};
    cx.run({d}, 3);
    const bool correct = out.as<uint32_t>()[0] == 3;
    saved = {};
    const bool released = weak.expired();
    cx.free(out);
    std::printf("copied pipeline dispatch: %s; last owner released: %s\n",
                correct ? "PASS" : "FAIL", released ? "PASS" : "FAIL");
    return correct && released ? 0 : 1;
}
