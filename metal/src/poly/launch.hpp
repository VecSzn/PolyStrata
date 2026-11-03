#pragma once
#include "strata/gpu/metal.hpp"
#include <fstream>
#include <iterator>
#include <stdexcept>
#include <string>
namespace strata::poly::metal_detail {
inline std::string read_source(const char* path) {
    std::ifstream f(path);
    if (!f) throw std::runtime_error(std::string("cannot read Metal kernel: ") + path);
    return std::string(std::istreambuf_iterator<char>(f), {});
}
template<class Args>
void dispatch(const char* library, const std::string& source, const char* name, const Args& args,
              void* stream, gpu::metal::Grid groups, uint32_t threads, gpu::metal::Resources resources = {}) {
    std::string error;
    auto* p = gpu::metal::pipeline(library, source.c_str(), name, error);
    if (!p) throw std::runtime_error(error);
    if (!resources.complete && !resources.indirect_table) {
        if constexpr (requires { args.p; }) resources = {args.p, std::size(args.p), true};
    }
    auto rc = gpu::metal::dispatch(static_cast<gpu::Stream>(stream), p, &args, sizeof(args), groups, {threads,1,1}, 0, resources);
    if (rc != gpu::kSuccess) throw std::runtime_error(gpu::error_string(rc));
}
}
