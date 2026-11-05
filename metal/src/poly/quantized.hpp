#pragma once
#include "strata/gpu/metal.hpp"
#include <string>
namespace strata::poly::metal_detail {
struct QArgs {
    const void* p[24]{};
    uint64_t stride=0,xstride=0,ystride=0,offset=0;
    int ni=0,no=0,rows=0,count=0,every=1,kind=0;
};
static_assert(sizeof(QArgs)==248);

void qrun(const std::string&,const QArgs&,void*,gpu::metal::Grid);
const char* quant_name(int);
}
