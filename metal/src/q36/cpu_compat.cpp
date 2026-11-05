// x86 entry points are unavailable on Apple Silicon; the shared CPU expert
// implementation selects ggml CPU (NEON) instead. Guard failures stop explicitly.
#include "strata/kernels/cpu/expert.hpp"
#include "strata/kernels/cpu/expert_layout.hpp"
#include "strata/kernels/cpu/iq_avx1.hpp"
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/cpu/iq_avx512.hpp"
#include "strata/kernels/cpu/kq_avx1.hpp"
#include "strata/kernels/cpu/kq_avx2.hpp"
#include "strata/kernels/cpu/native_expert.hpp"
#include "strata/kernels/cpu/pool.hpp"
#include "strata/kernels/cpu/nvfp4_avx2.hpp"
#include "strata/kernels/cpu/nvfp4_avx512.hpp"
#include <stdexcept>
namespace strata::metal_engine { [[noreturn]] void missing(const char* n){throw std::logic_error(n);} }
namespace strata::kernels::cpu { void act_quant_q8_1(const float * p0, int p1, ActQ & p2) { strata::metal_engine::missing("strata::kernels::cpu::act_quant_q8_1"); } }
namespace strata::kernels::cpu { void act_quant_q8_1_avx1(const float * p0, int p1, ActQ & p2) { strata::metal_engine::missing("strata::kernels::cpu::act_quant_q8_1_avx1"); } }
namespace strata::kernels::cpu { void act_quant_q8_1_avx2(const float * p0, int p1, ActQ & p2) { strata::metal_engine::missing("strata::kernels::cpu::act_quant_q8_1_avx2"); } }
namespace strata::kernels::cpu { void bf16_rows_dot_multi(const uint16_t * p0, int p1, int p2, const float * p3, int p4, float * p5) { strata::metal_engine::missing("strata::kernels::cpu::bf16_rows_dot_multi"); } }
namespace strata::kernels::cpu { void bf16_rows_dot_multi_avx1(const uint16_t * p0, int p1, int p2, const float * p3, int p4, float * p5) { strata::metal_engine::missing("strata::kernels::cpu::bf16_rows_dot_multi_avx1"); } }
namespace strata::kernels::cpu { void iq128_down_rows(int p0, const uint8_t * p1, size_t p2, int p3, const void *const * p4, int p5, float *const * p6, int p7, int p8) { strata::metal_engine::missing("strata::kernels::cpu::iq128_down_rows"); } }
namespace strata::kernels::cpu { bool iq128_down_supported(int p0) noexcept { strata::metal_engine::missing("strata::kernels::cpu::iq128_down_supported"); return {}; } }
namespace strata::kernels::cpu { void iq128_gu_rows(int p0, const uint8_t * p1, size_t p2, size_t p3, int p4, const void *const * p5, int p6, float *const * p7, int p8, int p9) { strata::metal_engine::missing("strata::kernels::cpu::iq128_gu_rows"); } }
namespace strata::kernels::cpu { bool iq128_supported(int p0) noexcept { strata::metal_engine::missing("strata::kernels::cpu::iq128_supported"); return {}; } }
namespace strata::kernels::cpu { void iq256_gu_rows(int p0, const uint8_t * p1, size_t p2, size_t p3, int p4, const void *const * p5, int p6, float *const * p7, int p8, int p9) { strata::metal_engine::missing("strata::kernels::cpu::iq256_gu_rows"); } }
namespace strata::kernels::cpu { void iq4nl256_down_rows(const uint8_t * p0, size_t p1, int p2, const void *const * p3, int p4, float *const * p5, int p6, int p7) { strata::metal_engine::missing("strata::kernels::cpu::iq4nl256_down_rows"); } }
namespace strata::kernels::cpu { void iq512_gu_rows(int p0, const uint8_t * p1, size_t p2, size_t p3, int p4, const void *const * p5, int p6, float *const * p7, int p8, int p9) { strata::metal_engine::missing("strata::kernels::cpu::iq512_gu_rows"); } }
namespace strata::kernels::cpu { void kq256_gu_rows(int p0, const uint8_t * p1, size_t p2, size_t p3, int p4, const void *const * p5, int p6, float *const * p7, int p8, int p9) { strata::metal_engine::missing("strata::kernels::cpu::kq256_gu_rows"); } }
namespace strata::kernels::cpu { void kq256_rows(int p0, const uint8_t * p1, size_t p2, int p3, const void *const * p4, int p5, float *const * p6, int p7, int p8) { strata::metal_engine::missing("strata::kernels::cpu::kq256_rows"); } }
namespace strata::kernels::cpu { void q2_0_gguf_rows_multi(const uint8_t * p0, size_t p1, int p2, const ActQ *const * p3, int p4, float *const * p5, int p6, int p7) { strata::metal_engine::missing("strata::kernels::cpu::q2_0_gguf_rows_multi"); } }
namespace strata::kernels::cpu { void q2_0_gguf_rows_multi_avx1(const uint8_t * p0, size_t p1, int p2, const ActQ *const * p3, int p4, float *const * p5, int p6, int p7) { strata::metal_engine::missing("strata::kernels::cpu::q2_0_gguf_rows_multi_avx1"); } }
namespace strata::kernels::cpu { void q2_0_gguf_rows_multi_avx1_legacy(const uint8_t * p0, size_t p1, int p2, const ActQ *const * p3, int p4, float *const * p5, int p6, int p7) { strata::metal_engine::missing("strata::kernels::cpu::q2_0_gguf_rows_multi_avx1_legacy"); } }
namespace strata::kernels::cpu { void q2_0_gguf_rows_multi_avx2(const uint8_t * p0, size_t p1, int p2, const ActQ *const * p3, int p4, float *const * p5, int p6, int p7) { strata::metal_engine::missing("strata::kernels::cpu::q2_0_gguf_rows_multi_avx2"); } }
namespace strata::kernels::cpu { void q2_0_gguf_rows_multi_avx2_legacy(const uint8_t * p0, size_t p1, int p2, const ActQ *const * p3, int p4, float *const * p5, int p6, int p7) { strata::metal_engine::missing("strata::kernels::cpu::q2_0_gguf_rows_multi_avx2_legacy"); } }
namespace strata::kernels::cpu { void q8k_quant_avx1(const float * p0, void * p1, int64_t p2) { strata::metal_engine::missing("strata::kernels::cpu::q8k_quant_avx1"); } }
namespace strata::kernels::cpu { void q8k_quant_avx2(const float * p0, void * p1, int64_t p2) { strata::metal_engine::missing("strata::kernels::cpu::q8k_quant_avx2"); } }
namespace strata::kernels::cpu { void s2_expert_down_rows(const uint8_t * p0, const ActQ & p1, float * p2, int p3, int p4) { strata::metal_engine::missing("strata::kernels::cpu::s2_expert_down_rows"); } }
namespace strata::kernels::cpu { void s2_expert_down_rows_multi(const uint8_t * p0, const ActQ *const * p1, int p2, float *const * p3, int p4, int p5) { strata::metal_engine::missing("strata::kernels::cpu::s2_expert_down_rows_multi"); } }
namespace strata::kernels::cpu { void s2_expert_gu_rows(const uint8_t * p0, const ActQ & p1, float * p2, int p3, int p4) { strata::metal_engine::missing("strata::kernels::cpu::s2_expert_gu_rows"); } }
namespace strata::kernels::cpu { void s2_expert_gu_rows_multi(const uint8_t * p0, const ActQ *const * p1, int p2, float *const * p3, int p4, int p5) { strata::metal_engine::missing("strata::kernels::cpu::s2_expert_gu_rows_multi"); } }
namespace strata::kernels::cpu { void s2_expert_vnni_q(const uint8_t * p0, const ActQ & p1, float * p2, ExpertScratch & p3) { strata::metal_engine::missing("strata::kernels::cpu::s2_expert_vnni_q"); } }
namespace strata::kernels::cpu {
bool nvfp4_256_fits(int) { return false; }
bool nvfp4_512_fits(int) { return false; }
void nvfp4_256_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float, float) {
    strata::metal_engine::missing("strata::kernels::cpu::nvfp4_256_gu_rows");
}
void nvfp4_256_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    strata::metal_engine::missing("strata::kernels::cpu::nvfp4_256_rows");
}
void nvfp4_512_gu_rows(const uint8_t*, size_t, size_t, int, const void* const*, int, float* const*, int, int, float, float) {
    strata::metal_engine::missing("strata::kernels::cpu::nvfp4_512_gu_rows");
}
void nvfp4_512_rows(const uint8_t*, size_t, int, const void* const*, int, float* const*, int, int, float) {
    strata::metal_engine::missing("strata::kernels::cpu::nvfp4_512_rows");
}
}

namespace strata::kernels::cpu { void iq256_gu_rows_exact_one(int p0, const uint8_t * p1, size_t p2, size_t p3, int p4, const void * p5, float * p6, int p7, int p8) { strata::metal_engine::missing("strata::kernels::cpu::iq256_gu_rows_exact_one"); } }