// New shared-host entry points: Metal uses window.cpp for decode and prefill.
// PC-only kernels fail explicitly; optional probes retain documented fallbacks.
#include "strata/kernels/cpu/iq_avx2.hpp"
#include "strata/kernels/fused_gr.hpp"
#include "strata/kernels/verify_kernels.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_select.hpp"
#include "strata/prefill/kernels.hpp"
#include "strata/prefill/gemm.hpp"

namespace strata::metal_engine { void missing(const char* name); }

namespace strata::kernels { void copy_rows_from_mapped_after(const uint32_t * p0, uint32_t p1, float * p2, const float * p3, int64_t p4, int64_t p5, const int32_t * p6, const int32_t * p7, void * p8) { strata::metal_engine::missing("strata::kernels::copy_rows_from_mapped_after"); } }
namespace strata::kernels::cpu { void iq256_gu_rows_exact_one(int p0, const uint8_t * p1, size_t p2, size_t p3, int p4, const void * p5, float * p6, int p7, int p8) { strata::metal_engine::missing("strata::kernels::cpu::iq256_gu_rows_exact_one"); } }
namespace strata::kernels { void fused_gr_persist_init() { /* HIP-only optional initialization; no work on Metal. */ } }
namespace strata::kernels { bool fused_gr_read_persist(const FusedGrArgs * p0, int p1, void * p2) { return false; /* caller retains its ordinary kernel path */ } }
namespace strata::kernels { void gdn_step_norm_groups(const GdnGroups & p0, const float * p1, int p2, const float * p3, const float * p4, const float * p5, const float * p6, float p7, float * p8, int p9, int p10, bool p11, void * p12) { strata::metal_engine::missing("strata::kernels::gdn_step_norm_groups"); } }
namespace strata::kernels { void handoff_publish(float * p0, const float * p1, int64_t p2, const float * p3, int64_t p4, const float * p5, int64_t p6, uint32_t * p7, uint32_t * p8, void * p9, const uint32_t * p10) { strata::metal_engine::missing("strata::kernels::handoff_publish"); } }
namespace strata::kernels { void hc_q8_to_bf16(const void * p0, const float * p1, uint16_t * p2, long long p3, void * p4) { strata::metal_engine::missing("strata::kernels::hc_q8_to_bf16"); } }
namespace strata::kernels { void native_quantize_q8_1_il(const float * p0, void * p1, void * p2, int p3, int p4, void * p5) { strata::metal_engine::missing("strata::kernels::native_quantize_q8_1_il"); } }
namespace strata::kernels { void qsa_block_scores_rows(const QsaScoreRows & p0, const float * p1, const int32_t * p2, int64_t p3, int64_t p4, const QsaShapes & p5, float * p6, void * p7) { strata::metal_engine::missing("strata::kernels::qsa_block_scores_rows"); } }
namespace strata::kernels { void qsa_decode_attn_batch_rows(const float * p0, const QsaAttnPoolsRows & p1, const int32_t * p2, const int32_t * p3, int64_t p4, const QsaShapes & p5, float * p6, float * p7, int64_t p8, void * p9) { strata::metal_engine::missing("strata::kernels::qsa_decode_attn_batch_rows"); } }
namespace strata::kernels { bool qsa_decode_attn_split(const float * p0, const QsaAttnPools & p1, const int32_t * p2, const int32_t * p3, int64_t p4, const QsaShapes & p5, float * p6, uint64_t p7, float * p8, int64_t p9, void * p10) { return false; /* caller retains its ordinary kernel path */ } }
namespace strata::kernels { void res_patch(int32_t * p0, const int32_t * p1, int p2, void * p3) { strata::metal_engine::missing("strata::kernels::res_patch"); } }
namespace strata::kernels { void wait_copy_i32_from_mapped(const uint32_t * p0, uint32_t p1, int32_t * p2, const int32_t * p3, int64_t p4, void * p5) { strata::metal_engine::missing("strata::kernels::wait_copy_i32_from_mapped"); } }
namespace strata::kernels { void wait_flag_ge_stamped(const uint32_t * p0, uint32_t p1, unsigned long long * p2, unsigned long long * p3, void * p4) { strata::metal_engine::missing("strata::kernels::wait_flag_ge_stamped"); } }
namespace strata::kernels { void wait_handoff(const uint32_t * p0, uint32_t p1, uint32_t * p2, unsigned long long p3, void * p4) { strata::metal_engine::missing("strata::kernels::wait_handoff"); } }
namespace strata::prefill { bool f16_range_read(float & p0, unsigned long long & p1, bool p2) { return false; /* no HIP FP16 range counters on Metal */ } }
namespace strata::prefill { bool gemm_f16_range_read(float * p0, unsigned long long * p1, bool p2) { return false; /* no HIP FP16 range counters on Metal */ } }
