# metal/ - Strata on Apple Silicon (Metal), MAC-PLAN.md.  Its own CMake project like sycl/, macOS only, never part of
# the CUDA / HIP / SYCL builds.
#
# Two parts:
#   * the kernel tree (mac-kern): the runtime face src/mtl_rt.*, the MSL kernels kernels/*.metal (compiled at run time),
#     CPU references src/ref_*.cpp and the parity / bench programs tests/*.cpp;
#   * the engine (mac-design): the shared host sources from ../src, UNCHANGED - include/metal_compat goes first on the
#     include path and gives them the CUDA runtime names on Metal, as include/strata/hip_compat does on HIP - over
#     the Metal device runtime src/gpu/runtime.cpp (metal-cpp) and the kernel library src/engine/*.
#
#   cmake -S metal -B build-metal -G Ninja -DCMAKE_BUILD_TYPE=Release -DMETAL_CPP_DIR=... [-DSTRATA_GGML_DIR=...]
#   ninja -C build-metal
cmake_minimum_required(VERSION 3.24)
project(strata_metal LANGUAGES C CXX OBJCXX)

if(NOT APPLE)
  message(FATAL_ERROR "metal/ builds on macOS only")
endif()

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_OBJCXX_STANDARD 20)
if(NOT CMAKE_BUILD_TYPE)
  set(CMAKE_BUILD_TYPE Release)
endif()
set(CMAKE_CXX_FLAGS_RELEASE "-O2")
set(CMAKE_OBJCXX_FLAGS_RELEASE "-O2")

include(CTest)

get_filename_component(STRATA_ROOT "${CMAKE_CURRENT_SOURCE_DIR}/.." ABSOLUTE)
set(GGML_COMMON "${STRATA_ROOT}/third_party/ggml/ggml-common.h")

# ---- the kernel tree (mac-kern) ------------------------------------------------------------------------------------
add_library(mtl_rt STATIC src/mtl_rt.mm src/ref_quant.cpp src/gguf_lite.cpp)
target_include_directories(mtl_rt PUBLIC src ${STRATA_ROOT}/third_party/ggml)
target_compile_options(mtl_rt PRIVATE $<$<COMPILE_LANGUAGE:OBJCXX>:-fobjc-arc>)
target_compile_definitions(mtl_rt PRIVATE
  STRATA_GGML_COMMON_DEFAULT="${GGML_COMMON}"
  STRATA_METAL_KERNELS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/kernels")
target_link_libraries(mtl_rt PUBLIC "-framework Metal" "-framework Foundation")

file(GLOB STRATA_METAL_TESTS CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/tests/*.cpp)
list(REMOVE_ITEM STRATA_METAL_TESTS ${CMAKE_CURRENT_SOURCE_DIR}/tests/runtime_hello.cpp
  ${CMAKE_CURRENT_SOURCE_DIR}/tests/nvfp4_parity.cpp)
foreach(src ${STRATA_METAL_TESTS})
  get_filename_component(t ${src} NAME_WE)
  add_executable(${t} ${src})
  target_link_libraries(${t} PRIVATE mtl_rt)
endforeach()

# These checks use generated inputs and CPU references; no model download is needed.
# Keep benchmarks and the model-dependent chain programs out of CTest. Serialize
# GPU checks even under ctest -j so they do not compete for unified memory.
if(BUILD_TESTING)
  add_test(NAME arena_budget_test COMMAND arena_budget_test)
  set_tests_properties(arena_budget_test PROPERTIES LABELS "metal;cpu")
  foreach(t dense_mm_parity expert_parity fuse_parity gdn_parity gdn_prompt_parity
            gemv_parity gr_parity gr2_parity mma_parity moe_mm_parity pair_parity
            prompt_parity qsa_parity qsa_prompt_parity route_parity sample_parity pipeline_lifetime)
    add_test(NAME ${t} COMMAND ${t})
    set_tests_properties(${t} PROPERTIES LABELS "metal;gpu;parity"
                         RESOURCE_LOCK metal_gpu TIMEOUT 300)
  endforeach()
endif()

target_sources(ready_stream PRIVATE src/engine/stream.cpp)
target_link_libraries(ready_stream PRIVATE mtl_rt)
add_test(NAME ready_stream COMMAND ready_stream)
set_tests_properties(ready_stream PROPERTIES LABELS "metal;gpu;synthetic" RESOURCE_LOCK metal_gpu TIMEOUT 120)

# ---- the device runtime and the engine (mac-design) ---------------------------------------------------------------
set(METAL_CPP_DIR "" CACHE PATH "metal-cpp headers (https://developer.apple.com/metal/cpp/), e.g. metal-cpp_26")
if(NOT EXISTS "${METAL_CPP_DIR}/Metal/Metal.hpp")
  message(STATUS "strata metal: METAL_CPP_DIR is not set: the device runtime and the engine are not built")
  return()
endif()

add_library(strata_gpu_metal STATIC src/gpu/runtime.cpp)
target_include_directories(strata_gpu_metal BEFORE PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include/metal_compat
                           ${CMAKE_CURRENT_SOURCE_DIR}/include ${STRATA_ROOT}/include)
target_include_directories(strata_gpu_metal SYSTEM PUBLIC ${METAL_CPP_DIR})
target_compile_definitions(strata_gpu_metal PUBLIC STRATA_GPU_METAL=1)
target_link_libraries(strata_gpu_metal PUBLIC "-framework Metal" "-framework Foundation" "-framework QuartzCore")

# Native strata-poly port, opt-in for the smaller families. This library uses
# the same stream and graph runtime as strata. Full-model validation is pending.
option(STRATA_METAL_POLY "Build the strata-poly Metal kernel port and checks" OFF)
if(STRATA_METAL_POLY)
  add_library(strata_poly_metal_kernels STATIC src/poly/basic.cpp src/poly/attention.cpp src/poly/recurrent.cpp src/poly/advanced.cpp)
  target_compile_definitions(strata_poly_metal_kernels PRIVATE
    STRATA_POLY_BASIC_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/poly/basic.metal"
    STRATA_POLY_ATTENTION_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/poly/attention.metal"
    STRATA_POLY_RECURRENT_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/poly/recurrent.metal"
    STRATA_POLY_ADVANCED_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/poly/advanced.metal")
  target_link_libraries(strata_poly_metal_kernels PUBLIC strata_gpu_metal)
  add_executable(poly_basic_parity poly_tests/basic_parity.cpp)
  target_link_libraries(poly_basic_parity PRIVATE strata_poly_metal_kernels)
  set_source_files_properties(${STRATA_ROOT}/src/poly/shortconv_test.cu PROPERTIES LANGUAGE CXX)
  add_executable(poly_shortconv_metal_test ${STRATA_ROOT}/src/poly/shortconv_test.cu)
  target_link_libraries(poly_shortconv_metal_test PRIVATE strata_poly_metal_kernels)
  set_source_files_properties(${STRATA_ROOT}/src/poly/route_test.cu PROPERTIES LANGUAGE CXX)
  add_executable(poly_route_metal_test ${STRATA_ROOT}/src/poly/route_test.cu)
  target_link_libraries(poly_route_metal_test PRIVATE strata_poly_metal_kernels)
  set_source_files_properties(${STRATA_ROOT}/src/poly/dsv4_test.cu PROPERTIES LANGUAGE CXX)
  add_executable(poly_dsv4_metal_test ${STRATA_ROOT}/src/poly/dsv4_test.cu)
  target_link_libraries(poly_dsv4_metal_test PRIVATE strata_poly_metal_kernels)
  add_executable(poly_advanced_parity poly_tests/advanced_parity.cpp)
  target_link_libraries(poly_advanced_parity PRIVATE strata_poly_metal_kernels)
  if(BUILD_TESTING)
    add_test(NAME poly_advanced_parity COMMAND poly_advanced_parity)
    set_tests_properties(poly_advanced_parity PROPERTIES LABELS "metal;gpu;poly" RESOURCE_LOCK metal_gpu TIMEOUT 120)
    add_test(NAME poly_dsv4_metal_test COMMAND poly_dsv4_metal_test)
    set_tests_properties(poly_dsv4_metal_test PROPERTIES LABELS "metal;gpu;poly" RESOURCE_LOCK metal_gpu TIMEOUT 120)
    add_test(NAME poly_basic_parity COMMAND poly_basic_parity)
    set_tests_properties(poly_basic_parity PROPERTIES LABELS "metal;gpu;poly"
                        RESOURCE_LOCK metal_gpu TIMEOUT 120)
    add_test(NAME poly_shortconv_metal_test COMMAND poly_shortconv_metal_test)
    set_tests_properties(poly_shortconv_metal_test PROPERTIES LABELS "metal;gpu;poly"
                        RESOURCE_LOCK metal_gpu TIMEOUT 120)
    add_test(NAME poly_route_metal_test COMMAND poly_route_metal_test)
    set_tests_properties(poly_route_metal_test PROPERTIES LABELS "metal;gpu;poly"
                        RESOURCE_LOCK metal_gpu TIMEOUT 120)
  endif()
endif()

add_library(strata_core_metal STATIC ${STRATA_ROOT}/src/core/graph.cpp)
target_link_libraries(strata_core_metal PUBLIC strata_gpu_metal)
add_executable(runtime_hello tests/runtime_hello.cpp)
target_link_libraries(runtime_hello PRIVATE strata_core_metal)
if(BUILD_TESTING)
  add_test(NAME runtime_hello COMMAND runtime_hello)
  set_tests_properties(runtime_hello PROPERTIES LABELS "metal;gpu;runtime"
                       RESOURCE_LOCK metal_gpu TIMEOUT 120)
endif()

# The engine: ggml (the native expert formats' type traits and ggml-cpu) from a llama.cpp checkout at the commit the
# PC build pins (CMakeLists.txt: 3cf03257).
option(STRATA_METAL_ENGINE "Build the engine (strata) for Metal" ON)
set(STRATA_GGML_DIR "" CACHE PATH "llama.cpp checkout at the pinned commit")
if(NOT STRATA_METAL_ENGINE)
  return()
endif()
if(NOT EXISTS "${STRATA_GGML_DIR}/ggml/CMakeLists.txt")
  message(FATAL_ERROR "STRATA_GGML_DIR='${STRATA_GGML_DIR}' is not a llama.cpp checkout")
endif()
set(BUILD_SHARED_LIBS OFF CACHE BOOL "" FORCE)
set(GGML_STATIC ON CACHE BOOL "" FORCE)
set(GGML_METAL OFF CACHE BOOL "" FORCE)
set(GGML_BLAS OFF CACHE BOOL "" FORCE)
set(GGML_OPENMP OFF CACHE BOOL "" FORCE)
set(GGML_BACKEND_DL OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_TESTS OFF CACHE BOOL "" FORCE)
set(GGML_BUILD_EXAMPLES OFF CACHE BOOL "" FORCE)
add_subdirectory(${STRATA_GGML_DIR}/ggml ${CMAKE_BINARY_DIR}/ggml EXCLUDE_FROM_ALL)

add_executable(nvfp4_parity tests/nvfp4_parity.cpp)
target_link_libraries(nvfp4_parity PRIVATE mtl_rt ggml-base)
add_test(NAME nvfp4_parity COMMAND nvfp4_parity)
set_tests_properties(nvfp4_parity PROPERTIES LABELS "metal;gpu;parity" RESOURCE_LOCK metal_gpu TIMEOUT 120)

set(S ${STRATA_ROOT}/src)
set(STRATA_METAL_HOST_SOURCES
  ${S}/core/device.cu ${S}/core/pinned.cu ${S}/platform/memory.cpp ${S}/core/graph.cpp ${S}/core/weights.cpp ${S}/core/layout.cpp
  ${S}/core/layer.cpp ${S}/core/session.cpp ${S}/core/expert_source.cpp ${S}/core/remote_experts.cpp ${S}/core/expert_cache.cpp
  ${S}/core/native_head.cpp ${S}/core/native_dense.cpp ${S}/core/verify.cpp ${S}/core/mtp.cpp
  ${S}/core/conversation_snapshot.cpp ${S}/core/conversation_state.cpp ${S}/core/conversation_memory.cpp
  ${S}/core/peer_experts.cpp ${S}/core/arch_defaults.cpp ${S}/core/vmm.cpp ${S}/core/conversation_file.cpp
  ${S}/kernels/ngram.cpp ${S}/ngram/ple_reader.cpp ${S}/platform/direct_file.cpp
  ${S}/prefill/prefill.cpp
  ${S}/spec/suffix_drafter.cpp ${S}/spec/controller.cpp ${S}/spec/draft_policy.cpp ${S}/spec/draft_source.cpp
  ${S}/kernels/cpu/pool.cpp ${S}/kernels/cpu/expert_layout.cpp ${S}/kernels/cpu/native_expert.cpp)
set_source_files_properties(${S}/core/device.cu ${S}/core/pinned.cu PROPERTIES LANGUAGE CXX)

# The kernel library: Metal implementations (src/engine/*.cpp) and, for every entry point not ported yet, a stub
# that stops the engine with the entry point's name (src/engine/stubs.cpp, generated by tools/gen_stubs.py).
file(GLOB STRATA_METAL_ENGINE_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/engine/*.cpp)

add_library(strata_metal_host STATIC ${STRATA_METAL_HOST_SOURCES})
target_include_directories(strata_metal_host PUBLIC ${STRATA_ROOT}/include ${S}/core ${STRATA_ROOT}/third_party/ggml
                           ${STRATA_GGML_DIR}/ggml/include ${STRATA_GGML_DIR}/ggml/src ${STRATA_GGML_DIR}/ggml/src/ggml-cpu)
target_compile_definitions(strata_metal_host PUBLIC STRATA_NATIVE_EXPERTS=1 STRATA_VERSION="0.1.45-metal")
target_compile_options(strata_metal_host PRIVATE -Wno-unused-parameter -Wno-deprecated-declarations)
target_link_libraries(strata_metal_host PUBLIC strata_gpu_metal ggml-cpu ggml-base)

add_library(strata_metal_kernels STATIC ${STRATA_METAL_ENGINE_SOURCES})
target_include_directories(strata_metal_kernels PRIVATE src ${STRATA_ROOT}/third_party/ggml)
target_compile_definitions(strata_metal_kernels PRIVATE
  STRATA_GGML_COMMON_DEFAULT="${GGML_COMMON}" STRATA_METAL_KERNELS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/kernels"
  STRATA_METAL_ENGINE_KERNELS_DEFAULT="${CMAKE_CURRENT_SOURCE_DIR}/src/engine/msl"
  STRATA_METAL_DATA_DEFAULT="${STRATA_ROOT}/data")
target_link_libraries(strata_metal_kernels PUBLIC strata_metal_host mtl_rt)

add_executable(strata ${S}/program/generate.cpp)
target_link_libraries(strata PRIVATE strata_metal_kernels strata_metal_host strata_metal_kernels)

# The smaller-family engine keeps the shared line protocol, expert cache, CPU
# experts, batching, conversations and speculative decoding.
if(STRATA_METAL_POLY)
  target_sources(strata_poly_metal_kernels PRIVATE src/poly/quantized.cpp)
  target_include_directories(strata_poly_metal_kernels PRIVATE ${S}/poly ${STRATA_GGML_DIR}/ggml/include)
  target_compile_definitions(strata_poly_metal_kernels PRIVATE
    STRATA_POLY_GGML_SOURCE="${STRATA_GGML_DIR}/ggml/src"
    STRATA_POLY_QUANT_SOURCE="${CMAKE_CURRENT_SOURCE_DIR}/src/poly/quantized.metal")
  target_link_libraries(strata_poly_metal_kernels PUBLIC ggml-base)
  add_executable(strata-poly ${S}/poly/main.cpp ${S}/poly/engine.cpp ${S}/poly/qwen35.cpp
    ${S}/poly/gemma4.cpp ${S}/poly/lfm2moe.cpp ${S}/poly/glm5next.cpp ${S}/poly/deepseek4.cpp ${S}/poly/cpu_experts.cpp
    ${S}/core/conversation_memory.cpp ${S}/spec/draft_policy.cpp ${S}/spec/suffix_drafter.cpp)
  target_include_directories(strata-poly PRIVATE ${STRATA_GGML_DIR}/ggml/include ${STRATA_GGML_DIR}/ggml/src)
  target_link_libraries(strata-poly PRIVATE strata_poly_metal_kernels ggml-cpu ggml-base)
  target_compile_definitions(strata-poly PRIVATE STRATA_VERSION="0.1.45-metal-poly")
  set_source_files_properties(${S}/poly/norm_q8_test.cu PROPERTIES LANGUAGE CXX)
  add_executable(poly_norm_q8_metal_test ${S}/poly/norm_q8_test.cu)
  target_link_libraries(poly_norm_q8_metal_test PRIVATE strata_poly_metal_kernels)
  if(BUILD_TESTING)
    add_test(NAME poly_norm_q8_metal_test COMMAND poly_norm_q8_metal_test)
    set_tests_properties(poly_norm_q8_metal_test PROPERTIES LABELS "metal;gpu;poly" RESOURCE_LOCK metal_gpu TIMEOUT 120)
  endif()
endif()

if(STRATA_METAL_POLY)
  foreach(t attention quantized recurrent operations)
    add_executable(poly_${t}_parity poly_tests/${t}_parity.cpp)
    target_include_directories(poly_${t}_parity PRIVATE ${S}/poly)
    target_link_libraries(poly_${t}_parity PRIVATE strata_poly_metal_kernels)
    if(BUILD_TESTING)
      add_test(NAME poly_${t}_parity COMMAND poly_${t}_parity)
      set_tests_properties(poly_${t}_parity PROPERTIES LABELS "metal;gpu;poly" RESOURCE_LOCK metal_gpu TIMEOUT 120)
    endif()
  endforeach()
endif()

if(STRATA_METAL_POLY AND BUILD_TESTING)
  add_executable(poly_tiny_model_fixture poly_tests/tiny_model_fixture.cpp)
  target_link_libraries(poly_tiny_model_fixture PRIVATE ggml-base)
  add_test(NAME poly_tiny_lfm_fixture COMMAND poly_tiny_model_fixture ${CMAKE_CURRENT_BINARY_DIR}/tiny-lfm.gguf)
  set_tests_properties(poly_tiny_lfm_fixture PROPERTIES LABELS "metal;poly;synthetic" FIXTURES_SETUP tiny_lfm)
  foreach(mode graphs launches cpu_experts)
    set(extra)
    if(mode STREQUAL "launches")
      set(extra --no-graphs)
    elseif(mode STREQUAL "cpu_experts")
      set(extra --no-gpu-experts)
    endif()
    add_test(NAME poly_tiny_lfm_${mode} COMMAND strata-poly --model ${CMAKE_CURRENT_BINARY_DIR}/tiny-lfm.gguf
      --tokens 1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17 --max-new 4 --max-context 128
      --prompt-batch 32 --threads 2 --no-draft ${extra})
    set_tests_properties(poly_tiny_lfm_${mode} PROPERTIES LABELS "metal;gpu;poly;synthetic"
      RESOURCE_LOCK metal_gpu TIMEOUT 120 FIXTURES_REQUIRED tiny_lfm)
  endforeach()
endif()

if(STRATA_METAL_POLY AND BUILD_TESTING)
  add_executable(poly_memory_test poly_tests/memory_test.cpp ${S}/core/conversation_memory.cpp)
  target_include_directories(poly_memory_test PRIVATE ${STRATA_ROOT}/include)
  add_test(NAME poly_memory_test COMMAND poly_memory_test)
  set_tests_properties(poly_memory_test PROPERTIES LABELS "metal;poly;cpu")
  add_executable(poly_conversation_memory_test ${S}/core/conversation_memory_test.cpp ${S}/core/conversation_memory.cpp)
  target_include_directories(poly_conversation_memory_test PRIVATE ${STRATA_ROOT}/include)
  add_test(NAME poly_conversation_memory_test COMMAND poly_conversation_memory_test)
  set_tests_properties(poly_conversation_memory_test PROPERTIES LABELS "metal;poly;cpu")
  add_test(NAME poly_tiny_lfm_batch COMMAND strata-poly --model ${CMAKE_CURRENT_BINARY_DIR}/tiny-lfm.gguf
    --tokens 1,2,3,4,5 --max-new 8 --max-context 128 --prompt-batch 32 --threads 2 --no-draft --batch 2 --check-batch 2)
  set_tests_properties(poly_tiny_lfm_batch PROPERTIES LABELS "metal;gpu;poly;synthetic"
    RESOURCE_LOCK metal_gpu TIMEOUT 120 FIXTURES_REQUIRED tiny_lfm)
  find_package(Python3 COMPONENTS Interpreter REQUIRED)
  add_test(NAME poly_protocol_test COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/protocol_test.py
    $<TARGET_FILE:strata-poly> ${CMAKE_CURRENT_BINARY_DIR}/tiny-lfm.gguf)
  set_tests_properties(poly_protocol_test PROPERTIES LABELS "metal;gpu;poly;synthetic"
    RESOURCE_LOCK metal_gpu TIMEOUT 120 FIXTURES_REQUIRED tiny_lfm)
endif()

if(STRATA_METAL_POLY AND BUILD_TESTING)
  foreach(arch qwen35moe gemma4 glm5-next deepseek4)
    add_test(NAME poly_tiny_${arch}_fixture COMMAND poly_tiny_model_fixture ${CMAKE_CURRENT_BINARY_DIR}/tiny-${arch}.gguf ${arch})
    set_tests_properties(poly_tiny_${arch}_fixture PROPERTIES LABELS "metal;poly;synthetic" FIXTURES_SETUP tiny_${arch})
    add_test(NAME poly_tiny_${arch}_batch COMMAND strata-poly --model ${CMAKE_CURRENT_BINARY_DIR}/tiny-${arch}.gguf
      --tokens 1,2,3,4,5 --max-new 8 --max-context 128 --prompt-batch 32 --threads 2 --no-draft --batch 2 --check-batch 2)
    set_tests_properties(poly_tiny_${arch}_batch PROPERTIES LABELS "metal;gpu;poly;synthetic"
      RESOURCE_LOCK metal_gpu TIMEOUT 120 FIXTURES_REQUIRED tiny_${arch})
    add_test(NAME poly_tiny_${arch}_protocol COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/protocol_test.py
      $<TARGET_FILE:strata-poly> ${CMAKE_CURRENT_BINARY_DIR}/tiny-${arch}.gguf)
    set_tests_properties(poly_tiny_${arch}_protocol PROPERTIES LABELS "metal;gpu;poly;synthetic"
      RESOURCE_LOCK metal_gpu TIMEOUT 120 FIXTURES_REQUIRED tiny_${arch})
  endforeach()
  add_test(NAME poly_tiny_qwen_q8 COMMAND strata-poly --model ${CMAKE_CURRENT_BINARY_DIR}/tiny-qwen35moe.gguf
    --tokens 1,2,3,4,5 --max-new 8 --max-context 128 --prompt-batch 32 --threads 2 --no-draft --batch 2 --check-batch 2 --kv q8_0)
  set_tests_properties(poly_tiny_qwen_q8 PROPERTIES LABELS "metal;gpu;poly;synthetic"
    RESOURCE_LOCK metal_gpu TIMEOUT 120 FIXTURES_REQUIRED tiny_qwen35moe)
endif()

if(STRATA_METAL_POLY AND BUILD_TESTING)
  add_test(NAME poly_tiny_qwen_draft COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/protocol_test.py
    $<TARGET_FILE:strata-poly> ${CMAKE_CURRENT_BINARY_DIR}/tiny-qwen35moe.gguf --window 4)
  set_tests_properties(poly_tiny_qwen_draft PROPERTIES LABELS "metal;gpu;poly;synthetic"
    RESOURCE_LOCK metal_gpu TIMEOUT 120 FIXTURES_REQUIRED tiny_qwen35moe)
endif()

# strata-q36 retains its own host driver (MTP, batching and line protocol).
# The source adapter substitutes native Metal dispatch and keeps CUDA untouched.
if(STRATA_METAL_POLY)
  find_package(Python3 REQUIRED COMPONENTS Interpreter)
  set(Q36_GENERATED ${CMAKE_CURRENT_BINARY_DIR}/q36-metal)
  add_custom_command(OUTPUT ${Q36_GENERATED}/q36_host.cpp ${Q36_GENERATED}/q36_launch.hpp ${Q36_GENERATED}/q36.metal
    COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/tools/q36_host.py ${S}/qwen35/q36_engine.cu ${Q36_GENERATED}
    DEPENDS ${S}/qwen35/q36_engine.cu tools/q36_host.py VERBATIM)
  add_executable(strata-q36 ${Q36_GENERATED}/q36_host.cpp ${S}/qwen35/q36_main.cpp src/q36/bindings.cpp src/q36/cpu_compat.cpp)
  target_include_directories(strata-q36 PRIVATE ${Q36_GENERATED} src/q36 ${S}/poly ${STRATA_GGML_DIR}/ggml/include)
  target_compile_definitions(strata-q36 PRIVATE STRATA_Q36_CPU_POOL=1 STRATA_Q36_METAL_SOURCE="${Q36_GENERATED}/q36.metal")
  target_link_libraries(strata-q36 PRIVATE strata_metal_host strata_poly_metal_kernels ggml-cpu ggml-base)
endif()
if(STRATA_METAL_POLY AND BUILD_TESTING)
  add_executable(q36_metal_parity poly_tests/q36_parity.cpp ${Q36_GENERATED}/q36_launch.hpp src/q36/bindings.cpp)
  target_include_directories(q36_metal_parity PRIVATE ${Q36_GENERATED} src/q36 ${S}/poly)
  target_compile_definitions(q36_metal_parity PRIVATE STRATA_Q36_METAL_SOURCE="${Q36_GENERATED}/q36.metal")
  target_link_libraries(q36_metal_parity PRIVATE strata_poly_metal_kernels ggml-cpu ggml-base)
  add_test(NAME q36_metal_parity COMMAND q36_metal_parity)
  set_tests_properties(q36_metal_parity PROPERTIES LABELS "metal;gpu;parity;q36" RESOURCE_LOCK metal_gpu TIMEOUT 120)
endif()
if(STRATA_METAL_POLY AND BUILD_TESTING)
  add_executable(q36_metal_fixture poly_tests/q36_fixture.cpp)
  target_include_directories(q36_metal_fixture PRIVATE ${STRATA_ROOT}/include)
  target_link_libraries(q36_metal_fixture PRIVATE ggml-base)
  add_test(NAME q36_metal_fixture COMMAND q36_metal_fixture ${CMAKE_CURRENT_BINARY_DIR}/tiny-q36.gguf)
  set_tests_properties(q36_metal_fixture PROPERTIES LABELS "metal;synthetic;q36" FIXTURES_SETUP tiny_q36)
endif()
if(STRATA_METAL_POLY AND BUILD_TESTING)
  add_test(NAME q36_metal_protocol COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/q36_protocol_test.py
    $<TARGET_FILE:strata-q36> ${CMAKE_CURRENT_BINARY_DIR}/tiny-q36.gguf)
  set_tests_properties(q36_metal_protocol PROPERTIES LABELS "metal;gpu;synthetic;q36" RESOURCE_LOCK metal_gpu TIMEOUT 180 FIXTURES_REQUIRED tiny_q36)
endif()
if(STRATA_METAL_POLY AND BUILD_TESTING)
  add_test(NAME poly_tiny_glm_draft COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/protocol_test.py
    $<TARGET_FILE:strata-poly> ${CMAKE_CURRENT_BINARY_DIR}/tiny-glm5-next.gguf --window 4)
  set_tests_properties(poly_tiny_glm_draft PROPERTIES LABELS "metal;gpu;poly;synthetic" RESOURCE_LOCK metal_gpu TIMEOUT 120 FIXTURES_REQUIRED tiny_glm5-next)
  add_test(NAME poly_tiny_deepseek_compression COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/protocol_test.py
    $<TARGET_FILE:strata-poly> ${CMAKE_CURRENT_BINARY_DIR}/tiny-deepseek4.gguf --long-prompt)
  set_tests_properties(poly_tiny_deepseek_compression PROPERTIES LABELS "metal;gpu;poly;synthetic" RESOURCE_LOCK metal_gpu TIMEOUT 120 FIXTURES_REQUIRED tiny_deepseek4)
endif()
if(STRATA_METAL_POLY AND BUILD_TESTING)
  foreach(quant q4 q6)
    add_test(NAME q36_metal_fixture_${quant} COMMAND q36_metal_fixture ${CMAKE_CURRENT_BINARY_DIR}/tiny-q36-${quant}.gguf ${quant})
    set_tests_properties(q36_metal_fixture_${quant} PROPERTIES LABELS "metal;synthetic;q36" FIXTURES_SETUP tiny_q36_${quant})
    add_test(NAME q36_metal_protocol_${quant} COMMAND ${Python3_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/q36_protocol_test.py
      $<TARGET_FILE:strata-q36> ${CMAKE_CURRENT_BINARY_DIR}/tiny-q36-${quant}.gguf)
    set_tests_properties(q36_metal_protocol_${quant} PROPERTIES LABELS "metal;gpu;synthetic;q36" RESOURCE_LOCK metal_gpu TIMEOUT 180 FIXTURES_REQUIRED tiny_q36_${quant})
  endforeach()
endif()

if(STRATA_METAL_POLY AND BUILD_TESTING)
  target_compile_definitions(strata_gpu_metal PRIVATE STRATA_METAL_TESTING=1)
  add_executable(poly_runtime_safety poly_tests/runtime_safety.cpp)
  target_compile_definitions(poly_runtime_safety PRIVATE STRATA_METAL_TESTING=1)
  target_link_libraries(poly_runtime_safety PRIVATE strata_gpu_metal)
  add_test(NAME poly_runtime_failure COMMAND poly_runtime_safety failure)
  set_tests_properties(poly_runtime_failure PROPERTIES RESOURCE_LOCK metal_gpu TIMEOUT 30 LABELS "metal;gpu;poly")
  add_test(NAME poly_runtime_safety COMMAND poly_runtime_safety)
  set_tests_properties(poly_runtime_safety PROPERTIES RESOURCE_LOCK metal_gpu TIMEOUT 60 LABELS "metal;gpu;poly")
endif()

if(STRATA_METAL_POLY AND BUILD_TESTING)
  get_target_property(poly_engine_test_sources strata-poly SOURCES)
  list(REMOVE_ITEM poly_engine_test_sources ${S}/poly/main.cpp)
  add_library(poly_engine_test_host STATIC ${poly_engine_test_sources})
  target_include_directories(poly_engine_test_host PUBLIC ${S}/poly ${STRATA_GGML_DIR}/ggml/include ${STRATA_GGML_DIR}/ggml/src)
  target_link_libraries(poly_engine_test_host PUBLIC strata_poly_metal_kernels ggml-cpu ggml-base)
  add_executable(poly_memory_growth poly_tests/memory_growth.cpp)
  target_compile_definitions(poly_memory_growth PRIVATE STRATA_METAL_TESTING=1)
  target_link_libraries(poly_memory_growth PRIVATE poly_engine_test_host)
  add_test(NAME poly_memory_growth COMMAND poly_memory_growth)
  set_tests_properties(poly_memory_growth PROPERTIES RESOURCE_LOCK metal_gpu TIMEOUT 60 LABELS "metal;gpu;poly")
  foreach(arch qwen35moe gemma4 glm5-next deepseek4)
    add_test(NAME poly_balanced_${arch}_protocol COMMAND ${Python3_EXECUTABLE}
      ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/protocol_test.py $<TARGET_FILE:strata-poly>
      ${CMAKE_CURRENT_BINARY_DIR}/tiny-${arch}.gguf --metal-optimize --metal-memory balanced --no-draft)
    set_tests_properties(poly_balanced_${arch}_protocol PROPERTIES RESOURCE_LOCK metal_gpu TIMEOUT 120
      LABELS "metal;gpu;poly;synthetic" FIXTURES_REQUIRED tiny_${arch})
  endforeach()
  add_test(NAME poly_balanced_qwen_q8_protocol COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/protocol_test.py $<TARGET_FILE:strata-poly>
    ${CMAKE_CURRENT_BINARY_DIR}/tiny-qwen35moe.gguf --metal-optimize --metal-memory balanced --kv q8_0)
  set_tests_properties(poly_balanced_qwen_q8_protocol PROPERTIES RESOURCE_LOCK metal_gpu TIMEOUT 120
    LABELS "metal;gpu;poly;synthetic" FIXTURES_REQUIRED tiny_qwen35moe)
endif()

if(STRATA_METAL_POLY AND BUILD_TESTING)
  add_test(NAME poly_tiny_qwen_fp16_fixture COMMAND poly_tiny_model_fixture
    ${CMAKE_CURRENT_BINARY_DIR}/tiny-qwen35moe-fp16.gguf qwen35moe-fp16)
  set_tests_properties(poly_tiny_qwen_fp16_fixture PROPERTIES LABELS "metal;poly;synthetic"
    FIXTURES_SETUP tiny_qwen_fp16)
  add_test(NAME poly_metal_defaults COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/defaults_test.py $<TARGET_FILE:strata-poly> ${CMAKE_CURRENT_BINARY_DIR})
  set_tests_properties(poly_metal_defaults PROPERTIES RESOURCE_LOCK metal_gpu TIMEOUT 180
    LABELS "metal;gpu;poly;synthetic"
    FIXTURES_REQUIRED "tiny_lfm;tiny_qwen35moe;tiny_gemma4;tiny_glm5-next;tiny_deepseek4;tiny_qwen_fp16")
  add_test(NAME poly_legacy_qwen_protocol COMMAND ${Python3_EXECUTABLE}
    ${CMAKE_CURRENT_SOURCE_DIR}/poly_tests/protocol_test.py $<TARGET_FILE:strata-poly>
    ${CMAKE_CURRENT_BINARY_DIR}/tiny-qwen35moe.gguf --no-metal-optimize --metal-memory legacy --kv fp16)
  set_tests_properties(poly_legacy_qwen_protocol PROPERTIES RESOURCE_LOCK metal_gpu TIMEOUT 120
    LABELS "metal;gpu;poly;synthetic" FIXTURES_REQUIRED tiny_qwen35moe)
endif()
