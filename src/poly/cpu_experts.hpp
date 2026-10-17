// src/poly/cpu_experts.hpp - routed experts computed on the CPU from their GGUF bytes.
//
// The dot products and the activation quantizers are ggml-cpu's (ggml_get_type_traits_cpu), so an expert here
// computes what llama.cpp's CPU backend computes for the same tensor.  The workers are this file's own: the pool
// in strata/kernels/cpu/pool.hpp is built around the other model's expert blob.
#pragma once

#include <atomic>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace strata::poly {

/// Runs fn(0..n-1) on the workers and the calling thread, and returns when all of them are done.  A call from
/// inside fn does its work on the thread it is on.
class Workers {
public:
    explicit Workers(int threads);
    ~Workers();
    Workers(const Workers&) = delete;
    Workers& operator=(const Workers&) = delete;

    void run(int n, const std::function<void(int)>& fn);
    int threads() const { return (int) th_.size() + 1; }

private:
    struct Batch {
        const std::function<void(int)>* fn;
        int n;
        std::atomic<int> next{0}, left{0};
    };
    void loop();
    static bool drain(Batch& b);

    std::vector<std::thread> th_;
    // every idle worker reads cur_ in a loop; users_ is written, so it has a cache line of its own
    alignas(64) std::atomic<Batch*> cur_{nullptr};
    alignas(64) std::atomic<int> users_{0};      ///< workers that may hold a Batch pointer
    std::atomic<int> sleepers_{0};
    std::mutex one_;                             ///< one batch at a time: on several cards, each card's thread posts
    std::atomic<uint32_t> posted_{0};            ///< counts the batches: what a sleeping worker waits on
    std::atomic<bool> quit_{false};
};

/// One routed expert of a token: its three matrices in the GGUF, its router weight, and which of the call's tokens
/// it is for.
struct CpuExpert {
    const uint8_t* gate;
    const uint8_t* up;
    const uint8_t* down;
    float weight;
    int row = 0;
};

/// Checks that ggml-cpu has a dot product for both types.  Call once per layer format at load.
bool cpu_expert_types_ok(int gu_type, int d_type, int n_embd, int n_ff, std::string& err);


/// For each of `rows` tokens, with n_embd values of x and of out each:
/// out = sum over the token's experts of weight * down(silu(min(gate x, limit)) * clamp(up x, -limit, limit)),
/// or with `gelu` of weight * down(gelu(gate x) * up x), in the tanh form of GELU the GPU experts compute.
/// The n experts come token after token, a token's together.  An expert that several of the tokens use is read
/// once for all of them: reading it is most of the work.  A token's result is the one it has in a call of its own.
/// `meanwhile`, when given, is run once by one of the threads, beside the first products.  It must not throw.
void cpu_experts(Workers& workers, const CpuExpert* experts, int n, int rows, int gu_type, int d_type, int n_embd,
                 int n_ff, const float* x, float limit, bool gelu, float* out,
                 const std::function<void()>* meanwhile = nullptr);

}  // namespace strata::poly
