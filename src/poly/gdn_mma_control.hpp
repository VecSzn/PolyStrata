// Per-engine GDN opt-in lifecycle, independent of the CUDA kernel implementation.
#pragma once
#include <cstdlib>
namespace strata::poly {
class GdnMmaControl {
public:
    void capture_request() {
        if (captured_) return;
        const char* value = std::getenv("STRATA_GDN_MMA");
        requested_ = value && std::atoi(value) == 1;
        captured_ = true;
    }
    // True once when the caller should report its capability decision. The
    // counting and allocation passes both call this, including on fallback.
    template<class Available> bool initialize(Available available) {
        if (!requested_ || checked_) return false;
        checked_ = true;
        enabled_ = available();
        return true;
    }
    bool enabled() const { return enabled_; }
private:
    bool captured_ = false, requested_ = false, checked_ = false, enabled_ = false;
};
} // namespace strata::poly
