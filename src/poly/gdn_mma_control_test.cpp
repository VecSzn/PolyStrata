// Host-only opt-in lifecycle; no CUDA context or model is needed.
#include "gdn_mma_control.hpp"
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <optional>
#include <stdexcept>
#include <string>

using strata::poly::GdnMmaControl;
namespace {
constexpr const char* key = "STRATA_GDN_MMA";
void set(const char* value) {
#ifdef _WIN32
    if (_putenv_s(key, value ? value : "") != 0) throw std::runtime_error("set environment");
#else
    if (value ? setenv(key, value, 1) != 0 : unsetenv(key) != 0) throw std::runtime_error("set environment");
#endif
}
struct Restore {
    std::optional<std::string> old;
    Restore() { if (const char* value = std::getenv(key)) old = value; }
    ~Restore() { set(old ? old->c_str() : nullptr); }
};
void require(bool ok, const char* why) { if (!ok) throw std::runtime_error(why); }
void parse(const char* value, bool expected) {
    set(value);
    GdnMmaControl control;
    control.capture_request();
    int probes = 0;
    const bool report = control.initialize([&] { ++probes; return true; });
    require(report == expected && control.enabled() == expected && probes == int(expected),
            "environment request parsing: only atoi(value)==1 enables");
}
}
int main() {
    Restore restore;
    try {
        for (const char* value : std::initializer_list<const char*>{nullptr, "", "0", "2", "true", "-1"}) parse(value, false);
        for (const char* value : {"1", "01", " 1", "1suffix"}) parse(value, true);
        for (bool initially_on : {false, true}) {
            set(initially_on ? "1" : "0");
            GdnMmaControl control;
            control.capture_request();
            set(initially_on ? "0" : "1");
            control.capture_request();
            int probes = 0;
            require(control.initialize([&] { ++probes; return true; }) == initially_on,
                    "capture request once per engine");
            require(control.enabled() == initially_on && probes == int(initially_on), "captured choice changed");
        }
        for (bool available : {false, true}) {
            set("1");
            GdnMmaControl control;
            control.capture_request();
            int probes = 0, reports = 0;
            for (int scratch_pass = 0; scratch_pass < 2; ++scratch_pass)
                reports += control.initialize([&] { ++probes; return available; });
            require(probes == 1 && reports == 1 && control.enabled() == available,
                    "double scratch setup must probe/report once even on fallback");
        }
        std::puts("PASS GDN opt-in: parsing, captured request, disabled no-probe and single setup/report");
        return 0;
    } catch (const std::exception& e) { std::fprintf(stderr, "FAIL %s\n", e.what()); return 1; }
}
