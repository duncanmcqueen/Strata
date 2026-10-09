// tests/sycl/device_profile_test.cpp - synthetic-profile unit tests for the section 1A dispatch table.
//
// No GPU and no device query: every case builds a DeviceProfile by hand and exercises the pure selectors.  This
// is the part of the device matrix that must hold on any machine, including an A770-only one.
#include "strata/sycl_runtime/device_profile.hpp"

#include <cstdio>
#include <cstring>
#include <limits>
#include <string>

namespace dp = strata::sycl_runtime;

namespace {

int g_fail = 0;
void check(bool ok, const char* what) {
    if (!ok) {
        std::fprintf(stderr, "FAIL: %s\n", what);
        ++g_fail;
    }
}

dp::DeviceProfile make_profile(dp::DeviceArch arch, uint32_t id, const char* drv = "26.31.39395") {
    dp::DeviceProfile p;
    p.ordinal = 0;
    p.vendor_id = 0x8086;
    p.device_id = id;
    p.device_id_available = true;
    p.arch = arch;
    p.driver_version = drv;
    p.compiler_id = "icpx test";
    p.name = (std::string("synthetic-") + dp::arch_name(arch));
    p.subgroup_sizes = {16, 32};
    p.max_workgroup_size = 1024;
    p.local_mem_bytes = 65536;
    p.global_mem_bytes = 16ull << 30;
    // generic scalar families are compiled in this build
    p.topk_hier = dp::KernelFamily{true, false, false};
    p.scores_tiled = dp::KernelFamily{true, false, false};
    p.prompt_attn_tiled = dp::KernelFamily{true, false, false};
    p.grouping_device = dp::KernelFamily{true, false, false};
    p.experts_bounded = dp::KernelFamily{true, false, false};
    p.scores_xmx = dp::KernelFamily{false, false, false};
    p.prompt_attn_xmx = dp::KernelFamily{false, false, false};
    p.experts_xmx = dp::KernelFamily{false, false, false};
    return p;
}

dp::SelectorInput base(const dp::DeviceProfile& d) {
    dp::SelectorInput in;
    in.device = &d;
    in.shape.queries = 1;
    in.shape.max_blocks = 32770;
    in.shape.active_blocks = -1;
    in.shape.cells = 131072;
    in.shape.width = 2048;
    return in;
}

dp::TuningRecord topk_record(dp::DeviceArch arch, const char* kernel, int64_t lo, int64_t hi) {
    dp::TuningRecord r;
    r.operation = "topk";
    r.kernel = kernel;
    r.key.arch = arch;
    r.key.arch_set = true;
    r.key.compiler = "icpx test";
    r.key.driver = "26.31.39395";
    r.key.bucket_lo = lo;
    r.key.bucket_hi = hi;
    r.value = 32768;
    return r;
}

// 1. Architecture classification is capability/ID based and name-free.
void classification() {
    dp::DeviceArch a{};
    check(dp::device_id_to_arch(0x8086, 0xE223, a) && a == dp::DeviceArch::Bmg, "B70 id -> bmg");
    check(dp::device_id_to_arch(0x8086, 0xE20B, a) && a == dp::DeviceArch::Bmg, "B580 id -> bmg");
    check(dp::device_id_to_arch(0x8086, 0x56A0, a) && a == dp::DeviceArch::Dg2, "A770 id -> dg2");
    check(dp::device_id_to_arch(0x8086, 0x5690, a) && a == dp::DeviceArch::Dg2, "ATS id -> dg2");
    check(!dp::device_id_to_arch(0x8086, 0x7D55, a), "iGPU id -> unknown");
    check(!dp::device_id_to_arch(0x10DE, 0x56A0, a), "non-Intel vendor -> unknown");
}

// 2. Established default: small shape -> legacy, no matter the architecture or extra records.
void default_is_legacy_for_small_shapes() {
    for (auto arch : {dp::DeviceArch::Bmg, dp::DeviceArch::Dg2, dp::DeviceArch::Unknown}) {
        auto p = make_profile(arch, arch == dp::DeviceArch::Bmg ? 0xE223u : 0x56A0u);
        auto in = base(p);
        in.shape.max_blocks = 33000;
        in.shape.cells = 131000;
        auto c = dp::select_topk(in);
        check(c.ok && c.kernel == "legacy", "small shape -> legacy");
        check(!c.fallback, "small shape legacy is not a fallback");
    }
}

// 3. B70 with a qualified record selects its measured winner; A770 does not inherit it.
void per_device_winners_are_independent() {
    auto b70 = make_profile(dp::DeviceArch::Bmg, 0xE223);
    auto a770 = make_profile(dp::DeviceArch::Dg2, 0x56A0);
    dp::TuningTable t;
    t.push_back(topk_record(dp::DeviceArch::Bmg, "hierarchical", 30000, 0));

    dp::SelectorInput inb = base(b70);
    inb.tuning = &t;
    auto cb = dp::select_topk(inb);
    check(cb.ok && cb.kernel == "hierarchical", "B70 qualified record -> hierarchical");
    check(cb.scratch_bytes > 0, "hierarchical reports workspace bytes");

    dp::SelectorInput ina = base(a770);
    ina.tuning = &t;  // same table copied to A770
    auto ca = dp::select_topk(ina);
    check(ca.ok && ca.kernel != "hierarchical", "B70-only record ignored on A770");
    check(ca.kernel == "legacy" || ca.kernel == "reference", "A770 falls back to its own path");
}

// 4. Unknown architecture always takes the safe established path with a clear reason.
void unknown_architecture_is_safe() {
    auto p = make_profile(dp::DeviceArch::Unknown, 0x7D55);
    auto c = dp::select_topk(base(p));
    check(c.ok && c.kernel == "legacy", "unknown arch -> legacy");
    check(!c.reason.empty(), "unknown arch has a reason");
    auto c2 = dp::select_grouping(base(p));
    check(c2.ok && c2.kernel == "host", "unknown arch grouping -> host");
    auto c3 = dp::select_experts(base(p));
    check(c3.ok && c3.kernel == "legacy", "unknown arch experts -> legacy");
}

// 5. Beyond the legacy register capacity: BMG's built-in qualified record (few queries) -> hierarchical;
//    dg2 has no measured record -> reference fallback.
void long_context_dispatch() {
    auto b70 = make_profile(dp::DeviceArch::Bmg, 0xE223);
    auto in = base(b70);
    in.shape.max_blocks = 70000;  // > 33792
    in.shape.cells = 280000;
    in.shape.queries = 8;         // few queries: the measured winner
    auto c = dp::select_topk(in);
    check(c.ok && c.kernel == "hierarchical", "B70 built-in record: few queries long context -> hierarchical");
    // many queries are NOT covered by the built-in record: reference fallback.
    in.shape.queries = 256;
    auto cm = dp::select_topk(in);
    check(cm.ok && cm.kernel == "reference", "B70 many-query long context -> reference");
    check(cm.fallback, "many-query reference flagged fallback");

    auto a770 = make_profile(dp::DeviceArch::Dg2, 0x56A0);
    auto ina = base(a770);
    ina.shape.max_blocks = 70000;
    ina.shape.cells = 280000;
    ina.shape.queries = 8;
    auto ca = dp::select_topk(ina);
    check(ca.ok && ca.kernel == "reference", "A770 has no record -> reference fallback");
}

// 6. Active bound does not change the physical stride; register kernel chosen by the active bound.
void active_bound_stride_semantics() {
    auto p = make_profile(dp::DeviceArch::Bmg, 0xE223);
    auto in = base(p);
    in.shape.max_blocks = 262144;   // huge physical stride
    in.shape.active_blocks = 33000; // live bound fits the register kernel
    in.shape.cells = 131000;
    auto c = dp::select_topk(in);
    check(c.ok && c.kernel == "legacy", "active bound within register capacity -> legacy at large stride");
    // Without the bound the same stride needs a fallback.
    in.shape.active_blocks = -1;
    auto c2 = dp::select_topk(in);
    check(c2.ok && (c2.kernel == "hierarchical" || c2.kernel == "reference"),
          "no bound at large stride -> long-context path");
}

// 7. Explicit unsupported force is rejected; a supported force is honored.
void overrides() {
    auto p = make_profile(dp::DeviceArch::Dg2, 0x56A0);
    auto in = base(p);
    in.shape.max_blocks = 70000;
    in.override_value = "legacy";
    auto c = dp::select_topk(in);
    check(!c.ok && c.rejected_override, "unsupported forced legacy rejected before launch");

    in.override_value = "hierarchical";
    in.scratch_budget = 1024;  // far too small
    auto c2 = dp::select_topk(in);
    check(!c2.ok && c2.rejected_override, "forced hierarchical with tiny workspace rejected");

    in.override_value = "bogus";
    auto c3 = dp::select_topk(in);
    check(!c3.ok && c3.rejected_override, "unknown override value rejected");

    in.override_value = "reference";
    auto c4 = dp::select_topk(in);
    check(c4.ok && c4.kernel == "reference", "explicit reference honored");
}

// 8. A missing/failed matrix capability excludes the matrix candidate; auto stays usable.
void matrix_capability_gating() {
    auto p = make_profile(dp::DeviceArch::Dg2, 0x56A0);  // no qualified XMX implementation
    auto in = base(p);
    in.shape.format = dp::kFmtFp16Input;  // even when the caller allows a precision change
    auto c = dp::select_prompt_attn(in);
    check(c.ok && c.kernel == "legacy", "A770 without XMX -> legacy prompt attention");
    in.override_value = "xmx";
    auto c2 = dp::select_prompt_attn(in);
    check(!c2.ok && c2.rejected_override, "forced xmx without capability rejected");
    // A failed correctness check also excludes a compiled family.
    auto p2 = p;
    p2.prompt_attn_tiled = dp::KernelFamily{true, true, false};  // checked, failed
    in.override_value = "";
    auto c3 = dp::select_prompt_attn(in);
    check(c3.kernel != "tiled", "failed capability check excludes tiled family");
}

// 9. Stale measurements are invalidated by driver/compiler changes.
void fingerprint_invalidation() {
    auto p = make_profile(dp::DeviceArch::Bmg, 0xE223, "27.00.00001");
    dp::TuningTable t;
    t.push_back(topk_record(dp::DeviceArch::Bmg, "hierarchical", 30000, 0));  // driver "26.31..."
    auto in = base(p);
    in.tuning = &t;
    auto c = dp::select_topk(in);
    check(c.kernel != "hierarchical", "record with stale driver does not apply");

    auto p2 = make_profile(dp::DeviceArch::Bmg, 0xE223, "26.31.39395");
    auto p3 = p2;
    p3.compiler_id = "icpx other";
    auto in3 = base(p3);
    in3.tuning = &t;
    check(dp::select_topk(in3).kernel != "hierarchical", "record with stale compiler does not apply");

    auto in2 = base(p2);
    in2.tuning = &t;
    check(dp::select_topk(in2).kernel == "hierarchical", "matching fingerprint applies");
}

// 10. Workspace arithmetic and 32-bit count safety.
void workspace_math() {
    const uint64_t a = dp::topk_hier_scratch_bytes(256, 262144, 1024);
    check(a != UINT64_MAX && a > 0, "scratch computes");
    // 256 queries * 256 tiles * 256 bins * 4 bytes = 64 MiB
    check(a >= (uint64_t)256 * 256 * 256 * 4, "scratch at least histogram size");
    check(dp::topk_hier_scratch_bytes(1, 0) == 0, "zero blocks -> zero scratch");
    check(dp::topk_hier_scratch_bytes(1, 1, 1024) != UINT64_MAX, "small case fits");
    check(!dp::topk_hier_counts_are_32bit_safe(1, (int64_t)std::numeric_limits<int32_t>::max()), "huge context unsafe");
    check(dp::topk_hier_counts_are_32bit_safe(256, 262144), "262144 context safe");
    check(dp::topk_hier_tile_count(262144, 1024) == 256, "tile count");
}

// 11. Workspace budget below the requirement selects a safe path, not a crash.
void workspace_budget() {
    auto p = make_profile(dp::DeviceArch::Bmg, 0xE223);
    auto in = base(p);
    in.shape.max_blocks = 70000;
    in.scratch_budget = 4096;  // too small
    in.tuning = nullptr;
    auto c = dp::select_topk(in);
    check(c.ok && c.kernel == "reference", "insufficient workspace -> reference fallback");
}

// 12. Grouping and experts defaults; device grouping only when compiled+qualified.
void grouping_and_experts() {
    auto p = make_profile(dp::DeviceArch::Bmg, 0xE223);
    p.grouping_device = dp::KernelFamily{false, false, false};  // not compiled in this build
    auto in = base(p);
    check(dp::select_grouping(in).kernel == "host", "auto grouping is host until qualified");
    in.override_value = "device";
    check(!dp::select_grouping(in).ok, "forced device grouping rejected while unqualified");
    in.override_value = "";
    check(dp::select_experts(in).kernel == "legacy", "auto experts is legacy until qualified");
    // With a compiled+correct device grouping family, auto still waits for a qualified record.
    auto p2 = p;
    p2.grouping_device = dp::KernelFamily{true, true, true};
    auto in2 = base(p2);
    check(dp::select_grouping(in2).kernel == "host", "auto grouping stays host without a record");
    in2.override_value = "device";
    check(dp::select_grouping(in2).ok, "forced device grouping honored when compiled+qualified");
}

// 13. Refinement cannot relax eligibility: a record recommending an unavailable kernel falls back.
void records_cannot_relax() {
    auto p = make_profile(dp::DeviceArch::Bmg, 0xE223);
    p.scores_tiled = dp::KernelFamily{false, false, false};  // not compiled
    dp::TuningTable t;
    dp::TuningRecord r;
    r.operation = "scores";
    r.kernel = "tiled";
    r.key.arch = dp::DeviceArch::Bmg;
    r.key.arch_set = true;
    r.key.compiler = "icpx test";
    r.key.driver = "26.31.39395";
    t.push_back(r);
    auto in = base(p);
    in.tuning = &t;
    auto c = dp::select_scores(in);
    check(c.ok && c.kernel == "legacy", "record cannot enable an uncompiled kernel");
}

}  // namespace

int main() {
    classification();
    default_is_legacy_for_small_shapes();
    per_device_winners_are_independent();
    unknown_architecture_is_safe();
    long_context_dispatch();
    active_bound_stride_semantics();
    overrides();
    matrix_capability_gating();
    fingerprint_invalidation();
    workspace_math();
    workspace_budget();
    grouping_and_experts();
    records_cannot_relax();
    if (g_fail == 0) std::printf("device_profile_test: PASS (13 groups)\n");
    else std::printf("device_profile_test: FAIL (%d checks)\n", g_fail);
    return g_fail == 0 ? 0 : 1;
}
