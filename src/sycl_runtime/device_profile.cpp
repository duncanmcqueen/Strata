// src/sycl_runtime/device_profile.cpp - architecture-aware dispatch (spec section 1A).
//
// The device query lives here; the selection functions are pure and are unit-tested with synthetic profiles.
#include "strata/sycl_runtime/device_profile.hpp"

#include "../sycl_runtime/internal.hpp"

#include <mutex>
#include <set>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

#if defined(__INTEL_LLVM_COMPILER) || defined(__clang__)
#define STRATA_SYCL_COMPILER_ID "icpx " __VERSION__
#else
#define STRATA_SYCL_COMPILER_ID __VERSION__
#endif

// Which kernel families this build compiles.  Set by CMake on the runtime target so the profile reflects the
// actual build, not a claim.  XMX families stay off until a kernel is implemented AND qualified.
#ifndef STRATA_SYCL_HAVE_TOPK_HIER
#define STRATA_SYCL_HAVE_TOPK_HIER 1
#endif
#ifndef STRATA_SYCL_HAVE_SCORES_TILED
#define STRATA_SYCL_HAVE_SCORES_TILED 0
#endif
#ifndef STRATA_SYCL_HAVE_PROMPT_ATTN_TILED
#define STRATA_SYCL_HAVE_PROMPT_ATTN_TILED 0
#endif
#ifndef STRATA_SYCL_HAVE_GROUPING_DEVICE
#define STRATA_SYCL_HAVE_GROUPING_DEVICE 0
#endif
#ifndef STRATA_SYCL_HAVE_EXPERTS_BOUNDED
#define STRATA_SYCL_HAVE_EXPERTS_BOUNDED 0
#endif

namespace strata::sycl_runtime {
namespace {

constexpr int64_t kLegacyRegisterCap = 1024 * 33;  // TK_T * TK_PER in qsa_select.cpp
constexpr int64_t kTopkTileDefault = 1024;

std::string normalize_override(const std::string& v) {
    if (v.empty() || v == "auto") return "auto";
    return v;
}

KernelFamily family(bool compiled) {
    KernelFamily f;
    f.compiled = compiled;
    return f;
}

}  // namespace

const char* arch_name(DeviceArch a) {
    switch (a) {
    case DeviceArch::Dg2: return "dg2";
    case DeviceArch::Bmg: return "bmg";
    default: return "unknown";
    }
}

bool device_id_to_arch(uint32_t vendor_id, uint32_t device_id, DeviceArch& out) {
    if (vendor_id != 0x8086u) return false;
    // Alchemist / DG2: PCI device IDs 0x56xx (A310/A380/A750/A770, ATS server parts, mobile).  This is the
    // A770 family the ordinary build targets with dg2-g10.
    if ((device_id & 0xFF00u) == 0x5600u) {
        out = DeviceArch::Dg2;
        return true;
    }
    // Battlemage / BMG: 0xE2xx (Arc B570/B580, Pro B50/B60/B70, BMG-G21/G31).  B70 is BMG-G31.
    if ((device_id & 0xFF00u) == 0xE200u) {
        out = DeviceArch::Bmg;
        return true;
    }
    // Everything else (integrated Xe-LPG/Xe2 iGPUs, future parts) is Unknown on purpose: it takes the
    // established safe path.  Do NOT guess an architecture from a name.
    return false;
}

// ---- device queries ----------------------------------------------------------------------------------

namespace {

std::mutex g_profile_mu;
std::vector<DeviceProfile> g_profiles;
bool g_profiles_built = false;

DeviceProfile build_profile(const sycl::device& d, int ordinal) {
    DeviceProfile p;
    p.ordinal = ordinal;
    p.compiler_id = STRATA_SYCL_COMPILER_ID;
    auto try_get = [&](auto fn) {
        try {
            return fn();
        } catch (...) {
            return decltype(fn()){};
        }
    };
    p.name = try_get([&] { return d.get_info<sycl::info::device::name>(); });
    p.driver_version = try_get([&] { return d.get_info<sycl::info::device::driver_version>(); });
    try {
        p.vendor_id = d.get_info<sycl::info::device::vendor_id>();
    } catch (...) {
    }
    try {
        const auto sizes = d.get_info<sycl::info::device::sub_group_sizes>();
        p.subgroup_sizes.assign(sizes.begin(), sizes.end());
    } catch (...) {
    }
    try {
        p.max_workgroup_size = d.get_info<sycl::info::device::max_work_group_size>();
    } catch (...) {
    }
    try {
        p.local_mem_bytes = d.get_info<sycl::info::device::local_mem_size>();
    } catch (...) {
    }
    try {
        p.global_mem_bytes = d.get_info<sycl::info::device::global_mem_size>();
    } catch (...) {
    }
    try {
        p.device_id = d.get_info<sycl::ext::intel::info::device::device_id>();
        p.device_id_available = true;
    } catch (...) {
    }
    try {
        p.eu_count = d.get_info<sycl::ext::intel::info::device::gpu_eu_count>();
    } catch (...) {
    }
    try {
        p.slices = d.get_info<sycl::ext::intel::info::device::gpu_slices>();
    } catch (...) {
    }
    if (p.device_id_available) device_id_to_arch(p.vendor_id, p.device_id, p.arch);

    // Kernel family availability from the build.  Correctness-gated families start unprobed (usable() then
    // requires only compilation; an in-application check raises capability_checked later).
    p.topk_hier = family(STRATA_SYCL_HAVE_TOPK_HIER != 0);
    p.scores_tiled = family(STRATA_SYCL_HAVE_SCORES_TILED != 0);
    p.prompt_attn_tiled = family(STRATA_SYCL_HAVE_PROMPT_ATTN_TILED != 0);
    p.grouping_device = family(STRATA_SYCL_HAVE_GROUPING_DEVICE != 0);
    p.experts_bounded = family(STRATA_SYCL_HAVE_EXPERTS_BOUNDED != 0);
    // Matrix candidates: no compiled XMX kernel exists in this build.  caps remain false until an
    // application-integrated probe validates the actual compiled image (never inferred from a B70 probe).
    p.scores_xmx = family(false);
    p.prompt_attn_xmx = family(false);
    p.experts_xmx = family(false);
    return p;
}

}  // namespace

void warm_device_profiles() {
    std::lock_guard<std::mutex> lock(g_profile_mu);
    if (g_profiles_built) return;
    Registry& reg = registry();
    g_profiles.clear();
    g_profiles.reserve(reg.devices.size());
    for (size_t i = 0; i < reg.devices.size(); ++i)
        g_profiles.push_back(build_profile(reg.devices[i], static_cast<int>(i)));
    g_profiles_built = true;
}

const DeviceProfile& profile_for_device(int ordinal) {
    warm_device_profiles();
    std::lock_guard<std::mutex> lock(g_profile_mu);
    if (ordinal < 0 || ordinal >= static_cast<int>(g_profiles.size())) {
        static const DeviceProfile unknown{};
        return unknown;
    }
    return g_profiles[static_cast<size_t>(ordinal)];
}

const DeviceProfile& profile_for_stream(void* stream) {
    // The stream selects the device; nullptr = the current device (mirrors cudaGetDevice semantics).
    int ordinal = current_device();
    if (stream != nullptr) {
        strata_cuda_stream* s = stream_or_default((cudaStream_t)stream);
        if (s != nullptr) ordinal = s->device;
    }
    return profile_for_device(ordinal);
}

int profile_device_count() {
    warm_device_profiles();
    std::lock_guard<std::mutex> lock(g_profile_mu);
    return static_cast<int>(g_profiles.size());
}

// ---- workspace sizing (shared with the W1 kernel) ----------------------------------------------------

int64_t topk_hier_tile_count(int64_t max_blocks, int64_t tile_blocks) {
    if (max_blocks <= 0 || tile_blocks <= 0) return 0;
    return (max_blocks + tile_blocks - 1) / tile_blocks;
}

uint64_t topk_hier_scratch_bytes(int64_t queries, int64_t max_blocks, int64_t tile_blocks) {
    if (queries <= 0 || max_blocks <= 0 || tile_blocks <= 0) return 0;
    const int64_t tiles = topk_hier_tile_count(max_blocks, tile_blocks);
    const uint64_t I64MAX = (uint64_t)std::numeric_limits<int64_t>::max();
    // histogram counts: queries * tiles * 256 32-bit ints
    if ((uint64_t)queries > I64MAX / (uint64_t)tiles) return UINT64_MAX;
    uint64_t hist = (uint64_t)queries * (uint64_t)tiles;
    if (hist > I64MAX / 256) return UINT64_MAX;
    hist *= 256;
    // per-query state (8 ints) + per (query,tile) gt/eq totals, admit, output start (4 ints)
    uint64_t state = (uint64_t)queries * 8;
    if ((uint64_t)queries > I64MAX / (uint64_t)tiles) return UINT64_MAX;
    state += (uint64_t)queries * (uint64_t)tiles * 4;
    if (hist > (I64MAX - state)) return UINT64_MAX;
    return (hist + state) * sizeof(int32_t);
}

bool topk_hier_counts_are_32bit_safe(int64_t queries, int64_t cells) {
    // A tile histogram bin holds at most its tile's cells; the batch's total cells bound the reduction.  Require
    // the sum for one query (<= cells) to leave headroom under INT32_MAX.
    if (queries <= 0 || cells < 0) return false;
    return cells <= (int64_t)std::numeric_limits<int32_t>::max() / 2;
}

// ---- tuning records ----------------------------------------------------------------------------------

bool tuning_record_matches(const TuningRecord& r, const DeviceProfile& d, const OpShape& s) {
    if (r.key.arch_set && r.key.arch != d.arch) return false;
    if (r.key.device_id_set && (!d.device_id_available || r.key.device_id != d.device_id)) return false;
    if (!r.key.compiler.empty() && r.key.compiler != d.compiler_id) return false;
    if (!r.key.driver.empty() && r.key.driver != d.driver_version) return false;
    // model / kv_mode are supplied by the caller's key; the selector has no model state, so a record that names
    // one is only rejected when the caller explicitly disagrees.  Here we require shape-bucket agreement.
    if (r.key.max_queries != 0 && s.queries > r.key.max_queries) return false;
    if (r.key.max_context != 0 && s.cells > r.key.max_context) return false;
    if (r.key.bucket_lo != 0 || r.key.bucket_hi != 0) {
        // bucket on the EFFECTIVE work size: a small live bound is small work, whatever the capacity stride is.
        const int64_t v = (s.active_blocks > 0 && s.active_blocks <= s.max_blocks)
                              ? s.active_blocks
                              : (s.max_blocks > 0 ? s.max_blocks : s.rows);
        if (r.key.bucket_lo != 0 && v < r.key.bucket_lo) return false;
        if (r.key.bucket_hi != 0 && v > r.key.bucket_hi) return false;
    }
    return true;
}

// Built-in qualified defaults, from measurements on this project's B70 (spec section 1A step 3).  No entry
// exists for dg2: an A770 configuration that has not been measured keeps the established safe path.
const TuningTable& builtin_tuning() {
    static const TuningTable t = [] {
        TuningTable r;
        TuningRecord topk;
        topk.operation = "topk";
        topk.kernel = "hierarchical";
        topk.key.arch = DeviceArch::Bmg;      // B70 / Battlemage only
        topk.key.arch_set = true;
        topk.key.max_queries = 32;            // measured crossover: wins for few queries, loses >= 64
        topk.key.bucket_lo = kLegacyRegisterCap + 1;  // only beyond the register selector's capacity
        topk.runs = 12;
        topk.median = 0.078;                  // 262144 ctx, 1 query (ms) vs reference 0.580
        topk.range_lo = 0.078;
        topk.range_hi = 0.580;
        r.push_back(topk);
        TuningRecord sc;
        sc.operation = "scores";
        sc.kernel = "tiled";
        sc.key.arch = DeviceArch::Bmg;  // B70 only; dg2 keeps the legacy scorer until measured
        sc.key.arch_set = true;
        sc.runs = 8;
        sc.median = 15.56;   // 262144 ctx, 256 queries (ms); legacy 19.04
        sc.range_lo = 15.56;
        sc.range_hi = 19.04;
        r.push_back(sc);
        return r;
    }();
    return t;
}

const TuningRecord* find_tuning_record(const TuningTable& t, const char* operation, const DeviceProfile& d,
                                       const OpShape& s) {
    for (const TuningRecord& r : t) {
        if (r.operation != operation) continue;
        if (!tuning_record_matches(r, d, s)) continue;
        return &r;
    }
    return nullptr;
}

namespace {

// A record refines, never relaxes: its kernel is accepted only if that kernel is usable on this device.
bool kernel_usable(const DeviceProfile& d, const std::string& k, bool& precision_ok) {
    precision_ok = true;
    if (k == "hierarchical") return d.topk_hier.usable();
    if (k == "tiled") return d.scores_tiled.usable() || d.prompt_attn_tiled.usable();
    if (k == "device") return d.grouping_device.usable();
    if (k == "bounded") return d.experts_bounded.usable();
    if (k == "legacy" || k == "host" || k == "reference") return true;
    if (k == "xmx") {
        precision_ok = false;  // caller decides after consulting format
        return false;
    }
    return false;
}

}  // namespace

// ---- selectors ---------------------------------------------------------------------------------------

static Choice finish(Choice c) { return c; }

Choice select_topk(const SelectorInput& in) {
    Choice c;
    if (in.device == nullptr) {
        c.ok = false;
        c.reason = "no device profile";
        return c;
    }
    const DeviceProfile& d = *in.device;
    const OpShape& s = in.shape;
    if (s.queries <= 0) {
        c.kernel = "none";
        c.reason = "no queries";
        return c;
    }
    if (s.max_blocks <= 0) {
        c.ok = false;
        c.reason = "invalid max_blocks";
        return c;
    }
    const int64_t reach =
        (s.active_blocks > 0 && s.active_blocks <= s.max_blocks) ? s.active_blocks : s.max_blocks;
    const bool legacy_fits = reach <= kLegacyRegisterCap;
    const uint64_t scratch = topk_hier_scratch_bytes(s.queries, s.max_blocks, kTopkTileDefault);
    const bool scratch_ok = scratch != UINT64_MAX &&
                            (in.scratch_budget == UINT64_MAX || scratch <= in.scratch_budget);
    const bool counts_ok = topk_hier_counts_are_32bit_safe(s.queries, s.cells > 0 ? s.cells : s.max_blocks * 4);
    const bool hier_ok = d.topk_hier.usable() && scratch_ok && counts_ok;

    const std::string ov = normalize_override(in.override_value);
    if (ov == "reference") {
        c.kernel = "reference";
        c.reason = "explicit override";
        return c;
    }
    if (ov == "legacy") {
        if (!legacy_fits) {
            c.ok = false;
            c.rejected_override = true;
            c.reason = "override 'legacy' cannot cover max_blocks above the register capacity";
            return c;
        }
        c.kernel = "legacy";
        c.reason = "explicit override";
        return c;
    }
    if (ov == "hierarchical") {
        if (!d.topk_hier.usable()) {
            c.ok = false;
            c.rejected_override = true;
            c.reason = "override 'hierarchical' rejected: no compiled qualified image on this device";
            return c;
        }
        if (!scratch_ok) {
            c.ok = false;
            c.rejected_override = true;
            c.reason = "override 'hierarchical' rejected: workspace budget too small";
            return c;
        }
        if (!counts_ok) {
            c.ok = false;
            c.rejected_override = true;
            c.reason = "override 'hierarchical' rejected: 32-bit histogram counts unsafe for this context";
            return c;
        }
        c.kernel = "hierarchical";
        c.scratch_bytes = scratch;
        c.tile = (int)kTopkTileDefault;
        c.reason = "explicit override";
        return c;
    }
    if (ov != "auto") {
        c.ok = false;
        c.rejected_override = true;
        c.reason = "unknown STRATA_SYCL_TOPK value '" + ov + "'";
        return c;
    }

    // automatic
    const TuningRecord* rec = in.tuning != nullptr ? find_tuning_record(*in.tuning, "topk", d, s) : nullptr;
    if (rec == nullptr)
        rec = find_tuning_record(builtin_tuning(), "topk", d, s);  // built-in qualified default
    if (rec != nullptr) {
        bool prec = true;
        if (kernel_usable(d, rec->kernel, prec) && rec->kernel != "reference") {
            if (rec->kernel == "hierarchical") {
                if (hier_ok) {
                    c.kernel = "hierarchical";
                    c.scratch_bytes = scratch;
                    c.tile = (int)kTopkTileDefault;
                    c.reason = "qualified record for this device/shape";
                    return c;
                }
            } else if (rec->kernel == "legacy" && legacy_fits) {
                c.kernel = "legacy";
                c.reason = "qualified record for this device/shape";
                return c;
            }
        }
    }
    if (legacy_fits) {
        c.kernel = "legacy";
        c.reason = "established default (no qualified long-context winner)";
        return c;
    }
    // Long context beyond the register capacity: the established path is the reference selector until the
    // hierarchical path is qualified for this exact configuration.
    c.kernel = "reference";
    c.fallback = true;
    c.reason = hier_ok ? "no qualified hierarchical record for this device/shape"
                       : "hierarchical unavailable or workspace too small";
    return c;
}

Choice select_scores(const SelectorInput& in) {
    Choice c;
    if (in.device == nullptr) {
        c.ok = false;
        c.reason = "no device profile";
        return c;
    }
    const DeviceProfile& d = *in.device;
    const std::string ov = normalize_override(in.override_value);
    auto xmx_allowed = [&] {
        const bool fmt_ok = (in.shape.format & (kFmtFp16Input | kFmtBf16Input)) != 0;
        return fmt_ok && d.matrix.usable_for_fp16() && d.scores_xmx.usable();
    };
    if (ov == "legacy") {
        c.kernel = "legacy";
        c.reason = "explicit override";
        return c;
    }
    if (ov == "tiled") {
        if (!d.scores_tiled.usable()) {
            c.ok = false;
            c.rejected_override = true;
            c.reason = "override 'tiled' rejected: no compiled qualified image on this device";
            return c;
        }
        c.kernel = "tiled";
        c.reason = "explicit override";
        return c;
    }
    if (ov == "xmx") {
        if (!xmx_allowed()) {
            c.ok = false;
            c.rejected_override = true;
            c.reason = "override 'xmx' rejected: matrix capability/format not qualified on this device";
            return c;
        }
        c.kernel = "xmx";
        c.reason = "explicit override";
        return c;
    }
    if (ov != "auto") {
        c.ok = false;
        c.rejected_override = true;
        c.reason = "unknown STRATA_SYCL_SCORES value '" + ov + "'";
        return c;
    }
    const TuningRecord* rec = in.tuning != nullptr ? find_tuning_record(*in.tuning, "scores", d, in.shape) : nullptr;
    if (rec == nullptr)
        rec = find_tuning_record(builtin_tuning(), "scores", d, in.shape);
    if (rec != nullptr) {
        if (rec->kernel == "tiled" && d.scores_tiled.usable()) {
            c.kernel = "tiled";
            c.reason = "qualified record for this device/shape";
            return c;
        }
        if (rec->kernel == "xmx" && xmx_allowed()) {
            c.kernel = "xmx";
            c.reason = "qualified record for this device/shape";
            return c;
        }
    }
    c.kernel = "legacy";
    c.reason = "established default (exact FP32 scorer; no qualified winner)";
    return c;
}

Choice select_prompt_attn(const SelectorInput& in) {
    Choice c;
    if (in.device == nullptr) {
        c.ok = false;
        c.reason = "no device profile";
        return c;
    }
    const DeviceProfile& d = *in.device;
    const std::string ov = normalize_override(in.override_value);
    auto xmx_allowed = [&] {
        const bool fmt_ok = (in.shape.format & (kFmtFp16Input | kFmtBf16Input)) != 0;
        return fmt_ok && d.matrix.usable_for_fp16() && d.prompt_attn_xmx.usable();
    };
    if (ov == "legacy") {
        c.kernel = "legacy";
        c.reason = "explicit override";
        return c;
    }
    if (ov == "tiled") {
        if (!d.prompt_attn_tiled.usable()) {
            c.ok = false;
            c.rejected_override = true;
            c.reason = "override 'tiled' rejected: no compiled qualified image on this device";
            return c;
        }
        c.kernel = "tiled";
        c.reason = "explicit override";
        return c;
    }
    if (ov == "xmx") {
        if (!xmx_allowed()) {
            c.ok = false;
            c.rejected_override = true;
            c.reason = "override 'xmx' rejected: matrix capability/format not qualified on this device";
            return c;
        }
        c.kernel = "xmx";
        c.reason = "explicit override";
        return c;
    }
    if (ov != "auto") {
        c.ok = false;
        c.rejected_override = true;
        c.reason = "unknown STRATA_SYCL_PROMPT_ATTN value '" + ov + "'";
        return c;
    }
    if (in.tuning != nullptr) {
        const TuningRecord* r = find_tuning_record(*in.tuning, "prompt_attn", d, in.shape);
        if (r != nullptr) {
            if (r->kernel == "tiled" && d.prompt_attn_tiled.usable()) {
                c.kernel = "tiled";
                c.reason = "qualified record for this device/shape";
                return c;
            }
            if (r->kernel == "xmx" && xmx_allowed()) {
                c.kernel = "xmx";
                c.reason = "qualified record for this device/shape";
                return c;
            }
        }
    }
    c.kernel = "legacy";
    c.reason = "established default (no qualified winner)";
    return c;
}

Choice select_grouping(const SelectorInput& in) {
    Choice c;
    if (in.device == nullptr) {
        c.ok = false;
        c.reason = "no device profile";
        return c;
    }
    const DeviceProfile& d = *in.device;
    const std::string ov = normalize_override(in.override_value);
    if (ov == "host") {
        c.kernel = "host";
        c.reason = "explicit override";
        return c;
    }
    if (ov == "device") {
        if (!d.grouping_device.usable()) {
            c.ok = false;
            c.rejected_override = true;
            c.reason = "override 'device' rejected: no compiled qualified image on this device";
            return c;
        }
        c.kernel = "device";
        c.reason = "explicit override";
        return c;
    }
    if (ov != "auto") {
        c.ok = false;
        c.rejected_override = true;
        c.reason = "unknown STRATA_SYCL_GROUPING value '" + ov + "'";
        return c;
    }
    if (in.tuning != nullptr) {
        const TuningRecord* r = find_tuning_record(*in.tuning, "grouping", d, in.shape);
        if (r != nullptr && (r->kernel == "device" || r->kernel == "host")) {
            if (r->kernel == "device" && !d.grouping_device.usable()) {
                // fall through to host
            } else {
                c.kernel = r->kernel;
                c.reason = "qualified record for this device/shape";
                return c;
            }
        }
    }
    c.kernel = "host";
    c.reason = "established default (host grouping; device path not qualified)";
    return c;
}

Choice select_experts(const SelectorInput& in) {
    Choice c;
    if (in.device == nullptr) {
        c.ok = false;
        c.reason = "no device profile";
        return c;
    }
    const DeviceProfile& d = *in.device;
    const std::string ov = normalize_override(in.override_value);
    if (ov == "legacy") {
        c.kernel = "legacy";
        c.reason = "explicit override";
        return c;
    }
    if (ov == "bounded") {
        if (!d.experts_bounded.usable()) {
            c.ok = false;
            c.rejected_override = true;
            c.reason = "override 'bounded' rejected: no compiled qualified image on this device";
            return c;
        }
        c.kernel = "bounded";
        c.reason = "explicit override";
        return c;
    }
    if (ov == "xmx") {
        const bool fmt_ok = (in.shape.format & (kFmtFp16Input | kFmtBf16Input)) != 0;
        if (!(fmt_ok && d.matrix.usable_for_fp16() && d.experts_xmx.usable())) {
            c.ok = false;
            c.rejected_override = true;
            c.reason = "override 'xmx' rejected: matrix capability/format not qualified on this device";
            return c;
        }
        c.kernel = "xmx";
        c.reason = "explicit override";
        return c;
    }
    if (ov != "auto") {
        c.ok = false;
        c.rejected_override = true;
        c.reason = "unknown STRATA_SYCL_EXPERTS value '" + ov + "'";
        return c;
    }
    if (in.tuning != nullptr) {
        const TuningRecord* r = find_tuning_record(*in.tuning, "experts", d, in.shape);
        if (r != nullptr && r->kernel == "bounded" && d.experts_bounded.usable()) {
            c.kernel = "bounded";
            c.reason = "qualified record for this device/shape";
            return c;
        }
    }
    c.kernel = "legacy";
    c.reason = "established default (no qualified winner)";
    return c;
}

Choice select_chunk_cap(const SelectorInput& in) {
    Choice c;
    c.kernel = "planner";
    c.reason = "no qualified chunk recommendation";
    if (in.device == nullptr || in.tuning == nullptr) return c;
    const TuningRecord* r = find_tuning_record(*in.tuning, "chunk", *in.device, in.shape);
    if (r == nullptr || r->value <= 0) return c;
    c.value = r->value;
    c.kernel = "chunk";
    c.reason = "qualified chunk recommendation for this device/shape";
    return c;
}

// ---- logging -----------------------------------------------------------------------------------------

namespace {
std::mutex g_log_mu;
std::set<std::string> g_logged;
}  // namespace

void reset_dispatch_log() {
    std::lock_guard<std::mutex> lock(g_log_mu);
    g_logged.clear();
}

void log_choice_once(const DeviceProfile& d, const char* operation, const Choice& c) {
    static const bool quiet = [] {
        const char* v = std::getenv("STRATA_SYCL_DISPATCH_QUIET");
        return v != nullptr && std::atoi(v) != 0;
    }();
    if (quiet) return;
    std::string key = std::string(operation) + "|" + arch_name(d.arch) + "|" + c.kernel + "|" +
                      (c.fallback ? "fallback" : "direct");
    {
        std::lock_guard<std::mutex> lock(g_log_mu);
        if (!g_logged.insert(key).second) return;
    }
    std::fprintf(stderr, "strata sycl dispatch: op=%s arch=%s dev=%s id=0x%04x -> %s (%s)%s\n", operation,
                 arch_name(d.arch), d.name.c_str(), d.device_id, c.kernel.c_str(), c.reason.c_str(),
                 c.fallback ? " [fallback]" : "");
}

}  // namespace strata::sycl_runtime
