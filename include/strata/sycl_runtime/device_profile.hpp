// include/strata/sycl_runtime/device_profile.hpp - architecture-aware dispatch (spec section 1A).
//
// One DeviceProfile describes an actual runtime device (the ones the runtime registry enumerated, B70 / A770 /
// B580 / B60 / unknown Intel GPU).  Profiles are immutable data: they may be shared, but their construction is
// synchronized.  The selection functions below are PURE: they take a profile, an operation shape, a workspace
// budget and an optional user override, and return the chosen kernel / tile / scratch need plus a diagnostic
// reason.  Host runtime queries live in device_profile.cpp, never in the selectors, so the table in section 1A
// can be unit-tested with synthetic profiles and no GPU.
//
// Nothing here keys on a marketing name.  Architecture comes from supported device-ID / capability queries; a
// name is carried for diagnostics only.  Unknown or unqualified configurations always fall through to the
// established safe path, and a B70 result is never applied to A770 merely because both are SYCL.
#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace strata::sycl_runtime {

/// Architecture family from supported queries (not a device-name substring).
enum class DeviceArch { Unknown, Dg2, Bmg };

const char* arch_name(DeviceArch a);

/// Classify a PCI device ID.  Pure and total: an unrecognized ID yields false and leaves `out` untouched.
/// Vendor 0x8086 is Intel.  DG2 = Alchemist (A380/A750/A770/A-series); BMG = Battlemage (B570/B580/Pro B-series).
bool device_id_to_arch(uint32_t vendor_id, uint32_t device_id, DeviceArch& out);

/// Per-architecture matrix (XMX) tile capabilities, represented independently (never one `has_xmx` boolean).
struct MatrixTileCaps {
    bool fp16_16x16x16 = false;  ///< joint_matrix FP16 x FP16 -> FP32 accumulate
    bool bf16_16x16x16 = false;
    bool int8_8x16x32 = false;
    bool int8_16x16x16 = false;
    bool capabilities_probed = false;  ///< a capability query was actually run on this device
    bool correctness_checked = false;  ///< an in-application correctness check was run
    bool correctness_passed = false;   ///< ... and every enabled probe passed
    bool any_tile() const {
        return fp16_16x16x16 || bf16_16x16x16 || int8_8x16x32 || int8_16x16x16;
    }
    /// A matrix path may be offered only when a tile exists AND (if checked) correctness passed.
    bool usable_for_fp16() const {
        return fp16_16x16x16 && (!correctness_checked || correctness_passed);
    }
};

/// A compiled kernel family's availability and its gate.  `compiled` means an AOT/JIT image exists for this
/// device; the capability flags are optional independent checks.  A failed capability check excludes the family
/// even though it is compiled.
struct KernelFamily {
    bool compiled = false;
    bool capability_checked = false;
    bool capability_passed = false;
    bool usable() const { return compiled && (!capability_checked || capability_passed); }
};

/// Immutable per-device data.  Built once, from the runtime's existing device registry.
struct DeviceProfile {
    int ordinal = -1;
    std::string name;  ///< diagnostics only
    uint32_t vendor_id = 0;
    uint32_t device_id = 0;
    bool device_id_available = false;
    DeviceArch arch = DeviceArch::Unknown;
    std::string driver_version;
    std::string compiler_id;  ///< build/compiler identity recorded in the binary

    std::vector<int> subgroup_sizes;
    uint32_t max_workgroup_size = 0;
    uint64_t local_mem_bytes = 0;
    uint32_t eu_count = 0;
    uint32_t slices = 0;
    uint64_t global_mem_bytes = 0;

    MatrixTileCaps matrix;

    // available compiled kernel families (W1/W2/W3/W4/W5)
    KernelFamily topk_hier;        ///< W1 hierarchical exact top-k
    KernelFamily scores_tiled;     ///< W2 shared-key tiled FP32 scoring
    KernelFamily prompt_attn_tiled;///< W3 tiled sparse prompt attention
    KernelFamily grouping_device;  ///< W4 device expert grouping
    KernelFamily experts_bounded;  ///< W5a bounded expert intermediates
    // precision-changing matrix candidates, gated off until qualified
    KernelFamily scores_xmx;        ///< W2 optional XMX scoring
    KernelFamily prompt_attn_xmx;   ///< W3 optional XMX prompt attention
    KernelFamily experts_xmx;       ///< W5c optional tile-local unpack + matrix product

    bool has_subgroup(int width) const {
        for (int s : subgroup_sizes)
            if (s == width) return true;
        return false;
    }
};

/// Query every runtime device once (uses the existing registry + context; no parallel discovery).  Safe to call
/// repeatedly; construction is synchronized and memoized.  Call OUTSIDE graph capture.  Missing extension
/// queries fail safely (the profile keeps Unknown / zero values).
void warm_device_profiles();
/// The memoized profile of a device ordinal; builds it on demand if warm_device_profiles() has not run yet.
const DeviceProfile& profile_for_device(int ordinal);
/// The profile of the device bound to a stream (nullptr = the current device).  Dispatch follows the stream's
/// device, never a process-global "first GPU".
const DeviceProfile& profile_for_stream(void* stream);
/// How many devices the runtime enumerated.
int profile_device_count();

// ---- operation shapes ---------------------------------------------------------------------------------

/// Bit flags for the format/precision a candidate would change, so a selector can refuse a precision change
/// unless it is explicitly qualified.  Kept coarse on purpose.
enum : uint32_t {
    kFmtFp32Exact = 0,      ///< no precision change
    kFmtFp16Input = 1u << 0,///< matrix inputs rounded to FP16
    kFmtBf16Input = 1u << 1,
    kFmtInt8Input = 1u << 2,
};

struct OpShape {
    int64_t queries = 1;
    int64_t max_blocks = 0;    ///< physical score-row stride (top-k)
    int64_t active_blocks = -1;///< live bound; -1 = unknown/captured
    int64_t width = 0;         ///< top-k selection width (cells)
    int64_t cells = 0;         ///< n_kv (for reporting/bucketing)
    int64_t experts = 0;
    int64_t rows = 0;          ///< grouped rows (experts) / selected cells (attention)
    int columns = 1;           ///< activation columns (MMVQ)
    uint32_t format = kFmtFp32Exact;
};

/// The result of a selection: a kernel ID, a tile/launch configuration, the scratch it needs and a reason.
struct Choice {
    std::string kernel = "none";
    std::string reason;
    bool ok = true;             ///< false only for a rejected explicit override or unsupported geometry
    bool rejected_override = false;
    bool fallback = false;      ///< a safe established path was chosen because nothing was qualified
    uint64_t scratch_bytes = 0;
    int64_t value = 0;          ///< recommended parameter (e.g. chunk cap) where relevant
    int tile = 0;               ///< primary tile parameter (blocks/cells/rows)
    int workgroup = 0;
    int subgroup = 0;
};

// ---- qualified tuning records ------------------------------------------------------------------------

/// A record's identity key.  A record matches only when every identity field it supplies is equal; a change to
/// driver, compiler, model or device invalidates it (spec section 1A / W6).
struct TuningKey {
    DeviceArch arch = DeviceArch::Unknown;
    bool arch_set = false;
    uint32_t device_id = 0;
    bool device_id_set = false;
    std::string compiler;
    std::string driver;
    std::string model;
    std::string kv_mode;
    int64_t max_queries = 0;   ///< 0 = any; upper bound on queries for this operation
    int64_t max_context = 0;   ///< 0 = any
    int64_t bucket_lo = 0;     ///< shape bucket on `max_blocks`/rows; 0 = any
    int64_t bucket_hi = 0;     ///< inclusive; 0 = any
};

struct TuningRecord {
    int schema_version = 1;
    TuningKey key;
    std::string operation;   ///< "topk" / "scores" / "prompt_attn" / "grouping" / "experts" / "chunk"
    std::string kernel;      ///< recommended kernel ID
    int64_t value = 0;       ///< recommended cap/parameter (e.g. chunk size), or 0
    int runs = 0;
    double median = 0;
    double range_lo = 0;
    double range_hi = 0;
};

using TuningTable = std::vector<TuningRecord>;

/// Does a record's key apply to this profile and shape?  Pure.
bool tuning_record_matches(const TuningRecord& r, const DeviceProfile& d, const OpShape& s);
/// The first matching record for `operation`, or nullptr.  Pure.
const TuningRecord* find_tuning_record(const TuningTable& t, const char* operation, const DeviceProfile& d,
                                       const OpShape& s);

/// W1 hierarchical top-k workspace bytes for a call: [queries, ceil(max_blocks / tile_blocks), 256] 32-bit tile
/// histograms plus bounded tile/query state.  Checked 64-bit arithmetic; returns UINT64_MAX on overflow.
/// The same formula sizes the kernel's workspace view, so caller and selector cannot disagree.
uint64_t topk_hier_scratch_bytes(int64_t queries, int64_t max_blocks, int64_t tile_blocks = 1024);
/// Number of tile rows the hierarchical workspace uses.
int64_t topk_hier_tile_count(int64_t max_blocks, int64_t tile_blocks = 1024);
/// Whether 32-bit histogram counts are safe for this configuration (cells must fit int32 with headroom).
bool topk_hier_counts_are_32bit_safe(int64_t queries, int64_t cells);

// ---- pure selectors ----------------------------------------------------------------------------------

/// Shared input to every selector.
struct SelectorInput {
    const DeviceProfile* device = nullptr;
    OpShape shape;
    uint64_t scratch_budget = UINT64_MAX;
    std::string override_value;      ///< "" or "auto" = automatic; anything else is an explicit force
    const TuningTable* tuning = nullptr;  ///< optional W6 records that may refine (never relax) a choice
};

/// W1: exact top-k.  Values: reference|legacy|hierarchical|auto.
Choice select_topk(const SelectorInput& in);
/// W2: blocked scoring.  Values: legacy|tiled|xmx|auto.
Choice select_scores(const SelectorInput& in);
/// W3: sparse prompt attention.  Values: legacy|tiled|xmx|auto.
Choice select_prompt_attn(const SelectorInput& in);
/// W4: expert route grouping.  Values: host|device|auto.
Choice select_grouping(const SelectorInput& in);
/// W5: expert intermediates.  Values: legacy|bounded|xmx|auto.
Choice select_experts(const SelectorInput& in);
/// W6: recommended prefill chunk cap (returns value = cap; 0 = no recommendation).  Never relaxes user limits.
Choice select_chunk_cap(const SelectorInput& in);

/// Log the chosen implementation and fallback reason once per configuration (deduplicated, mutex-guarded),
/// unless STRATA_SYCL_DISPATCH_QUIET=1.  Safe to call from hot host code (the check is a cheap set lookup
/// after the first call).  Never called from a device/graph context.
void log_choice_once(const DeviceProfile& d, const char* operation, const Choice& c);

/// Reset the once-log dedup (tests).
void reset_dispatch_log();

}  // namespace strata::sycl_runtime
