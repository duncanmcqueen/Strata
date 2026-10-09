#pragma once
#include <cstdint>

namespace strata::prefill {
// Half-open slot range. Sized-cache offsets must contain slots + 1 entries.
struct CacheLoan {
    int64_t first = 0, end = 0;
    uint64_t bytes = 0;
    int64_t count() const { return end - first; }
    bool contains(int64_t slot) const { return slot >= first && slot < end; }
};
inline CacheLoan cache_loan(int64_t slots, const uint64_t* offsets, uint64_t uniform_bytes,
                            uint64_t need, bool head) {
    auto bytes_for = [&](int64_t k) {
        if (!offsets) return uint64_t(k) * uniform_bytes;
        return head ? offsets[k] - offsets[0] : offsets[slots] - offsets[slots - k];
    };
    int64_t k = 0;
    while (k < slots && bytes_for(k) < need) ++k;
    // A count greater than slots signals that even the entire cache cannot fit.
    if (bytes_for(k) < need) return {0, slots + 1, bytes_for(k)};
    return {head ? 0 : slots - k, head ? k : slots, bytes_for(k)};
}
} // namespace strata::prefill
