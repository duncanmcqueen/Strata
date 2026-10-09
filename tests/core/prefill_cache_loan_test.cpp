#include "strata/prefill/cache_loan.hpp"
#include <cstdio>
#include <cstdlib>
#include <vector>
static void check(bool ok) { if (!ok) std::abort(); }
int main() {
    using strata::prefill::cache_loan;
    std::vector<uint64_t> offsets(401);
    for (int i = 0; i < 400; ++i) offsets[i + 1] = offsets[i] + (i < 200 ? 1 : 2);
    auto head = cache_loan(400, offsets.data(), 2, 120, true);
    auto tail = cache_loan(400, offsets.data(), 2, 120, false);
    check(head.first == 0 && head.end == 120 && head.bytes == 120);
    check(tail.first == 340 && tail.end == 400 && tail.bytes == 120);
    // Repeated server loans: shrinking/growing must mark and restore only the actual range.
    for (bool at_head : {false, true}) {
        std::vector<int> resident(400);
        for (int i = 0; i < 400; ++i) resident[i] = i;
        for (uint64_t need : {120, 7, 211, 0, 119}) {
            const auto loan = cache_loan(400, offsets.data(), 2, need, at_head);
            check(loan.bytes >= need && loan.count() <= 400);
            check(!loan.contains(-1) && !loan.contains(400));
            auto before = resident;
            for (int& slot : resident) if (loan.contains(slot)) slot = -1;
            for (int i = 0; i < 400; ++i) {
                check((resident[i] == -1) == loan.contains(i));
                if (loan.contains(i)) resident[i] = before[i];
            }
            check(resident == before);
        }
    }
    check(cache_loan(400, offsets.data(), 2, 601, true).count() > 400);
    check(cache_loan(400, offsets.data(), 2, 601, false).count() > 400);
    check(cache_loan(10, nullptr, 8, 17, true).end == 3);
    check(cache_loan(10, nullptr, 8, 17, false).first == 7);
    check(cache_loan(0, nullptr, 8, 1, true).count() == 1);
    std::puts("prefill cache loan: nonuniform sizing, bounds and repeated restoration passed");
}
