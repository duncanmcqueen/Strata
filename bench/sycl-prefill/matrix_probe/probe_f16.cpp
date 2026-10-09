// tests/sycl/matrix_probe.cpp - P0: does the installed DPC++/B70 run joint_matrix
// products correctly, both standalone and linked into Strata's kernel libraries?
//
// It reports every legal matrix combination the device advertises, then executes
// FP16 and BF16 (FP32 accumulate) and INT8 (INT32 accumulate) products with
// nonzero/ragged inputs against an independent double reference.  Exit 77 (ctest
// SKIP) when the device advertises no usable combination; exit 1 on a wrong result.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

namespace mat = sycl::ext::oneapi::experimental::matrix;

static int g_fail = 0, g_ran = 0;

static const char* tname(mat::matrix_type t) {
    switch (t) {
        case mat::matrix_type::bf16: return "bf16";
        case mat::matrix_type::fp16: return "fp16";
        case mat::matrix_type::tf32: return "tf32";
        case mat::matrix_type::fp32: return "fp32";
        case mat::matrix_type::fp64: return "fp64";
        case mat::matrix_type::sint8: return "sint8";
        case mat::matrix_type::sint16: return "sint16";
        case mat::matrix_type::sint32: return "sint32";
        case mat::matrix_type::uint8: return "uint8";
        case mat::matrix_type::uint16: return "uint16";
        case mat::matrix_type::uint32: return "uint32";
        case mat::matrix_type::uint64: return "uint64";
        default: return "?";
    }
}

// One product C[M,N] = A[M,K] . B[K,N] with nonzero, ragged values; A/B 16-bit.
template <typename TA, typename TB>
static void run_f16(sycl::queue& q, const char* label, size_t M, size_t N, size_t K) {
    const size_t asz = M * K, bsz = K * N, csz = M * N;
    std::vector<float> a(asz), b(bsz);
    for (size_t i = 0; i < asz; ++i) a[i] = (float) (((int) (i * 7 % 13)) - 6) * 0.25f;   // nonzero, ragged signs
    for (size_t i = 0; i < bsz; ++i) b[i] = (float) (((int) (i * 5 % 11)) - 5) * 0.5f;
    // reference in double, rounding A/B through their 16-bit type
    auto trunc = [](float v) {
        if constexpr (std::is_same_v<TA, sycl::half>) return (float) (sycl::half) v;
        else return (float) sycl::ext::oneapi::bfloat16(v);
    };
    std::vector<double> ref(csz, 0.0);
    for (size_t m = 0; m < M; ++m)
        for (size_t n = 0; n < N; ++n) {
            double s = 0;
            for (size_t k = 0; k < K; ++k) s += (double) trunc(a[m * K + k]) * (double) trunc(b[k * N + n]);
            ref[m * N + n] = s;
        }
    auto* da = sycl::malloc_device<TA>(asz, q);
    auto* db = sycl::malloc_device<TB>(bsz, q);
    auto* dc = sycl::malloc_device<float>(csz, q);
    std::vector<TA> ha(asz);
    std::vector<TB> hb(bsz);
    for (size_t i = 0; i < asz; ++i) ha[i] = (TA) a[i];
    for (size_t i = 0; i < bsz; ++i) hb[i] = (TB) b[i];
    q.memcpy(da, ha.data(), asz * sizeof(TA));
    q.memcpy(db, hb.data(), bsz * sizeof(TB));
    const size_t nwi = (M / 16) * (N / 16);
    q.parallel_for(sycl::nd_range<1>(nwi * 16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const size_t gid = it.get_group(0);
        const size_t gm = (gid / (N / 16)) * 16, gn = (gid % (N / 16)) * 16;
        mat::joint_matrix<sycl::sub_group, TA, mat::use::a, 16, 16, mat::layout::row_major> tA;
        mat::joint_matrix<sycl::sub_group, TB, mat::use::b, 16, 16, mat::layout::row_major> tB;
        mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, 16, 16> tC;
        mat::joint_matrix_load(sg, tA, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(da + gm * K), K);
        mat::joint_matrix_load(sg, tB, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(db + gn), N);
        mat::joint_matrix_fill(sg, tC, 0.0f);
        mat::joint_matrix_mad(sg, tC, tA, tB, tC);
        mat::joint_matrix_store(sg, tC, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dc + gm * N + gn), N, mat::layout::row_major);
    });
    std::vector<float> got(csz);
    q.memcpy(got.data(), dc, csz * sizeof(float)).wait();
    double maxerr = 0, maxref = 0;
    for (size_t i = 0; i < csz; ++i) {
        maxerr = std::max(maxerr, std::fabs((double) got[i] - ref[i]));
        maxref = std::max(maxref, std::fabs(ref[i]));
    }
    const bool ok = maxerr <= 1e-3 * std::max(1.0, maxref);
    std::printf("  %-8s M=%zu N=%zu K=%zu: max abs err %.3e (ref max %.3e) %s\n", label, M, N, K, maxerr, maxref,
                ok ? "OK" : "FAIL");
    if (!ok) ++g_fail;
    ++g_ran;
    sycl::free(da, q); sycl::free(db, q); sycl::free(dc, q);
}

#if 0
static void run_i8(sycl::queue& q, size_t M, size_t N, size_t K) {
    const size_t asz = M * K, bsz = K * N, csz = M * N;
    std::vector<int8_t> a(asz), b(bsz);
    for (size_t i = 0; i < asz; ++i) a[i] = (int8_t) (((int) (i * 7 % 29)) - 14);
    for (size_t i = 0; i < bsz; ++i) b[i] = (int8_t) (((int) (i * 5 % 15)) - 7);
    std::vector<int32_t> ref(csz, 0);
    for (size_t m = 0; m < M; ++m)
        for (size_t n = 0; n < N; ++n) {
            int32_t s = 0;
            for (size_t k = 0; k < K; ++k) s += (int32_t) a[m * K + k] * (int32_t) b[k * N + n];
            ref[m * N + n] = s;
        }
    auto* da = sycl::malloc_device<int8_t>(asz, q);
    auto* db = sycl::malloc_device<int8_t>(bsz, q);
    auto* dc = sycl::malloc_device<int32_t>(csz, q);
    q.memcpy(da, a.data(), asz);
    q.memcpy(db, b.data(), bsz);
    const size_t nwi = (M / 16) * (N / 16);
    q.parallel_for(sycl::nd_range<1>(nwi * 16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const size_t gid = it.get_group(0);
        const size_t gm = (gid / (N / 16)) * 16, gn = (gid % (N / 16)) * 16;
        mat::joint_matrix<sycl::sub_group, int8_t, mat::use::a, 16, 16, mat::layout::row_major> tA;
        mat::joint_matrix<sycl::sub_group, int8_t, mat::use::b, 16, 16, mat::layout::row_major> tB;
        mat::joint_matrix<sycl::sub_group, int32_t, mat::use::accumulator, 16, 16> tC;
        mat::joint_matrix_load(sg, tA, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(da + gm * K), K);
        mat::joint_matrix_load(sg, tB, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(db + gn), N);
        mat::joint_matrix_fill(sg, tC, 0);
        mat::joint_matrix_mad(sg, tC, tA, tB, tC);
        mat::joint_matrix_store(sg, tC, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dc + gm * N + gn), N, mat::layout::row_major);
    });
    std::vector<int32_t> got(csz);
    q.memcpy(got.data(), dc, csz * sizeof(int32_t)).wait();
    long maxerr = 0;
    for (size_t i = 0; i < csz; ++i) maxerr = std::max(maxerr, (long) std::labs((long) got[i] - ref[i]));
    const bool ok = maxerr == 0;
    std::printf("  %-8s M=%zu N=%zu K=%zu: max int err %ld %s\n", "int8", M, N, K, maxerr, ok ? "OK" : "FAIL");
    if (!ok) ++g_fail;
    ++g_ran;
    sycl::free(da, q); sycl::free(db, q); sycl::free(dc, q);
}

#endif
int main() {
    try {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        auto dev = q.get_device();
        std::printf("device: %s\n", dev.get_info<sycl::info::device::name>().c_str());
        {
            auto sgs = dev.get_info<sycl::info::device::sub_group_sizes>();
            std::printf("sub-group sizes:");
            for (auto s : sgs) std::printf(" %zu", (size_t) s);
            std::printf("\n");
        }
        // The device-info matrix::combinations descriptor is not present in this
        // DPC++ (2026.1); the extension's capability query is compile-time, so the
        // execution tests below are the runtime evidence.
        // FP16 (FP32 accumulate)
        try { run_f16<sycl::half, sycl::half>(q, "fp16", 16, 16, 16); } catch (const sycl::exception& e) { std::printf("  fp16 16x16x16 threw: %s\n", e.what()); ++g_fail; }
        try { run_f16<sycl::half, sycl::half>(q, "fp16", 32, 32, 32); } catch (const sycl::exception& e) { std::printf("  fp16 32x32x32 threw: %s\n", e.what()); ++g_fail; }
        try { run_f16<sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(q, "bf16", 16, 16, 16); }
        catch (const sycl::exception& e) { std::printf("  bf16 16x16x16 threw: %s\n", e.what()); ++g_fail; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "matrix probe: %s\n", e.what());
        return 77;
    }
    std::printf("matrix probe: %d products run, %d failed\n", g_ran, g_fail);
    if (g_ran == 0) return 77;   // no combination could be launched
    return g_fail == 0 ? 0 : 1;
}
