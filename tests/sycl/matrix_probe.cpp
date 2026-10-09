// tests/sycl/matrix_probe.cpp - P0: does the installed DPC++/B70 run joint_matrix
// products correctly, both standalone and linked into Strata's kernel libraries?
//
// Measured on this machine (Arc Pro B70 / bmg-g31, DPC++ 2026.0): the AOT backend
// accepts FP16 and BF16 joint_matrix at 16x16x16 (and rejects INT8 there: "unsupported
// number of rows/columns", plus undefined OpJointMatrix...INTEL builtins).  The only
// legal INT8 tile is 8x16x32 (A 8x32, B 32x16, C 8x16).  This test therefore exercises
//   FP16   16x16x16 tiles, K accumulated in a loop
//   BF16   16x16x16 tiles
//   INT8    8x16x32 tiles
// with nonzero values, multiple work-items, ragged (zero-padded) tails, and a double /
// integer independent reference over the real dimensions.  Exit 77 when nothing runs.
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <limits>
#include <vector>

namespace mat = sycl::ext::oneapi::experimental::matrix;

static int g_fail = 0, g_ran = 0;

// Reject nonfinite values before std::max: max(finite, NaN) can hide a corrupt output.
static bool record_error(double got, double ref, double& maxerr, double& maxref) {
    if (!std::isfinite(got) || !std::isfinite(ref)) return false;
    maxerr = std::max(maxerr, std::fabs(got - ref));
    maxref = std::max(maxref, std::fabs(ref));
    return true;
}
static bool oracle_selftest() {
    double err = 0, ref = 0;
    return record_error(1, 1, err, ref) &&
        !record_error(std::numeric_limits<double>::quiet_NaN(), 1, err, ref) &&
        !record_error(std::numeric_limits<double>::infinity(), 1, err, ref) &&
        !record_error(-std::numeric_limits<double>::infinity(), 1, err, ref) &&
        !record_error(1, std::numeric_limits<double>::quiet_NaN(), err, ref);
}

template <typename T>
static T narrow(float v) {
    if constexpr (std::is_same_v<T, sycl::half>) return (T) v;
    else return (T) v;   // bfloat16 constructor rounds
}

// C[M,N] = A[M,K] . B[K,N], A/B 16-bit (fp16 or bf16), FP32 accumulate, tile TM x TN x TK.
template <typename TA, typename TB, int TM, int TN, int TK>
static void run_f16(sycl::queue& q, const char* label, int M, int N, int K) {
    const int Kpad = (K + TK - 1) / TK * TK;
    const int gx = (M + TM - 1) / TM, gy = (N + TN - 1) / TN;
    const int Mp = gx * TM, Np = gy * TN;   // guard regions: the tiles read whole TMxTK / TKxTN blocks
    // A is M x Kpad guarded to Mp, B is Kpad x N guarded to Np.  Zero-padding makes the padded products exactly zero;
    // M/N tails are computed but never compared (the guard regions a tiled kernel really uses).
    std::vector<TA> ha((size_t) Mp * Kpad, (TA) 0), hb((size_t) Kpad * Np, (TB) 0);
    std::vector<float> fa((size_t) M * Kpad), fb((size_t) Kpad * N);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) {
            float v = (float) (((int) ((m * 131 + k * 7) % 29)) - 14) * 0.25f;
            ha[(size_t) m * Kpad + k] = narrow<TA>(v);
            fa[(size_t) m * Kpad + k] = (float) ha[(size_t) m * Kpad + k];
        }
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) {
            float v = (float) (((int) ((k * 17 + n * 5) % 23)) - 11) * 0.5f;
            hb[(size_t) k * Np + n] = narrow<TB>(v);
            fb[(size_t) k * N + n] = (float) hb[(size_t) k * Np + n];
        }
    std::vector<double> ref((size_t) M * N, 0.0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            double s = 0;
            for (int k = 0; k < K; ++k) s += (double) fa[(size_t) m * Kpad + k] * (double) fb[(size_t) k * N + n];
            ref[(size_t) m * N + n] = s;
        }
    auto* da = sycl::malloc_device<TA>((size_t) Mp * Kpad, q);
    auto* db = sycl::malloc_device<TB>((size_t) Kpad * Np, q);
    auto* dc = sycl::malloc_device<float>((size_t) Mp * Np, q);
    q.memcpy(da, ha.data(), ha.size() * sizeof(TA));
    q.memcpy(db, hb.data(), hb.size() * sizeof(TB));
    q.parallel_for(sycl::nd_range<2>({(size_t) gx, (size_t) gy * 16}, {1, 16}),
                   [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int gm = (int) it.get_group(0) * TM, gn = (int) it.get_group(1) * TN;
        mat::joint_matrix<sycl::sub_group, TA, mat::use::a, TM, TK, mat::layout::row_major> tA;
        mat::joint_matrix<sycl::sub_group, TB, mat::use::b, TK, TN, mat::layout::row_major> tB;
        mat::joint_matrix<sycl::sub_group, float, mat::use::accumulator, TM, TN> tC;
        mat::joint_matrix_fill(sg, tC, 0.0f);
        for (int tk = 0; tk < Kpad; tk += TK) {
            mat::joint_matrix_load(sg, tA,
                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(da + (size_t) gm * Kpad + tk), Kpad);
            mat::joint_matrix_load(sg, tB,
                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(db + (size_t) tk * Np + gn), Np);
            mat::joint_matrix_mad(sg, tC, tA, tB, tC);
        }
        mat::joint_matrix_store(sg, tC,
            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dc + (size_t) gm * Np + gn), Np, mat::layout::row_major);
    });
    std::vector<float> got((size_t) Mp * Np);
    q.memcpy(got.data(), dc, got.size() * sizeof(float)).wait_and_throw();
    double maxerr = 0, maxref = 0;
    bool finite = true;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            finite = record_error(got[(size_t) m * Np + n], ref[(size_t) m * N + n], maxerr, maxref) && finite;
        }
    const bool ok = finite && maxerr <= 1e-3 * std::max(1.0, maxref);
    std::printf("  %-4s tile %dx%dx%d M=%d N=%d K=%d (Kpad %d): max abs err %.3e (ref %.3e) %s\n", label, TM, TN, TK, M,
                N, K, Kpad, maxerr, maxref, ok ? "OK" : "FAIL");
    if (!ok) ++g_fail;
    ++g_ran;
    sycl::free(da, q); sycl::free(db, q); sycl::free(dc, q);
}

// INT8: C[M,N] = A[M,K] . B[K,N], A 8x32 tile, B 32x16 tile, C 8x16, INT32 accumulate.
static void run_i8(sycl::queue& q, int M, int N, int K) {
    constexpr int TM = 8, TN = 16, TK = 32;
    const int Kpad = (K + TK - 1) / TK * TK;
    const int gx = (M + TM - 1) / TM, gy = (N + TN - 1) / TN;
    const int Mp = gx * TM, Np = gy * TN;
    std::vector<int8_t> ha((size_t) Mp * Kpad, 0), hb((size_t) Kpad * Np, 0);
    for (int m = 0; m < M; ++m)
        for (int k = 0; k < K; ++k) ha[(size_t) m * Kpad + k] = (int8_t) (((m * 131 + k * 7) % 61) - 30);
    for (int k = 0; k < K; ++k)
        for (int n = 0; n < N; ++n) hb[(size_t) k * Np + n] = (int8_t) (((k * 17 + n * 5) % 51) - 25);
    std::vector<int32_t> ref((size_t) M * N, 0);
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) {
            int32_t s = 0;
            for (int k = 0; k < K; ++k)
                s += (int32_t) ha[(size_t) m * Kpad + k] * (int32_t) hb[(size_t) k * Np + n];
            ref[(size_t) m * N + n] = s;
        }
    auto* da = sycl::malloc_device<int8_t>((size_t) Mp * Kpad, q);
    auto* db = sycl::malloc_device<int8_t>((size_t) Kpad * Np, q);
    auto* dc = sycl::malloc_device<int32_t>((size_t) Mp * Np, q);
    q.memcpy(da, ha.data(), ha.size());
    q.memcpy(db, hb.data(), hb.size());
    q.parallel_for(sycl::nd_range<2>({(size_t) gx, (size_t) gy * 16}, {1, 16}),
                   [=](sycl::nd_item<2> it) [[sycl::reqd_sub_group_size(16)]] {
        const sycl::sub_group sg = it.get_sub_group();
        const int gm = (int) it.get_group(0) * TM, gn = (int) it.get_group(1) * TN;
        mat::joint_matrix<sycl::sub_group, int8_t, mat::use::a, TM, TK, mat::layout::row_major> tA;
        mat::joint_matrix<sycl::sub_group, int8_t, mat::use::b, TK, TN, mat::layout::row_major> tB;
        mat::joint_matrix<sycl::sub_group, int32_t, mat::use::accumulator, TM, TN> tC;
        mat::joint_matrix_fill(sg, tC, 0);
        for (int tk = 0; tk < Kpad; tk += TK) {
            mat::joint_matrix_load(sg, tA,
                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(da + (size_t) gm * Kpad + tk), Kpad);
            mat::joint_matrix_load(sg, tB,
                sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(db + (size_t) tk * Np + gn), Np);
            mat::joint_matrix_mad(sg, tC, tA, tB, tC);
        }
        mat::joint_matrix_store(sg, tC,
            sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(dc + (size_t) gm * Np + gn), Np, mat::layout::row_major);
    });
    std::vector<int32_t> got((size_t) Mp * Np);
    q.memcpy(got.data(), dc, got.size() * sizeof(int32_t)).wait_and_throw();
    long maxerr = 0;
    for (int m = 0; m < M; ++m)
        for (int n = 0; n < N; ++n) maxerr = std::max(maxerr, (long) std::labs((long) got[(size_t) m * Np + n] - ref[(size_t) m * N + n]));
    const bool ok = maxerr == 0;
    std::printf("  i8   tile %dx%dx%d M=%d N=%d K=%d (Kpad %d): max int err %ld %s\n", TM, TN, TK, M, N, K, Kpad, maxerr,
                ok ? "OK" : "FAIL");
    if (!ok) ++g_fail;
    ++g_ran;
    sycl::free(da, q); sycl::free(db, q); sycl::free(dc, q);
}

int main(int argc, char** argv) {
    if (!oracle_selftest()) return 1;
    if (argc > 1 && std::string(argv[1]) == "--oracle-selftest") {
        std::puts("matrix oracle: NaN/Inf negative controls passed");
        return 0;
    }
    const bool linked = argc > 1 && std::string(argv[1]) == "--linked";
    try {
        sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order{});
        auto dev = q.get_device();
        std::printf("matrix probe%s: device %s\n", linked ? " (linked)" : "", dev.get_info<sycl::info::device::name>().c_str());
        // representative widths: the Coder's expert shapes are 1280x2560 and 2560x640; here a tiled K and multiple
        // work-items over a handful of tiles, plus a ragged tail case.
        try { run_f16<sycl::half, sycl::half, 16, 16, 16>(q, "fp16", 32, 32, 64); } catch (const sycl::exception& e) { std::printf("  fp16 basic threw: %s\n", e.what()); ++g_fail; }
        try { run_f16<sycl::half, sycl::half, 16, 16, 16>(q, "fp16", 24, 40, 80); } catch (const sycl::exception& e) { std::printf("  fp16 rag  threw: %s\n", e.what()); ++g_fail; }
        try { run_f16<sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, 16, 16, 16>(q, "bf16", 32, 32, 64); } catch (const sycl::exception& e) { std::printf("  bf16 basic threw: %s\n", e.what()); ++g_fail; }
        try { run_f16<sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, 16, 16, 16>(q, "bf16", 24, 40, 80); } catch (const sycl::exception& e) { std::printf("  bf16 rag  threw: %s\n", e.what()); ++g_fail; }
        try { run_i8(q, 16, 32, 64); } catch (const sycl::exception& e) { std::printf("  i8 basic threw: %s\n", e.what()); ++g_fail; }
        try { run_i8(q, 24, 40, 80); } catch (const sycl::exception& e) { std::printf("  i8 rag  threw: %s\n", e.what()); ++g_fail; }
    } catch (const std::exception& e) {
        std::fprintf(stderr, "matrix probe: %s\n", e.what());
        return 77;
    }
    std::printf("matrix probe: %d products run, %d failed\n", g_ran, g_fail);
    if (g_ran == 0) return 77;
    return g_fail == 0 ? 0 : 1;
}
