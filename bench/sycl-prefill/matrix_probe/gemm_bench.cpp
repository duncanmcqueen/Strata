// Microbenchmark: the fallback expert GEMMs through oneMKL SYCL BLAS, timing GPU work
// (events) separately from host submission, at the Coder expert shapes.  It prints
// both, so a "GEMM phase" that is really host dispatch is visible.
//
// Shapes (row-major Y[T,N] = X[T,K] . W[N,K]^T, as Gemm::f16 sends them):
//   gate/up: T=ne, N=1280, K=2560
//   down:    T=ne, N=2560, K=640
#include <sycl/sycl.hpp>
#include <oneapi/mkl/blas.hpp>

#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <vector>

using Clock = std::chrono::steady_clock;

static double bench(sycl::queue& q, int T, int N, int K, int iters, bool host_timed) {
    const size_t ysz = (size_t) N * T;
    sycl::half* X = sycl::malloc_device<sycl::half>((size_t) T * K, q);
    sycl::half* W = sycl::malloc_device<sycl::half>((size_t) N * K, q);
    float* Y = sycl::malloc_device<float>(ysz, q);
    q.fill(X, sycl::half(1.0f), (size_t) T * K);
    q.fill(W, sycl::half(0.5f), (size_t) N * K);
    q.wait();
    // warmup
    for (int i = 0; i < 3; ++i)
        oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, N, T, K,
                                              1.0f, W, K, X, K, 0.0f, Y, N);
    q.wait();
    sycl::event e0 = q.submit([&](sycl::handler& h) { h.single_task([=] {}); });
    auto t0 = Clock::now();
    for (int i = 0; i < iters; ++i)
        oneapi::mkl::blas::column_major::gemm(q, oneapi::mkl::transpose::trans, oneapi::mkl::transpose::nontrans, N, T, K,
                                              1.0f, W, K, X, K, 0.0f, Y, N);
    sycl::event e1 = q.submit([&](sycl::handler& h) { h.single_task([=] {}); });
    auto t_submit = std::chrono::duration<double, std::milli>(Clock::now() - t0).count();
    e0.wait();
    e1.wait();
    double gpu_ms = (e1.get_profiling_info<sycl::info::event_profiling::command_end>() -
                     e0.get_profiling_info<sycl::info::event_profiling::command_start>()) / 1e6;
    if (host_timed) std::printf("    submit %.3f ms/call\n", t_submit / iters);
    std::printf("    gpu %.3f ms/call  (%.1f GFLOP/s)\n", gpu_ms / iters,
                2.0 * T * N * K / 1e9 / (gpu_ms / iters / 1000.0));
    sycl::free(X, q); sycl::free(W, q); sycl::free(Y, q);
    return gpu_ms / iters;
}

int main() {
    sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::enable_profiling{});
    std::printf("gemm microbench: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());
    struct S { int T, N, K; const char* name; };
    const S shapes[] = {
        {320, 1280, 2560, "gate/up ne=320"},   { 64, 1280, 2560, "gate/up ne=64"},
        {320, 2560, 640, "down ne=320"},       { 64, 2560, 640, "down ne=64"},
        {1024, 2560, 640, "down ne=1024"},     { 1024, 1280, 2560, "gate/up ne=1024"},
        {8192, 2560, 2560, "dense 8192x2560x2560"},
    };
    for (const S& s : shapes) {
        std::printf("  %-22s T=%d N=%d K=%d\n", s.name, s.T, s.N, s.K);
        bench(q, s.T, s.N, s.K, 30, true);
    }
    return 0;
}
