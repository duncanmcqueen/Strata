// The BLAS shim's GEMM with operands placed past 4 GiB into a large allocation: oneMKL 2026.1 reads the last, partial
// tile of such a B from the wrong place (sycl_runtime/blas.cpp stages those operands).  A BF16 GEMM of the prompt
// path's shape (N 10240, K 320, 1845 columns - a partial last tile) with B, then C, placed at offsets below and past
// 4 GiB of one 4.5 GiB allocation, compared bitwise with the same GEMM on separate allocations.
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

int main() {
    const int N = 10240, K = 320, T = 1845;
    std::mt19937 rng(11);
    std::vector<uint16_t> X((size_t) T * K), W((size_t) N * K);
    for (auto& v : X) v = (uint16_t) (0x3c00 + (rng() & 0x3ff)) ^ (uint16_t) ((rng() & 1) << 15);
    for (auto& v : W) v = (uint16_t) (0x3c00 + (rng() & 0x3ff)) ^ (uint16_t) ((rng() & 1) << 15);
    cublasHandle_t h;
    cudaStream_t st;
    cublasCreate(&h);
    cudaStreamCreate(&st);
    cublasSetStream(h, st);
    uint16_t *dX, *dW;
    float* dY;
    cudaMalloc((void**) &dX, X.size() * 2);
    cudaMalloc((void**) &dW, W.size() * 2);
    cudaMalloc((void**) &dY, (size_t) T * N * 4);
    cudaMemcpy(dX, X.data(), X.size() * 2, cudaMemcpyHostToDevice);
    cudaMemcpy(dW, W.data(), W.size() * 2, cudaMemcpyHostToDevice);
    const float one = 1.0f, zero = 0.0f;
    auto gemm = [&](const uint16_t* x, float* y, float beta) {
        return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, N, T, K, &one, dW, CUDA_R_16BF, K, x, CUDA_R_16BF, K, &beta, y,
                            CUDA_R_32F, N, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
    };
    gemm(dX, dY, zero);
    gemm(dX, dY, one);   // beta = 1: twice the product
    cudaStreamSynchronize(st);
    std::vector<float> ref((size_t) T * N), got(ref.size());
    cudaMemcpy(ref.data(), dY, ref.size() * 4, cudaMemcpyDeviceToHost);
    const size_t big = (size_t) 4608 << 20;   // 4.5 GiB
    uint8_t* arena = nullptr;
    if (cudaMalloc((void**) &arena, big) != cudaSuccess) {
        std::printf("gemm_4g: SKIP (no 4.5 GiB allocation on this device)\n");
        return 0;
    }
    cudaMemset(arena, 0x7f, big);
    int fails = 0;
    const size_t xb = X.size() * 2, yb = (size_t) T * N * 4;
    for (int far = 0; far < 2; ++far)
        for (int which = 0; which < 2; ++which) {   // 0: B in the arena, 1: C in the arena
            const size_t off = far ? big - (which ? yb : xb) - 4096 : 4096;
            const uint16_t* x = dX;
            float* y = dY;
            if (which == 0) {
                cudaMemcpy(arena + off, X.data(), xb, cudaMemcpyHostToDevice);
                x = (const uint16_t*) (arena + off);
            } else {
                y = (float*) (arena + off);
            }
            cudaMemset(y, 0, yb);
            gemm(x, y, zero);
            gemm(x, y, one);
            cudaStreamSynchronize(st);
            cudaMemcpy(got.data(), y, yb, cudaMemcpyDeviceToHost);
            const bool ok = std::memcmp(got.data(), ref.data(), yb) == 0;
            std::printf("gemm_4g: %s at %.2f GiB: %s\n", which ? "C" : "B", off / 1073741824.0,
                        ok ? "bitwise equal" : "DIFFERS");
            fails += !ok;
        }
    // two GEMMs of half the rows into one C interleaved by ldc = 2m past 4 GiB (the prompt path's alpha and beta
    // projections): each writes its m rows only, and the pair gives the full GEMM's outputs
    {
        const int M = 48, TT = 3;
        const size_t cb = (size_t) TT * 2 * M * 4;
        float* y = (float*) (arena + big - cb - 4096);
        cudaMemset(y, 0, cb);
        auto half = [&](int r0, float beta) {
            return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, M, TT, K, &one, dW + (size_t) r0 * K, CUDA_R_16BF, K, dX,
                                CUDA_R_16BF, K, &beta, y + r0, CUDA_R_32F, 2 * M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        };
        half(0, zero); half(0, one); half(M, zero); half(M, one);
        float* y2 = nullptr;
        cudaMalloc((void**) &y2, cb);
        auto whole = [&](float beta) {
            return cublasGemmEx(h, CUBLAS_OP_T, CUBLAS_OP_N, 2 * M, TT, K, &one, dW, CUDA_R_16BF, K, dX, CUDA_R_16BF, K,
                                &beta, y2, CUDA_R_32F, 2 * M, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT);
        };
        cudaMemset(y2, 0, cb);
        whole(zero); whole(one);
        cudaStreamSynchronize(st);
        std::vector<float> a((size_t) TT * 2 * M), b2(a.size());
        cudaMemcpy(a.data(), y, cb, cudaMemcpyDeviceToHost);
        cudaMemcpy(b2.data(), y2, cb, cudaMemcpyDeviceToHost);
        const bool ok = std::memcmp(a.data(), b2.data(), cb) == 0;
        std::printf("gemm_4g: two interleaved halves of C (ldc = 2m) past 4 GiB: %s\n", ok ? "bitwise equal" : "DIFFERS");
        fails += !ok;
        // few columns run the shim's own kernel: against a double-precision reference (2 x the product: beta = 1)
        auto bf = [](uint16_t v) { uint32_t u = (uint32_t) v << 16; float f; std::memcpy(&f, &u, 4); return (double) f; };
        double worst = 0;
        for (int t = 0; t < TT; ++t)
            for (int r = 0; r < 2 * M; ++r) {
                double ref = 0;
                for (int q = 0; q < K; ++q) ref += bf(W[(size_t) r * K + q]) * bf(X[(size_t) t * K + q]);
                ref *= 2;
                const double got = a[(size_t) t * 2 * M + r];
                worst = std::max(worst, std::abs(got - ref) / std::max(1.0, std::abs(ref)));
            }
        const bool close = worst < 1e-5;
        std::printf("gemm_4g: few columns against a double reference: worst relative error %.2e: %s\n", worst,
                    close ? "ok" : "TOO LARGE");
        fails += !close;
        cudaFree(y2);
    }
    std::printf("gemm_4g: %d failures\n", fails);
    return fails ? 1 : 0;
}
