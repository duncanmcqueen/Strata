// Q6_K in the reordered layout (native_mmvq.hpp): the same random matrix uploaded as GGUF blocks and reordered and
// registered, compared bitwise - native_mmvq at 1..8 columns (the exact path, the engine's) and the prompt path's
// dequant_f16, whole and in a row slice.  A matrix of 4,104 rows by 2,560: rows a row group does not fill.
#include <cuda_runtime.h>
#include "strata/kernels/dequant_bf16.hpp"
#include "strata/kernels/native_mmvq.hpp"
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace k = strata::kernels;

int main() {
    const int n_in = 2560, n_out = 4104;
    const size_t bytes = k::native_mmvq_weight_bytes(14, n_in, n_out), nb = bytes / 210;
    std::mt19937 rng(7);
    std::vector<uint8_t> w(bytes), r(bytes);
    for (auto& b : w) b = (uint8_t) rng();
    for (size_t i = 0; i < nb; ++i) {   // a finite d of moderate size per block
        const uint16_t d = (uint16_t) (0x1c00 + (rng() & 0x3ff));
        std::memcpy(&w[i * 210 + 208], &d, 2);
    }
    k::native_q6_k_reorder(w.data(), r.data(), nb);
    cudaStream_t s;
    cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);
    void *dw = nullptr, *dr = nullptr, *xq = nullptr;
    float *x = nullptr, *y0 = nullptr, *y1 = nullptr;
    cudaMalloc(&dw, bytes);
    cudaMalloc(&dr, bytes);
    cudaMemcpy(dw, w.data(), bytes, cudaMemcpyHostToDevice);
    cudaMemcpy(dr, r.data(), bytes, cudaMemcpyHostToDevice);
    k::native_q6_k_register_reordered(dr, nb);
    if (k::native_q6_k_reordered_blocks(dr) != nb || k::native_q6_k_reordered_blocks(dw) != 0) {
        std::printf("FAIL: registry\n");
        return 1;
    }
    cudaMalloc((void**) &x, (size_t) 8 * n_in * 4);
    cudaMalloc(&xq, k::native_q8_1_bytes(n_in, 8));
    cudaMalloc((void**) &y0, (size_t) 8 * n_out * 4);
    cudaMalloc((void**) &y1, (size_t) 8 * n_out * 4);
    int failures = 0;
    for (int ncols = 1; ncols <= 8; ++ncols) {
        std::vector<float> hx((size_t) ncols * n_in);
        for (auto& v : hx) v = (float) ((int) (rng() % 4001) - 2000) / 1000.0f;
        cudaMemcpy(x, hx.data(), hx.size() * 4, cudaMemcpyHostToDevice);
        k::native_quantize_q8_1(x, xq, n_in, ncols, s);
        k::native_mmvq(14, dw, xq, y0, n_in, n_out, ncols, s);
        k::native_mmvq(14, dr, xq, y1, n_in, n_out, ncols, s);
        cudaStreamSynchronize(s);
        std::vector<float> a((size_t) ncols * n_out), b(a.size());
        cudaMemcpy(a.data(), y0, a.size() * 4, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), y1, b.size() * 4, cudaMemcpyDeviceToHost);
        size_t diff = 0, nonzero = 0;
        for (size_t i = 0; i < a.size(); ++i) {
            diff += std::memcmp(&a[i], &b[i], 4) != 0;
            nonzero += a[i] != 0.0f;
        }
        std::printf("mmvq ncols %d: %zu of %zu outputs differ\n", ncols, diff, a.size());
        if (diff || nonzero < a.size() / 2) ++failures;
    }
    // dequant: whole, and rows [1000, 1000 + 777)
    const int64_t slices[2][2] = {{0, n_out}, {1000, 777}};
    for (const auto& sl : slices) {
        const size_t n = (size_t) sl[1] * n_in;
        uint16_t *d0 = nullptr, *d1 = nullptr;
        cudaMalloc((void**) &d0, n * 2);
        cudaMalloc((void**) &d1, n * 2);
        k::dequant_f16(14, dw, sl[0], sl[1], n_in, d0, s);
        k::dequant_f16(14, dr, sl[0], sl[1], n_in, d1, s);
        cudaStreamSynchronize(s);
        std::vector<uint16_t> a(n), b(n);
        cudaMemcpy(a.data(), d0, n * 2, cudaMemcpyDeviceToHost);
        cudaMemcpy(b.data(), d1, n * 2, cudaMemcpyDeviceToHost);
        const bool same = std::memcmp(a.data(), b.data(), n * 2) == 0;
        std::printf("dequant_f16 rows %lld+%lld: %s\n", (long long) sl[0], (long long) sl[1], same ? "same" : "DIFFERENT");
        failures += !same;
        cudaFree(d0);
        cudaFree(d1);
    }
    k::native_q6_k_unregister(dr);
    if (k::native_q6_k_reordered_blocks(dr) != 0) ++failures;
    std::printf(failures ? "FAIL\n" : "PASS\n");
    return failures ? 1 : 0;
}
