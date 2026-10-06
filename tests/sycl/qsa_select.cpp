// CPU sorting oracle and FP64 scoring oracle, with changing graph replay.
#include "strata/kernels/qsa_select.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <numeric>
#include <random>
#include <vector>
namespace k = strata::kernels;
namespace {
void ck(cudaError_t e) {
    if (e != cudaSuccess) {
        std::fprintf(stderr, "%s\n", cudaGetErrorString(e));
        std::exit(2);
    }
}
struct Memory {
    std::vector<void *> ptrs;
    template <class T> T *alloc(size_t n) {
        T *p = nullptr;
        ck(cudaMalloc(&p, n * sizeof(T)));
        ptrs.push_back(p);
        return p;
    }
    template <class T> T *upload(const std::vector<T> &v) {
        auto p = alloc<T>(v.size());
        ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
        return p;
    }
    ~Memory() {
        for (auto p : ptrs)
            ck(cudaFree(p));
    }
};
std::vector<int32_t> sorted_ids(const float *scores, int n, int width) {
    std::vector<int32_t> ids(n);
    std::iota(ids.begin(), ids.end(), 0);
    std::stable_sort(ids.begin(), ids.end(), [&](int a, int b) {
        float x = scores[a / 4], y = scores[b / 4];
        if (std::isnan(x))
            return false;
        if (std::isnan(y))
            return true;
        return x > y;
    });
    ids.resize(width);
    std::sort(ids.begin(), ids.end());
    return ids;
}
bool selection(int stride) {
    const auto s = k::qsa_real_shapes();
    int cap = int(k::qsa_selection_width(k::kTopkMaxCells, s));
    constexpr int NQ = 4;
    Memory mem;
    std::vector<float> scores(NQ * stride);
    auto ds = mem.upload(scores);
    auto dt = mem.alloc<int32_t>(NQ * k::kStepCount);
    auto a = mem.alloc<int32_t>(NQ * cap);
    auto b = mem.alloc<int32_t>(NQ * cap);
    auto c = mem.alloc<int32_t>(NQ * cap);
    cudaStream_t st;
    ck(cudaStreamCreate(&st));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    k::qsa_block_topk(ds, dt, NQ, stride, cap, s, a, st);
    k::qsa_block_topk_ref(ds, dt, NQ, stride, cap, s, b, st);
    ck(cudaStreamEndCapture(st, &graph));
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    const int contexts[] = {0, 1, 4, 511, 2051, 2052, 33000, 131072, 135167, 135168, 262144};
    bool ok = true;
    int checks = 0;
    for (int ctx : contexts) {
        if (ctx / 4 + 1 > stride)
            continue;
        for (int mode = 0; mode < 3; ++mode) {
            std::mt19937 rng(913 + ctx + mode);
            std::vector<int32_t> steps(NQ * k::kStepCount);
            int active = 1;
            for (int t = 0; t < NQ; ++t) {
                int n = std::max(0, ctx - t), w = int(k::qsa_selection_width(n, s));
                // Also exercise a small equal-score budget that cuts a four-cell block.
                if (t == 3 && w > 13)
                    w = 13;
                steps[t * k::kStepCount + k::kStepNKv] = n;
                steps[t * k::kStepCount + k::kStepNBid] = n / 4;
                steps[t * k::kStepCount + k::kStepWidth] = w;
                active = std::max(active, n / 4 + 1);
                for (int j = 0; j < stride; ++j) {
                    float x = mode == 0   ? float(int(rng() % 4096) - 2048)
                              : mode == 1 ? 0.f
                                          : float(j % 7) - 3.f;
                    if (mode == 2) {
                        switch (j % 29) {
                        case 0:
                            x = std::numeric_limits<float>::quiet_NaN();
                            break;
                        case 1:
                            x = -std::numeric_limits<float>::infinity();
                            break;
                        case 2:
                            x = std::numeric_limits<float>::infinity();
                            break;
                        case 3:
                            x = -0.f;
                            break;
                        case 4:
                            x = 0.f;
                            break;
                        }
                    }
                    scores[t * stride + j] = x;
                }
            }
            ck(cudaMemcpyAsync(ds, scores.data(), scores.size() * 4, cudaMemcpyHostToDevice, st));
            ck(cudaMemcpyAsync(dt, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice, st));
            for (auto out : {a, b, c})
                ck(cudaMemsetAsync(out, 0xff, NQ * cap * 4, st));
            ck(cudaGraphLaunch(exec, st));
            k::qsa_block_topk(ds, dt, NQ, stride, cap, s, c, st, active);
            ck(cudaStreamSynchronize(st));
            for (auto out : {a, b, c}) {
                std::vector<int32_t> actual(NQ * cap);
                ck(cudaMemcpy(actual.data(), out, actual.size() * 4, cudaMemcpyDeviceToHost));
                for (int t = 0; t < NQ; ++t) {
                    int n = steps[t * k::kStepCount + k::kStepNKv],
                        w = steps[t * k::kStepCount + k::kStepWidth];
                    auto expected = sorted_ids(scores.data() + t * stride, n, w);
                    if (!std::equal(expected.begin(), expected.end(), actual.begin() + t * cap)) {
                        std::fprintf(stderr, "sorting mismatch stride %d ctx %d mode %d query %d\n", stride,
                                     ctx, mode, t);
                        ok = false;
                    }
                    if (!std::all_of(actual.begin() + t * cap + w, actual.begin() + (t + 1) * cap,
                                     [](int32_t x) { return x == -1; }))
                        ok = false;
                }
            }
            ++checks;
        }
    }
    ck(cudaMemsetAsync(c, 0xff, NQ * cap * 4, st));
    if (k::qsa_block_topk_cluster(ds, dt, NQ, stride, cap, s, c, st))
        ok = false;
    ck(cudaStreamSynchronize(st));
    std::vector<int32_t> untouched(NQ * cap);
    ck(cudaMemcpy(untouched.data(), c, untouched.size() * 4, cudaMemcpyDeviceToHost));
    if (!std::all_of(untouched.begin(), untouched.end(), [](int32_t x) { return x == -1; }))
        ok = false;
    ck(cudaGraphExecDestroy(exec));
    ck(cudaGraphDestroy(graph));
    ck(cudaStreamDestroy(st));
    std::printf("selection stride %d: %d replay/sort cases %s\n", stride, checks, ok ? "PASS" : "FAIL");
    return ok;
}
bool scoring() {
    constexpr int NQ = 8, NB = 1025;
    const auto s = k::qsa_real_shapes();
    Memory mem;
    std::mt19937 rng(737);
    std::uniform_real_distribution<float> dist(-1.f, 1.f);
    std::vector<float> pooled(NB * 128), dead(128), q(NQ * 512);
    for (auto *v : {&pooled, &dead, &q})
        for (auto &x : *v)
            x = dist(rng);
    auto dp = mem.upload(pooled);
    auto dd = mem.upload(dead);
    auto dq = mem.upload(q);
    auto dt = mem.alloc<int32_t>(NQ * k::kStepCount);
    auto multi = mem.alloc<float>(NQ * NB);
    auto bounded = mem.alloc<float>(NQ * NB);
    cudaStream_t st;
    ck(cudaStreamCreate(&st));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    k::qsa_block_scores(dp, dd, dq, dt, NQ, NB, s, multi, st);
    ck(cudaStreamEndCapture(st, &graph));
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    bool ok = true;
    double error = 0;
    for (int ctx : {0, 511, 2052, 4096}) {
        std::vector<int32_t> steps(NQ * k::kStepCount);
        int active = 1;
        for (int t = 0; t < NQ; ++t) {
            int n = std::max(0, ctx - t);
            steps[t * k::kStepCount + k::kStepNKv] = n;
            steps[t * k::kStepCount + k::kStepNBid] = n / 4;
            active = std::max(active, n / 4 + 1);
        }
        ck(cudaMemcpyAsync(dt, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice, st));
        std::vector<float> poison(NQ * NB, -12345.f);
        ck(cudaMemcpyAsync(multi, poison.data(), poison.size() * 4, cudaMemcpyHostToDevice, st));
        ck(cudaMemcpyAsync(bounded, poison.data(), poison.size() * 4, cudaMemcpyHostToDevice, st));
        ck(cudaGraphLaunch(exec, st));
        k::qsa_block_scores(dp, dd, dq, dt, NQ, NB, s, bounded, st, active);
        ck(cudaStreamSynchronize(st));
        std::vector<float> a(NQ * NB), b(a.size());
        ck(cudaMemcpy(a.data(), multi, a.size() * 4, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(b.data(), bounded, b.size() * 4, cudaMemcpyDeviceToHost));
        if (std::memcmp(a.data(), b.data(), a.size() * 4)) {
            std::fprintf(stderr, "multi/bounded scores differ ctx %d\n", ctx);
            ok = false;
        }
        for (int t = 0; t < NQ; ++t) {
            int n = steps[t * k::kStepCount + k::kStepNKv], bid = n / 4;
            for (int block = 0; block < NB; ++block) {
                if (block > bid) {
                    if (a[t * NB + block] != -12345.f)
                        ok = false;
                    continue;
                }
                const float *key = block == bid ? dead.data() : pooled.data() + block * 128;
                double ref = 0;
                for (int h = 0; h < 4; ++h) {
                    double dot = 0;
                    for (int d = 0; d < 128; ++d)
                        dot += double(q[t * 512 + h * 128 + d]) * key[d];
                    ref += std::max(0.0, dot);
                }
                if (block == bid && n % 4)
                    ref += 1e9;
                double err = std::fabs(a[t * NB + block] - ref);
                if (block == bid && n % 4) {
                    if (a[t * NB + block] != float(ref))
                        ok = false;
                } else {
                    error = std::max(error, err);
                    if (!std::isfinite(a[t * NB + block]) || err > 2e-5 * (1 + std::fabs(ref)))
                        ok = false;
                }
            }
        }
    }
    // Architecture-specific probes must not alter any output when unavailable.
    ck(cudaMemsetAsync(bounded, 0x5a, NQ * NB * 4, st));
    if (k::qsa_block_scores_tc(dp, dd, dq, dt, NQ, NB, s, bounded, st, NB))
        ok = false;
    ck(cudaStreamSynchronize(st));
    std::vector<uint32_t> sentinel(NQ * NB);
    ck(cudaMemcpy(sentinel.data(), bounded, sentinel.size() * 4, cudaMemcpyDeviceToHost));
    if (!std::all_of(sentinel.begin(), sentinel.end(), [](uint32_t x) { return x == 0x5a5a5a5a; }))
        ok = false;
    ck(cudaGraphExecDestroy(exec));
    ck(cudaGraphDestroy(graph));
    ck(cudaStreamDestroy(st));
    std::printf("scoring FP64 max error %.3g, multi/bounded replay %s\n", error, ok ? "PASS" : "FAIL");
    return ok;
}
} // namespace
int main() {
    bool ok = scoring();
    for (int stride : {32770, 65538})
        ok = selection(stride) && ok;
    return ok ? 0 : 1;
}
