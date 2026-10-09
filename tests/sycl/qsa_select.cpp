// CPU sorting oracle and FP64 scoring oracle, with changing graph replay.
#include "strata/kernels/qsa_select.hpp"
#include "strata/sycl_runtime/device_profile.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
#include <limits>
#include <cstdint>
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
bool hierarchical() {
    const auto s = k::qsa_real_shapes();
    const int cap = int(k::qsa_selection_width(k::kTopkMaxCells, s));
    constexpr int NQ = 4;
    // strides around 1024*33 = 33792 blocks and up to the 262144-cell context
    const int64_t strides[] = {33793, 33794, 65538, 65539};
    bool ok = true;
    int checks = 0;
    for (int64_t stride : strides) {
        Memory mem;
        std::vector<float> scores(NQ * stride);
        auto ds = mem.upload(scores);
        auto dt = mem.alloc<int32_t>(NQ * k::kStepCount);
        auto a = mem.alloc<int32_t>(NQ * cap);
        auto b = mem.alloc<int32_t>(NQ * cap);
        auto h = mem.alloc<int32_t>(NQ * cap);
        const uint64_t ws_bytes = k::qsa_topk_workspace_bytes(NQ, stride);
        auto ws = mem.alloc<uint8_t>(ws_bytes);
        // the runtime selector must agree on the workspace size (single source of truth)
        if (ws_bytes != strata::sycl_runtime::topk_hier_scratch_bytes(NQ, stride, k::kQsaTopkTileBlocks)) {
            std::fprintf(stderr, "workspace formula mismatch: %llu vs runtime\n", (unsigned long long) ws_bytes);
            ok = false;
        }
        cudaStream_t st;
        ck(cudaStreamCreate(&st));
        cudaGraph_t graph;
        cudaGraphExec_t exec;
        ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
        k::qsa_block_topk_ref(ds, dt, NQ, stride, cap, s, b, st);
        ck(cudaStreamEndCapture(st, &graph));
        ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
        const int64_t contexts[] = {2051, 2052, 3000, 4100, 5000, 9000, 20000, 40000, 135167, 262144};
        for (int64_t ctx : contexts) {
            if (ctx / 4 + 1 > stride)
                continue;
            for (int mode = 0; mode < 4; ++mode) {
                std::mt19937 rng(1234 + (unsigned) ctx + mode);
                std::vector<int32_t> steps(NQ * k::kStepCount);
                for (int t = 0; t < NQ; ++t) {
                    int64_t n = ctx - t;
                    if (n < 0) n = 0;
                    int64_t w = (mode == 3) ? std::min<int64_t>(n + 5, (int64_t) cap) : (int64_t) k::qsa_selection_width(n, s);
                    if (t == 3 && w > 13 && mode != 3) w = 13;  // equal-score budget cutting a block
                    steps[t * k::kStepCount + k::kStepNKv] = (int32_t) n;
                    steps[t * k::kStepCount + k::kStepNBid] = (int32_t) (n / 4);
                    steps[t * k::kStepCount + k::kStepWidth] = (int32_t) w;
                    for (int64_t j = 0; j < stride; ++j) {
                        float x = mode == 0 ? float(int(rng() % 4096) - 2048)
                                  : mode == 1 ? 0.f
                                  : mode == 2 ? float(j % 7) - 3.f
                                              : float(int(rng() % 8));
                        if (mode == 2) {
                            switch (j % 29) {
                            case 0: x = std::numeric_limits<float>::quiet_NaN(); break;
                            case 1: x = -std::numeric_limits<float>::infinity(); break;
                            case 2: x = std::numeric_limits<float>::infinity(); break;
                            case 3: x = -0.f; break;
                            case 4: x = 0.f; break;
                            }
                        }
                        scores[t * stride + j] = x;
                    }
                }
                ck(cudaMemcpyAsync(ds, scores.data(), scores.size() * 4, cudaMemcpyHostToDevice, st));
                ck(cudaMemcpyAsync(dt, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice, st));
                for (auto out : {a, b, h})
                    ck(cudaMemsetAsync(out, 0xff, NQ * cap * 4, st));
                // replay the reference graph (short/long/short change via steps), then run hier directly
                ck(cudaGraphLaunch(exec, st));
                if (!k::qsa_block_topk_hier(ds, dt, NQ, stride, cap, s, h, st, ws, ws_bytes)) {
                    std::fprintf(stderr, "hier refused stride %lld ctx %lld\n", (long long) stride, (long long) ctx);
                    ok = false;
                }
                ck(cudaStreamSynchronize(st));
                std::vector<int32_t> av(NQ * cap), bv(NQ * cap), hv(NQ * cap);
                ck(cudaMemcpy(av.data(), a, av.size() * 4, cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(bv.data(), b, bv.size() * 4, cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(hv.data(), h, hv.size() * 4, cudaMemcpyDeviceToHost));
                for (int t = 0; t < NQ; ++t) {
                    int64_t n = steps[t * k::kStepCount + k::kStepNKv];
                    int64_t w = steps[t * k::kStepCount + k::kStepWidth];
                    int64_t ww = std::min<int64_t>(std::min<int64_t>(w, (int64_t) cap), n);
                    auto expected = sorted_ids(scores.data() + t * stride, (int) n, (int) ww);
                    if (!std::equal(expected.begin(), expected.end(), hv.begin() + t * cap)) {
                        std::fprintf(stderr, "hier mismatch stride %lld ctx %lld mode %d q %d n=%lld w=%lld cap=%d exp=%zu\n",
                                     (long long) stride, (long long) ctx, mode, t, (long long) n, (long long) w,
                                     cap, expected.size());
                        for (size_t z = 0; z < 8 && z < expected.size(); ++z)
                            std::fprintf(stderr, "   [%zu] exp=%d ref=%d hier=%d\n", z, expected[z],
                                         bv[t * cap + z], hv[t * cap + z]);
                        ok = false;
                    }
                    if (!std::equal(expected.begin(), expected.end(), bv.begin() + t * cap)) ok = false;
                    // poisoned padding past the selection width untouched
                    if (!std::all_of(hv.begin() + t * cap + expected.size(), hv.begin() + (t + 1) * cap,
                                     [](int32_t x) { return x == -1; }))
                        ok = false;
                }
                ++checks;
            }
        }
        ck(cudaGraphExecDestroy(exec));
        ck(cudaGraphDestroy(graph));
        ck(cudaStreamDestroy(st));
    }
    std::printf("hierarchical: %d replay/sort cases %s\n", checks, ok ? "PASS" : "FAIL");
    return ok;
}
bool tiled_scoring() {
    const auto s = k::qsa_real_shapes();
    bool ok = true;
    const int64_t strides[] = {1025, 32770};
    for (int64_t stride : strides) {
        for (int nq : {1, 4, 8, 16, 256}) {
            Memory mem;
            std::mt19937 rng(4242 + nq);
            std::uniform_real_distribution<float> dist(-1.f, 1.f);
            const int64_t NB = stride;
            std::vector<float> pooled((size_t)(NB * 128)), dead(128), q((size_t)(nq * 512));
            for (auto *v : {&pooled, &dead, &q})
                for (auto &x : *v) x = dist(rng);
            // per query a different n_bid, some tails incomplete (n_kv % 4 != 0)
            std::vector<int32_t> steps((size_t)(nq * k::kStepCount));
            int64_t active = 1;
            for (int t = 0; t < nq; ++t) {
                int64_t n = 100 + (t * 37) % 800;           // varies across the query tile
                if (t % 3 == 0) n += 1;                     // incomplete tail
                steps[t * k::kStepCount + k::kStepNKv] = (int32_t)n;
                steps[t * k::kStepCount + k::kStepNBid] = (int32_t)(n / 4);
                steps[t * k::kStepCount + k::kStepWidth] = (int32_t)k::qsa_selection_width(n, s);
                active = std::max(active, n / 4 + 1);
            }
            auto dp = mem.upload(pooled);
            auto dd = mem.upload(dead);
            auto dq = mem.upload(q);
            auto dt = mem.upload(steps);
            auto a = mem.alloc<float>((size_t)(nq * NB));
            auto b = mem.alloc<float>((size_t)(nq * NB));
            cudaStream_t st;
            ck(cudaStreamCreate(&st));
            for (int64_t ab : {int64_t(-1), active}) {
                std::vector<float> poison((size_t)(nq * NB), -999.f);
                ck(cudaMemcpyAsync(a, poison.data(), poison.size() * 4, cudaMemcpyHostToDevice, st));
                ck(cudaMemcpyAsync(b, poison.data(), poison.size() * 4, cudaMemcpyHostToDevice, st));
                k::qsa_block_scores(dp, dd, dq, dt, nq, NB, s, a, st, ab);
                if (!k::qsa_block_scores_tiled(dp, dd, dq, dt, nq, NB, s, b, st, ab)) {
                    std::fprintf(stderr, "tiled scorer refused stride %lld nq %d\n", (long long)stride, nq);
                    ok = false;
                }
                ck(cudaStreamSynchronize(st));
                std::vector<float> av((size_t)(nq * NB)), bv(av.size());
                ck(cudaMemcpy(av.data(), a, av.size() * 4, cudaMemcpyDeviceToHost));
                ck(cudaMemcpy(bv.data(), b, bv.size() * 4, cudaMemcpyDeviceToHost));
                for (int t = 0; t < nq; ++t) {
                    const int64_t nbid = steps[t * k::kStepCount + k::kStepNBid];
                    for (int64_t blk = 0; blk < NB; ++blk) {
                        const float x = av[(size_t)(t * NB + blk)], y = bv[(size_t)(t * NB + blk)];
                        if (blk > nbid) {
                            if (x != -999.f || y != -999.f) {
                                std::fprintf(stderr, "tiled wrote past n_bid stride %lld nq %d q %d b %lld\n",
                                             (long long)stride, nq, t, (long long)blk);
                                ok = false;
                            }
                            continue;
                        }
                        // bitwise identical (memcmp the float bits)
                        if (std::memcmp(&x, &y, 4) != 0) {
                            std::fprintf(stderr, "tiled score differs stride %lld nq %d q %d b %lld: %a vs %a\n",
                                         (long long)stride, nq, t, (long long)blk, x, y);
                            ok = false;
                            break;
                        }
                    }
                }
            }
            ck(cudaStreamDestroy(st));
        }
    }
    std::printf("tiled scoring bitwise vs legacy: %s\n", ok ? "PASS" : "FAIL");
    return ok;
}
} // namespace
int main() {
    bool ok = scoring();
    for (int stride : {32770, 65538})
        ok = selection(stride) && ok;
    ok = hierarchical() && ok;
    ok = tiled_scoring() && ok;
    return ok ? 0 : 1;
}
