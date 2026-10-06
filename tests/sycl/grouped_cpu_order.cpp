// Reuse the established packed-weight fixtures and double dot-product oracle.
// The original parity source remains unchanged and retains its own target.
#define main grouped_original_parity_main
#include "../../src/kernels/s2_expert_grouped_parity.cpp"
#undef main

namespace {
// Scalar transcription of the CPU AVX reduction: eight FMA lanes, low+high,
// adjacent horizontal pairs, then the independently accumulated bias correction.
float cpu_projection(const uint8_t *codes, const uint8_t *scales, const std::vector<uint8_t> &x,
                     const std::vector<float> &xs) {
    float lanes[8] = {}, correction = 0.0f;
    for (int b = 0; b < NCH_GU / 2; ++b) {
        const float dw = f16f(scales + 2 * b);
        float hx[2];
        for (int half = 0; half < 2; ++half) {
            const int chunk = 2 * b + half;
            const uint8_t *q = x.data() + size_t(chunk) * 34 + 2;
            int total = 0;
            for (int j = 0; j < 32; ++j)
                total += int(int8_t(q[j]));
            hx[half] = xs[chunk] * float(total);
            const float factor = dw * xs[chunk];
            for (int lane = 0; lane < 8; ++lane) {
                int dot = 0;
                const unsigned packed = codes[b * 16 + half * 8 + lane];
                for (int j = 0; j < 4; ++j)
                    dot += int((packed >> (2 * j)) & 3) * int(int8_t(q[lane * 4 + j]));
                lanes[lane] = std::fma(factor, float(dot), lanes[lane]);
            }
        }
        correction += dw * (hx[0] + hx[1]);
    }
    float pairs[4];
    for (int i = 0; i < 4; ++i)
        pairs[i] = lanes[i] + lanes[i + 4];
    return ((pairs[0] + pairs[1]) + (pairs[2] + pairs[3])) - correction;
}
void check_routing() {
    constexpr int n = 128, experts = 8;
    auto *ids = dalloc<int32_t>(n);
    auto *rows = dalloc<int32_t>(experts);
    auto *slots = dalloc<int32_t>(n);
    auto *dst = dalloc<int32_t>(n);
    auto *count = dalloc<int32_t>(1);
    cudaStream_t stream = nullptr;
    ck(cudaStreamCreate(&stream), "routing stream");
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "routing capture");
    k::moe_hit_select_multi(ids, rows, n, experts, slots, dst, count, stream);
    ck(cudaStreamEndCapture(stream, &graph), "routing end capture");
    ck(cudaGraphInstantiate(&executable, graph, 0ull), "routing instantiate");
    for (int pattern = 0; pattern < 3; ++pattern) {
        std::vector<int32_t> input(n), resident(experts);
        for (int i = 0; i < n; ++i)
            input[i] = pattern == 2 ? experts + 1 : (i % (experts + 2)) - 1;
        input[31] = input[127] = pattern == 2 ? -1 : experts - 1;
        for (int i = 0; i < experts; ++i)
            resident[i] = pattern == 1 || i % 2 ? i + 17 : -1;
        up(ids, input);
        up(rows, resident);
        auto verify = [&](int width) {
            std::vector<int32_t> want_slots, want_dst;
            for (int i = 0; i < width; ++i)
                if (input[i] >= 0 && input[i] < experts && resident[input[i]] >= 0) {
                    want_slots.push_back(resident[input[i]]);
                    want_dst.push_back(i);
                }
            const int got = down(count, 1)[0];
            if (got != int(want_slots.size()) || down(slots, want_slots.size()) != want_slots ||
                down(dst, want_dst.size()) != want_dst)
                ++g_fail;
        };
        ck(cudaGraphLaunch(executable, stream), "routing replay");
        ck(cudaStreamSynchronize(stream), "routing replay sync");
        verify(n);
        k::moe_hit_select(ids, rows, 32, experts, slots, dst, count, stream);
        ck(cudaStreamSynchronize(stream), "single-warp routing sync");
        verify(32);
    }
    ck(cudaGraphExecDestroy(executable), "routing executable destroy");
    ck(cudaGraphDestroy(graph), "routing graph destroy");
    ck(cudaStreamDestroy(stream), "routing stream destroy");
    for (void *p : {static_cast<void *>(ids), static_cast<void *>(rows), static_cast<void *>(slots),
                    static_cast<void *>(dst), static_cast<void *>(count)})
        ck(cudaFree(p), "routing free");
}

} // namespace

int main() {
    check_routing();
    std::mt19937 rng(20261004);
    const Fixture fx = make_blobs(4, rng);
    constexpr int hits = 3, routed = 5;
    std::vector<uint8_t> x;
    std::vector<float> xs;
    fill_x(x, xs, 1, rng);
    auto *dx = dalloc<uint8_t>(x.size());
    auto *dxs = dalloc<float>(xs.size());
    auto *slots = dalloc<int32_t>(hits);
    auto *destinations = dalloc<int32_t>(hits);
    auto *trace = dalloc<float>(hits * 2 * FF);
    auto *output = dalloc<float>(routed * H);
    const Scratch scratch = make_scratch(hits);
    cudaStream_t stream = nullptr;
    ck(cudaStreamCreate(&stream), "CPU-order stream");
    auto launch = [&] {
        k::moe_hit_grouped_s2_cpu_order(fx.d, slots, destinations, hits, BLOB, dx, scratch.p, output, stream,
                                        dxs, trace);
    };
    cudaGraph_t graph = nullptr;
    cudaGraphExec_t executable = nullptr;
    ck(cudaStreamBeginCapture(stream, cudaStreamCaptureModeThreadLocal), "CPU-order capture");
    launch();
    ck(cudaStreamEndCapture(stream, &graph), "CPU-order end capture");
    ck(cudaGraphInstantiate(&executable, graph, 0ull), "CPU-order instantiate");
    for (int replay = 0; replay < 2; ++replay) {
        if (replay)
            fill_x(x, xs, 1, rng);
        const std::vector<int32_t> host_slots =
            replay ? std::vector<int32_t>{3, 0, 2} : std::vector<int32_t>{1, 2, 0};
        const std::vector<int32_t> host_dst =
            replay ? std::vector<int32_t>{1, 3, 0} : std::vector<int32_t>{4, 0, 2};
        up(dx, x);
        up(dxs, xs);
        up(slots, host_slots);
        up(destinations, host_dst);
        ck(cudaMemset(output, 0xa5, routed * H * 4), "CPU-order output poison");
        ck(cudaMemset(scratch.p, 0x5a, scratch.bytes), "CPU-order scratch poison");
        ck(cudaGraphLaunch(executable, stream), "CPU-order replay");
        ck(cudaStreamSynchronize(stream), "CPU-order replay sync");
        Run run{down(output, routed * H), down(static_cast<const uint8_t *>(scratch.p), scratch.bytes)};
        const auto captured_trace = down(trace, hits * 2 * FF);
        size_t bad_trace = 0;
        std::vector<Entry> entries;
        for (int h = 0; h < hits; ++h) {
            const uint8_t *blob = fx.hb(host_slots[h]);
            entries.push_back({blob, 0, host_dst[h]});
            for (int row = 0; row < 2 * FF; ++row) {
                const float expected =
                    cpu_projection(blob + row * ROW_GU, blob + O_GU_SCALES + row * SC_GU * 2, x, xs);
                const size_t at = ((row & 1) ? hits * FF : 0) + h * FF + (row >> 1);
                bad_trace += std::memcmp(&expected, &captured_trace[at], sizeof(float)) != 0;
            }
        }
        std::printf("CPU-order replay %d: %zu gate/up words differ from scalar CPU order\n", replay,
                    bad_trace);
        if (bad_trace)
            ++g_fail;
        reference("CPU-order final projections", run, scratch, entries, x, &xs);
        ck(cudaMemset(output, 0xa5, routed * H * 4), "CPU-order direct output poison");
        ck(cudaMemset(scratch.p, 0x5a, scratch.bytes), "CPU-order direct scratch poison");
        launch();
        ck(cudaStreamSynchronize(stream), "CPU-order direct sync");
        const auto direct = down(output, routed * H);
        const auto direct_trace = down(trace, hits * 2 * FF);
        if (std::memcmp(direct.data(), run.out.data(), direct.size() * sizeof(float)) ||
            std::memcmp(direct_trace.data(), captured_trace.data(), direct_trace.size() * sizeof(float)))
            ++g_fail;
    }
    ck(cudaGraphExecDestroy(executable), "CPU-order executable destroy");
    ck(cudaGraphDestroy(graph), "CPU-order graph destroy");
    ck(cudaStreamDestroy(stream), "CPU-order stream destroy");
    for (void *p : {static_cast<void *>(fx.d), static_cast<void *>(dx), static_cast<void *>(dxs),
                    static_cast<void *>(slots), static_cast<void *>(destinations), static_cast<void *>(trace),
                    static_cast<void *>(output), scratch.p})
        ck(cudaFree(p), "CPU-order free");
    std::printf("grouped_cpu_order: %d failures\n", g_fail);
    return g_fail ? 1 : 0;
}
