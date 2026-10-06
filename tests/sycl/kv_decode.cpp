// Independent FP64 decode oracle, plus changing-count graph replay.
#include "strata/kernels/f16_bits.hpp"
#include "strata/kernels/kv_q4.hpp"
#include "strata/kernels/kv_q8.hpp"
#include "strata/kernels/qsa_decode_attn.hpp"
#include "strata/kernels/qsa_prompt_attn.hpp"
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cuda_runtime.h>
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
        T *p = alloc<T>(v.size());
        ck(cudaMemcpy(p, v.data(), v.size() * sizeof(T), cudaMemcpyHostToDevice));
        return p;
    }
    ~Memory() {
        for (void *p : ptrs)
            ck(cudaFree(p));
    }
};
bool decode(int fmt) {
    constexpr int HD = 256, NH = 24, NK = 2, PS = 4, CTX = 160, CAP = 129, NQ = 3;
    const auto s = k::qsa_real_shapes();
    const int rows = CTX * NK;
    std::mt19937 rng(19 + fmt);
    std::uniform_real_distribution<float> rand(-1.0f, 1.0f);
    std::vector<uint16_t> kh(rows * HD), vh(kh.size()), ks(rows * 4), vs(ks.size());
    std::vector<int8_t> kq(kh.size()), vq(kh.size());
    std::vector<uint8_t> k4(rows * 144), v4(k4.size());
    for (size_t i = 0; i < kh.size(); ++i) {
        kh[i] = k::f16_from_f32(rand(rng));
        vh[i] = k::f16_from_f32(rand(rng));
        kq[i] = int8_t(int(rng() % 255) - 127);
        vq[i] = int8_t(int(rng() % 255) - 127);
    }
    for (size_t i = 0; i < ks.size(); ++i) {
        ks[i] = k::f16_from_f32(0.012f);
        vs[i] = k::f16_from_f32(0.008f);
    }
    for (auto *pool : {&k4, &v4})
        for (int b = 0; b < rows * 8; ++b) {
            uint16_t d = k::f16_from_f32((b & 1) ? -0.09f : 0.07f);
            std::memcpy(pool->data() + b * 18, &d, 2);
            for (int j = 0; j < 16; ++j)
                (*pool)[b * 18 + 2 + j] = uint8_t(rng());
        }
    std::vector<int32_t> table(CTX / PS), ids(NQ * CAP), steps(NQ * k::kStepCount);
    for (size_t i = 0; i < table.size(); ++i)
        table[i] = int32_t(table.size() - 1 - i);
    std::vector<float> q(NQ * NH * HD);
    for (auto &x : q)
        x = rand(rng) * 2;
    Memory mem;
    k::QsaAttnPools pools;
    if (fmt == 0) {
        pools.k_pool = mem.upload(kh);
        pools.v_pool = mem.upload(vh);
    }
    if (fmt == 1 || fmt == 3) {
        pools.k_q = mem.upload(kq);
        pools.k_scale = mem.upload(ks);
    }
    if (fmt == 1) {
        pools.v_q = mem.upload(vq);
        pools.v_scale = mem.upload(vs);
    }
    if (fmt == 2)
        pools.k_q4 = mem.upload(k4);
    if (fmt == 2 || fmt == 3)
        pools.v_q4 = mem.upload(v4);
    pools.page_table = mem.upload(table);
    auto dq = mem.upload(q);
    auto di = mem.upload(ids);
    auto ds = mem.upload(steps);
    const auto stride = k::qsa_decode_attn_scratch_floats(CAP, s);
    auto scratch = mem.alloc<float>(NQ * stride);
    auto out = mem.alloc<float>(q.size());
    auto single = mem.alloc<float>(q.size());
    auto prompt_out = mem.alloc<float>(q.size());
    auto prompt_direct = mem.alloc<float>(q.size());
    cudaStream_t st;
    ck(cudaStreamCreate(&st));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    k::qsa_decode_attn_batch(dq, pools, di, ds, CAP, s, scratch, out, NQ, st);
    if (!k::qsa_prompt_attn_batch(dq, pools, di, ds, CAP, s, prompt_out, NQ, st))
        std::exit(2);
    ck(cudaStreamEndCapture(st, &graph));
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    auto value = [&](bool v, int row, int d) -> double {
        if (fmt == 0)
            return k::f32_from_f16((v ? vh : kh)[row * HD + d]);
        if (fmt == 1 || (!v && fmt == 3))
            return double((v ? vq : kq)[row * HD + d]) * k::f32_from_f16((v ? vs : ks)[row * 4 + d / 64]);
        const auto &p = v ? v4 : k4;
        const auto *b = p.data() + row * 144 + d / 32 * 18;
        uint16_t scale;
        std::memcpy(&scale, b, 2);
        int j = d % 32;
        int code = j < 16 ? (b[2 + j] & 15) : (b[2 + j - 16] >> 4);
        return (code - 8) * double(k::f32_from_f16(scale));
    };
    bool ok = true;
    double max_error = 0, prompt_error = 0;
    for (int replay = 0; replay < 3; ++replay) {
        const int widths[3][NQ] = {{1, 65, 0}, {129, 64, 17}, {0, 129, 65}};
        for (int t = 0; t < NQ; ++t) {
            steps[t * k::kStepCount + k::kStepWidth] = widths[replay][t];
            for (int i = 0; i < CAP; ++i)
                ids[t * CAP + i] = (i + 7 * t + 3 * replay) % CTX;
        }
        // Include missing pages, an entirely masked first tile followed by valid cells,
        // and a nonempty selection with every page absent.
        auto current_table = table;
        for (int b = 0; b < CTX / PS; ++b)
            if (replay == 2 || b % 7 == 0 || (replay == 1 && b < 10))
                current_table[b] = -1;
        ck(cudaMemcpyAsync(const_cast<int32_t *>(pools.page_table), current_table.data(),
                           current_table.size() * 4, cudaMemcpyHostToDevice, st));
        ck(cudaMemcpyAsync(di, ids.data(), ids.size() * 4, cudaMemcpyHostToDevice, st));
        ck(cudaMemcpyAsync(ds, steps.data(), steps.size() * 4, cudaMemcpyHostToDevice, st));
        ck(cudaMemsetAsync(scratch, 0x7f, NQ * stride * 4, st));
        ck(cudaMemsetAsync(out, 0x7f, q.size() * 4, st));
        ck(cudaMemsetAsync(prompt_out, 0x7f, q.size() * 4, st));
        ck(cudaGraphLaunch(exec, st));
        ck(cudaStreamSynchronize(st));
        std::vector<float> actual(q.size());
        ck(cudaMemcpy(actual.data(), out, actual.size() * 4, cudaMemcpyDeviceToHost));
        std::vector<float> prompt(q.size());
        ck(cudaMemcpy(prompt.data(), prompt_out, prompt.size() * 4, cudaMemcpyDeviceToHost));
        for (int t = 0; t < NQ; ++t)
            for (int h = 0; h < NH; ++h) {
                int w = widths[replay][t], kvh = h / 12;
                std::vector<double> probs(w);
                std::vector<int> r(w, -1);
                double mx = -1e300, sum = 0;
                for (int i = 0; i < w; ++i) {
                    int cell = ids[t * CAP + i], page = current_table[cell / PS];
                    if (page < 0)
                        continue;
                    r[i] = (page * NK + kvh) * PS + cell % PS;
                    double dot = 0;
                    for (int d = 0; d < HD; ++d)
                        dot += double(q[(t * NH + h) * HD + d]) * value(false, r[i], d);
                    probs[i] = dot / 16.0;
                    mx = std::max(mx, probs[i]);
                }
                for (int i = 0; i < w; ++i) {
                    probs[i] = r[i] < 0 ? 0 : std::exp(probs[i] - mx);
                    sum += probs[i];
                }
                for (int d = 0; d < HD; ++d) {
                    double ref = 0;
                    for (int i = 0; i < w; ++i)
                        if (r[i] >= 0)
                            ref += probs[i] * value(true, r[i], d);
                    if (sum > 0)
                        ref /= sum;
                    float a = actual[(t * NH + h) * HD + d];
                    double error = std::fabs(a - ref);
                    max_error = std::max(max_error, error);
                    if (!std::isfinite(a) || error > 2e-5)
                        ok = false;
                    float pa = prompt[(t * NH + h) * HD + d];
                    double pe = std::fabs(pa - ref);
                    prompt_error = std::max(prompt_error, pe);
                    if (!std::isfinite(pa) || pe > 2e-5)
                        ok = false;
                }
            }
        // The single-query public entry point must agree with captured batched execution.
        for (int t = 0; t < NQ; ++t)
            k::qsa_decode_attn_step(dq + t * NH * HD, pools, di + t * CAP, ds + t * k::kStepCount, CAP, s,
                                    scratch, single + t * NH * HD, st);
        ck(cudaStreamSynchronize(st));
        std::vector<float> direct(q.size());
        ck(cudaMemcpy(direct.data(), single, direct.size() * 4, cudaMemcpyDeviceToHost));
        if (std::memcmp(actual.data(), direct.data(), actual.size() * 4) != 0)
            ok = false;
        if (!k::qsa_prompt_attn_batch(dq, pools, di, ds, CAP, s, prompt_direct, NQ, st))
            ok = false;
        ck(cudaStreamSynchronize(st));
        ck(cudaMemcpy(direct.data(), prompt_direct, direct.size() * 4, cudaMemcpyDeviceToHost));
        if (std::memcmp(prompt.data(), direct.data(), prompt.size() * 4))
            ok = false;
    }
    // Unsupported calls must return false and leave output untouched.
    ck(cudaMemsetAsync(prompt_direct, 0x5a, q.size() * 4, st));
    auto bad = s;
    bad.head_dim = 128;
    if (k::qsa_prompt_attn_batch(dq, pools, di, ds, CAP, bad, prompt_direct, NQ, st))
        ok = false;
    auto incomplete = pools;
    if (fmt == 0)
        incomplete.v_pool = nullptr;
    else if (fmt == 2)
        incomplete.v_q4 = nullptr;
    else
        incomplete.k_scale = nullptr;
    if (k::qsa_prompt_attn_batch(dq, incomplete, di, ds, CAP, s, prompt_direct, NQ, st))
        ok = false;
    if (!k::qsa_prompt_attn_batch(nullptr, {}, nullptr, nullptr, 0, bad, nullptr, 0, st))
        ok = false;
    ck(cudaStreamSynchronize(st));
    std::vector<uint32_t> sentinel(q.size());
    ck(cudaMemcpy(sentinel.data(), prompt_direct, sentinel.size() * 4, cudaMemcpyDeviceToHost));
    if (!std::all_of(sentinel.begin(), sentinel.end(), [](uint32_t x) { return x == 0x5a5a5a5a; }))
        ok = false;
    std::printf("prompt format %d: FP64 max error %.3g, changing graph/refusal %s\n", fmt, prompt_error,
                ok ? "PASS" : "FAIL");
    ck(cudaGraphExecDestroy(exec));
    ck(cudaGraphDestroy(graph));
    ck(cudaStreamDestroy(st));
    std::printf("decode format %d: FP64 max error %.3g, graph/single %s\n", fmt, max_error,
                ok ? "PASS" : "FAIL");
    return ok;
}
// Changing selections must resolve and copy through the same captured graph.
bool streaming(int fmt) {
    constexpr int NB = 1300, NS = 1024, CAP = 2048;
    auto s = k::qsa_real_shapes();
    const int rows = int(s.n_head_kv * s.page_size);
    Memory mem;
    k::KvStreamMap m;
    m.n_blocks = NB;
    m.n_slots = NS;
    m.page_table = mem.alloc<int32_t>(NB);
    m.slot_block = mem.alloc<int32_t>(NS);
    m.slot_stamp = mem.alloc<int32_t>(NS);
    m.slot_ref = mem.alloc<int32_t>(NS);
    m.miss_block = mem.alloc<int32_t>(NS);
    m.miss_slot = mem.alloc<int32_t>(NS);
    m.ctl = mem.alloc<int32_t>(k::kKvCtlInts);
    k::QsaAttnPools slots, stage;
    k::KvHostPools host;
    slots.page_table = m.page_table;
    struct Run {
        uint8_t *host;
        uint8_t *slots;
        uint8_t *stage;
        int bytes;
    };
    std::vector<Run> runs;
    auto add = [&](int bytes) {
        uint8_t *h = nullptr;
        ck(cudaHostAlloc(reinterpret_cast<void **>(&h), NB * bytes, cudaHostAllocMapped));
        for (int b = 0; b < NB; ++b)
            for (int j = 0; j < bytes; ++j)
                h[b * bytes + j] = uint8_t((b * 13 + j * 7 + (b >> 8)) % 251);
        runs.push_back({h, mem.alloc<uint8_t>(NS * bytes), mem.alloc<uint8_t>(NB * bytes), bytes});
    };
    if (fmt == k::kKvQ4) {
        add(rows * 144);
        add(rows * 144);
        host.k_q4 = runs[0].host;
        host.v_q4 = runs[1].host;
        slots.k_q4 = runs[0].slots;
        slots.v_q4 = runs[1].slots;
        stage.k_q4 = runs[0].stage;
        stage.v_q4 = runs[1].stage;
    } else if (fmt == k::kKvInt8) {
        add(rows * 256);
        add(rows * 256);
        add(rows * 8);
        add(rows * 8);
        host.k_q = reinterpret_cast<int8_t *>(runs[0].host);
        host.v_q = reinterpret_cast<int8_t *>(runs[1].host);
        host.k_scale = reinterpret_cast<uint16_t *>(runs[2].host);
        host.v_scale = reinterpret_cast<uint16_t *>(runs[3].host);
        slots.k_q = reinterpret_cast<int8_t *>(runs[0].slots);
        slots.v_q = reinterpret_cast<int8_t *>(runs[1].slots);
        slots.k_scale = reinterpret_cast<uint16_t *>(runs[2].slots);
        slots.v_scale = reinterpret_cast<uint16_t *>(runs[3].slots);
        stage.k_q = reinterpret_cast<int8_t *>(runs[0].stage);
        stage.v_q = reinterpret_cast<int8_t *>(runs[1].stage);
        stage.k_scale = reinterpret_cast<uint16_t *>(runs[2].stage);
        stage.v_scale = reinterpret_cast<uint16_t *>(runs[3].stage);
    } else {
        add(rows * 512);
        add(rows * 512);
        host.k_pool = reinterpret_cast<uint16_t *>(runs[0].host);
        host.v_pool = reinterpret_cast<uint16_t *>(runs[1].host);
        slots.k_pool = reinterpret_cast<uint16_t *>(runs[0].slots);
        slots.v_pool = reinterpret_cast<uint16_t *>(runs[1].slots);
        stage.k_pool = reinterpret_cast<uint16_t *>(runs[0].stage);
        stage.v_pool = reinterpret_cast<uint16_t *>(runs[1].stage);
    }
    auto ids = mem.alloc<int32_t>(CAP);
    auto steps = mem.alloc<int32_t>(k::kStepCount);
    cudaStream_t st;
    ck(cudaStreamCreate(&st));
    k::kv_stream_reset(m, st);
    ck(cudaStreamSynchronize(st));
    cudaGraph_t graph;
    cudaGraphExec_t exec;
    ck(cudaStreamBeginCapture(st, cudaStreamCaptureModeGlobal));
    k::kv_stream_resolve(m, slots, host, fmt, ids, steps, 1, CAP, s, st);
    ck(cudaStreamEndCapture(st, &graph));
    ck(cudaGraphInstantiate(&exec, graph, nullptr, nullptr, 0));
    bool ok = true;
    const int first[] = {0, 0, 512, 788, -1, 0};
    for (int start : first) {
        std::vector<int32_t> hids(CAP), hsteps(k::kStepCount, 0);
        if (start >= 0) {
            for (int i = 0; i < CAP; ++i)
                hids[i] = start * 4 + i;
            hsteps[k::kStepWidth] = CAP;
        }
        ck(cudaMemcpyAsync(ids, hids.data(), CAP * 4, cudaMemcpyHostToDevice, st));
        ck(cudaMemcpyAsync(steps, hsteps.data(), hsteps.size() * 4, cudaMemcpyHostToDevice, st));
        ck(cudaGraphLaunch(exec, st));
        ck(cudaStreamSynchronize(st));
        std::vector<int32_t> table(NB), blocks(NS);
        ck(cudaMemcpy(table.data(), m.page_table, NB * 4, cudaMemcpyDeviceToHost));
        ck(cudaMemcpy(blocks.data(), m.slot_block, NS * 4, cudaMemcpyDeviceToHost));
        for (int b = 0; b < NB; ++b)
            if (table[b] < -1 || table[b] >= NS || (table[b] >= 0 && blocks[table[b]] != b))
                ok = false;
        for (int sl = 0; sl < NS; ++sl)
            if (blocks[sl] >= 0 && (blocks[sl] >= NB || table[blocks[sl]] != sl))
                ok = false;
        for (const auto &r : runs) {
            std::vector<uint8_t> actual(NS * r.bytes);
            ck(cudaMemcpy(actual.data(), r.slots, actual.size(), cudaMemcpyDeviceToHost));
            if (start >= 0)
                for (int b = start; b < start + 512; ++b) {
                    if (table[b] < 0)
                        ok = false;
                    else if (std::memcmp(actual.data() + table[b] * r.bytes, r.host + b * r.bytes, r.bytes))
                        ok = false;
                }
        }
    }
    auto c = k::kv_stream_counters(m);
    if (c.calls != 6 || c.lookups != 2560 || c.misses <= 1024 || c.overflow)
        ok = false;
    k::kv_stage_from_host(stage, host, fmt, NB, s, st);
    ck(cudaStreamSynchronize(st));
    for (const auto &r : runs) {
        std::vector<uint8_t> actual(NB * r.bytes);
        ck(cudaMemcpy(actual.data(), r.stage, actual.size(), cudaMemcpyDeviceToHost));
        if (std::memcmp(actual.data(), r.host, actual.size()))
            ok = false;
    }
    ck(cudaGraphExecDestroy(exec));
    ck(cudaGraphDestroy(graph));
    ck(cudaStreamDestroy(st));
    for (const auto &r : runs)
        ck(cudaFreeHost(r.host));
    std::printf("stream format %d: graph eviction/empty/hits and staging %s\n", fmt, ok ? "PASS" : "FAIL");
    return ok;
}
} // namespace
int main() {
    bool ok = true;
    for (int fmt = 0; fmt < 4; ++fmt)
        ok = decode(fmt) && ok;
    for (int fmt : {k::kKvF16, k::kKvInt8, k::kKvQ4})
        ok = streaming(fmt) && ok;
    return ok ? 0 : 1;
}
