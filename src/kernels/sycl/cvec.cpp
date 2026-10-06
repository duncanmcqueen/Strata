// src/kernels/sycl/cvec.cpp - SYCL port of src/kernels/cuda/cvec.cu; see include/strata/kernels/cvec.hpp.
//
// The pending FFN write must be bitwise the one the fused hyper-connection read folds, so it is written with
// fused_gr.cpp's own SYCL expressions (`sigmoidf_` with `sycl::exp`, `sycl::fma(b, w, r)`), not CUDA's `__expf`.
// The block reduction keeps its order on 32-wide sub-groups; the per-device tables are the CUDA file's.
#include "strata/kernels/cvec.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"

#include <cuda_runtime.h>

#include <stdexcept>

namespace strata::kernels {
namespace {

constexpr int THREADS = 256;
constexpr int MAXK = 16;   // n_embd up to 4096, held in registers between the dot and the update

Cvec g_cvec;
bool g_on_host = false;
constexpr int kDevices = 64;
struct DevTables { float* dir = nullptr; float* s = nullptr; int* on = nullptr; };
DevTables g_dev[kDevices];
std::vector<float> g_dir_host, g_s_host;
int cur_device() {
    int d = 0;
    if (cudaGetDevice(&d) != cudaSuccess || d < 0 || d >= kDevices) d = 0;
    return d;
}
bool upload_here(std::string& err) {
    DevTables& t = g_dev[cur_device()];
    if (t.dir != nullptr) return true;
    const int flag = g_on_host ? 1 : 0;
    if (cudaMalloc(&t.dir, g_dir_host.size() * sizeof(float)) != cudaSuccess ||
        cudaMalloc(&t.s, g_s_host.size() * sizeof(float)) != cudaSuccess || cudaMalloc(&t.on, sizeof(int)) != cudaSuccess ||
        cudaMemcpy(t.dir, g_dir_host.data(), g_dir_host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(t.s, g_s_host.data(), g_s_host.size() * sizeof(float), cudaMemcpyHostToDevice) != cudaSuccess ||
        cudaMemcpy(t.on, &flag, sizeof(int), cudaMemcpyHostToDevice) != cudaSuccess) {
        err = "control vector: device allocation failed";
        t = DevTables{};
        return false;
    }
    return true;
}

// fused_gr.cpp's gate, verbatim
inline float sigmoidf_(float x) { return 1.0f / (1.0f + sycl::exp(-x)); }

}  // namespace

const Cvec& cvec() { return g_cvec; }

bool cvec_upload(const std::vector<float>& dir, const std::vector<float>& s, int mode, int first, int last,
                 int64_t n_embd, int64_t hc, std::string& err) {
    if (n_embd < 1 || n_embd > (int64_t) THREADS * MAXK) { err = "control vector: unsupported n_embd"; return false; }
    if (s.empty() || dir.size() != s.size() * (size_t) n_embd) { err = "control vector: bad table sizes"; return false; }
    int prev = 0;
    cudaGetDevice(&prev);
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].dir == nullptr) continue;
        cudaSetDevice(d);
        cudaDeviceSynchronize();
        cudaFree(g_dev[d].dir);
        cudaFree(g_dev[d].s);
        cudaFree(g_dev[d].on);
        g_dev[d] = DevTables{};
    }
    cudaSetDevice(prev);
    g_dir_host = dir;
    g_s_host = s;
    g_on_host = true;
    if (!upload_here(err)) return false;
    const DevTables& t = g_dev[cur_device()];
    g_cvec.dir = t.dir;
    g_cvec.s = t.s;
    g_cvec.on = t.on;
    g_cvec.mode = mode;
    g_cvec.first = first;
    g_cvec.last = last;
    g_cvec.n_embd = n_embd;
    g_cvec.hc = hc;
    g_cvec.steered.assign(s.size(), false);
    for (size_t l = 0; l < s.size(); ++l) g_cvec.steered[l] = s[l] != 0.0f;
    return true;
}

bool cvec_replicate(std::string& err) { return !g_cvec.loaded() || upload_here(err); }

void cvec_set_enabled(bool on) {
    if (!g_cvec.loaded() || on == g_on_host) return;
    int prev = 0;
    cudaGetDevice(&prev);
    const int v = on ? 1 : 0;
    for (int d = 0; d < kDevices; ++d) {
        if (g_dev[d].on == nullptr) continue;
        cudaSetDevice(d);
        cudaDeviceSynchronize();   // nothing in flight may still read the flag
        cudaMemcpy(g_dev[d].on, &v, sizeof(int), cudaMemcpyHostToDevice);
    }
    cudaSetDevice(prev);
    g_on_host = on;
}

bool cvec_enabled() { return g_cvec.loaded() && g_on_host; }

void cvec_apply(float* R, int64_t layer, int64_t T, int64_t r_ld, const float* bo, int64_t bo_ld, const float* inj,
                int64_t inj_ld, bool write, void* stream) {
    if (!g_cvec.loaded() || T < 1) return;
    const DevTables& tb = g_dev[cur_device()];
    if (tb.dir == nullptr) throw std::runtime_error("cvec_apply: the control vector is not on this device (cvec_replicate)");
    const float* dir = tb.dir;
    const float* s_l = tb.s;
    const int* on = tb.on;
    const int mode = g_cvec.mode, n = (int) g_cvec.n_embd, hc = (int) g_cvec.hc;
    const int wr = write ? 1 : 0;
    try {
        // one work-group per (stream, token), flattened stream-fastest: the pending write, h . v, the update
        sycl_runtime::queue_from_stream(stream).submit([&](sycl::handler& h) {
            sycl::local_accessor<float, 1> part(sycl::range<1>(THREADS / 32), h);
            h.parallel_for(sycl::nd_range<1>((size_t) hc * T * THREADS, THREADS),
                           [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(32)]] {
                const int tid = (int) it.get_local_id(0);
                const int c = (int) (it.get_group(0) % hc);
                const int64_t t = (int64_t) (it.get_group(0) / hc);
                float* r = R + t * r_ld + (int64_t) c * n;
                const float s = s_l[layer];
                const bool steer = *on != 0 && s != 0.0f;   // uniform over the work-group
                if (!steer && !wr) return;
                const float* v = dir + layer * n;
                const float w = wr ? 2.0f * sigmoidf_(inj[t * inj_ld + c] / (float) hc) : 0.0f;
                const float* b = wr ? bo + t * bo_ld : nullptr;
                float x[MAXK];
                float dot = 0.0f;
#pragma unroll
                for (int k = 0; k < MAXK; ++k) {
                    const int d = tid + k * THREADS;
                    if (d < n) {
                        float xv = r[d];
                        if (wr) xv = sycl::fma(b[d], w, xv);
                        x[k] = xv;
                        if (steer && mode == 0) dot = sycl::fma(xv, v[d], dot);
                    }
                }
                if (steer && mode == 0) {
                    const sycl::sub_group sg = it.get_sub_group();
#pragma unroll
                    for (int o = 16; o > 0; o >>= 1) dot += sycl::permute_group_by_xor(sg, dot, o);
                    if ((tid & 31) == 0) part[tid >> 5] = dot;
                    sycl::group_barrier(it.get_group());
                    if (tid < 32) {
                        float p = tid < THREADS / 32 ? part[tid] : 0.0f;
#pragma unroll
                        for (int o = 16; o > 0; o >>= 1) p += sycl::permute_group_by_xor(sg, p, o);
                        if (tid == 0) part[0] = p;
                    }
                    sycl::group_barrier(it.get_group());
                    dot = part[0] * s;   // s (h . v)
                }
#pragma unroll
                for (int k = 0; k < MAXK; ++k) {
                    const int d = tid + k * THREADS;
                    if (d < n) {
                        float xv = x[k];
                        if (steer) xv = mode == 0 ? sycl::fma(-dot, v[d], xv) : xv + v[d];
                        r[d] = xv;
                    }
                }
            });
        });
    } catch (const sycl::exception&) {
        throw std::runtime_error("cvec_apply: launch failed");
    }
}

}  // namespace strata::kernels
