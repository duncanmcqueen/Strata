// src/kernels/sycl/rope.cpp - SYCL port of src/kernels/cuda/rope.cu (P2.S2: NEOX
// partial RoPE).  The host side (table builders, rope config) is the CUDA file's code
// verbatim - the parity test holds it bit-exact against a float64 transcription, so
// it must not drift.  Only the kernel and its launch are rewritten: one work-item per
// row, same as the CUDA one-thread-per-row choice (see the race note below, carried
// over from rope.cu).
//
// The mrope/rope-table state (mrope_table_set/get, rope_table_set/release/for) lives
// in native_rope.cu in the CUDA tree; the SYCL port keeps it in native_rope.cpp like
// the original, and this file links against it as rope.cu does.
#include "strata/kernels/rope.hpp"
#include "strata/kernels/mrope.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"

#include <cuda_runtime.h>

#include <cmath>
#include <cstdio>
#include <cstdlib>

namespace strata::kernels {

namespace {
// The process's rope config (rope_scaling.hpp).  One writer - the engine's startup thread, before
// session_init builds any table or captures any graph - and readers after it.
RopeScaling g_rope_scaling;
}  // namespace

void rope_scaling_set(const RopeScaling& scaling) { g_rope_scaling = scaling; }
const RopeScaling& rope_scaling() { return g_rope_scaling; }

void build_rope_table(int n_rot, double theta, int max_pos, float* cos_tab, float* sin_tab) {
    const int half = n_rot / 2;
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            // float64 throughout, in the reference's order: inv, then ang, then cos/sin
            const double inv = std::pow(theta, -2.0 * (double) i / (double) n_rot);
            const double ang = (double) p * inv;
            cos_tab[(size_t) p * half + i] = (float) std::cos(ang);
            sin_tab[(size_t) p * half + i] = (float) std::sin(ang);
        }
    }
}

void build_rope_table(int n_rot, const RopeScaling& sc, int max_pos, float* cos_tab, float* sin_tab) {
    if (sc.type == RopeScalingType::None) {
        build_rope_table(n_rot, sc.freq_base, max_pos, cos_tab, sin_tab);   // the original loop, verbatim
        return;
    }
    const int half = n_rot / 2;
    const double fs = sc.freq_scale();
    const double ms = sc.mscale();
    double cd[2];
    sc.corr_dims(n_rot, cd);
    const bool correct = sc.ext_factor != 0;   // ggml: the correction rides on ext_factor, not the type
    for (int p = 0; p < max_pos; ++p) {
        for (int i = 0; i < half; ++i) {
            const double inv = std::pow(sc.freq_base, -2.0 * (double) i / (double) n_rot);
            const double extrap = (double) p * inv;    // the trained angle, ggml's theta_extrap
            const double interp = fs * extrap;         // ggml's theta_interp
            double ang = interp;
            if (correct) {
                const double ramp = (double) rope_yarn_ramp((float) cd[0], (float) cd[1], i) * sc.ext_factor;
                ang = interp * (1.0 - ramp) + extrap * ramp;
            }
            cos_tab[(size_t) p * half + i] = (float) (std::cos(ang) * ms);
            sin_tab[(size_t) p * half + i] = (float) (std::sin(ang) * ms);
        }
    }
}

namespace {

// ONE WORK-ITEM PER ROW, not per (row, pair) - rope.cu's note on why, verbatim:
//
// ONE THREAD PER ROW, not per (row, pair).  The first version had one thread per pair and did the tail copy
// with threadIdx.x strided loops - so up to 32 threads wrote into the SAME row, with no barrier between the
// copy and the rotation, and a thread's copy of orow[i] could land AFTER another thread had rotated it.  A
// race that silently writes the unrotated value is worse than a wrong answer: it is intermittent.
//
// A row is head_dim floats (256 here) and there are batch x heads of them, so one thread per row is a
// handful of threads for a token and removes the ordering question entirely rather than synchronising it.
void rope_neox_row(long long r, const float* x, float* out, long long rows, int head_dim, int n_rot,
                   const float* cos_tab, const float* sin_tab, const int* pos, const int32_t* mtab) {
    if (r >= rows) return;
    const int half = n_rot / 2;
    const float* xr = x + r * head_dim;
    float* orow = out + r * head_dim;

    for (int d = n_rot; d < head_dim; ++d) orow[d] = xr[d];      // the PARTIAL rotation's untouched tail
    for (int i = 0; i < half; ++i) {
        const size_t toff = (size_t) mrope_pos(mtab, pos[r], i) * half;   // the image path's per-pair position
        // THE PAIRING AND THE SIGNS COME FROM `rope_neox_pair`, shared with the indexer's pooling kernel.
        rope_neox_pair(xr[i], xr[half + i], cos_tab[toff + i], sin_tab[toff + i], orow[i], orow[half + i]);
    }
}

}  // namespace

void rope_neox_apply(const float* x, float* out, int64_t rows, int head_dim, int n_rot, const float* cos_tab,
                     const float* sin_tab, const int* pos, void* stream) {
    if (rows <= 0 || n_rot <= 0) return;
    if (n_rot % 2 != 0 || n_rot > head_dim) {
        std::fprintf(stderr, "rope_neox_apply: n_rot %d must be even and <= head_dim %d\n", n_rot, head_dim);
        std::exit(1);
    }
    sycl::queue& q = sycl_runtime::queue_from_stream(stream);
    const int32_t* mtab = mrope_table();
    q.parallel_for(sycl::range<1>(static_cast<size_t>(rows)), [=](sycl::id<1> idx) {
        rope_neox_row(static_cast<long long>(idx[0]), x, out, rows, head_dim, n_rot, cos_tab, sin_tab, pos, mtab);
    });
    if (stream == nullptr) q.wait();  // the CUDA original's cudaDeviceSynchronize on the legacy default stream
}

}  // namespace strata::kernels
