#pragma once
// src/kernels/sycl/fp_exact.hpp - private to the SYCL kernels: single-rounded float arithmetic without the device
// library's software routines.
//
// The kernels pin rounding where results are compared bitwise (one product, one sum, each rounded on its own; an FMA
// only where one is written).  sycl::ext::intel::math::fmul_rn / fadd_rn give that, but on the A770 they are
// software: a loop of fadd_rn(s, fmul_rn(a, b)) ran 16x slower than the same loop written with * and + (2.15 vs
// 0.13 ms), and native MMVQ built on them reached 64-98 GB/s of the card's ~560.  The SYCL kernel libraries compile
// with -ffp-contract=off (CMakeLists.txt), so a product and a sum written as separate operations are separate IEEE
// round-to-nearest operations, and IGC does not fuse them (measured: 65,536 of 65,536 a*b+c equal to the separately
// rounded value, 52,535 to the FMA).  Division and square root keep the device library: IGC's own are not correctly
// rounded (docs/INTEL_SYCL.md).
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/math.hpp>

namespace strata::kernels::fp_exact {
inline float fmul_rn(float a, float b) { return a * b; }
inline float fadd_rn(float a, float b) { return a + b; }
inline float fsub_rn(float a, float b) { return a - b; }
inline float fmaf_rn(float a, float b, float c) { return sycl::fma(a, b, c); }
using sycl::ext::intel::math::fdiv_rn;
using sycl::ext::intel::math::fsqrt_rn;
}  // namespace strata::kernels::fp_exact
