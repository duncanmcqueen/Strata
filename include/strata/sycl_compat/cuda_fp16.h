#pragma once
// The half-precision surface the SYCL port needs, on top of DPC++'s sycl::half.
// Included explicitly by the sources that use it (it is NOT force-included).
#include <sycl/half_type.hpp>

using __half = sycl::half;
using __half_raw = sycl::detail::half_impl::StorageT;

struct __half2 {
    sycl::half x, y;
};

inline __half __float2half(float f) { return __half(f); }
inline float __half2float(__half h) { return static_cast<float>(h); }
