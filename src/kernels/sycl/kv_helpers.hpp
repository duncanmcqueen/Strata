#pragma once
// Local implementation helpers for the KV and decode-attention ports.
#include "strata/kernels/f16_bits.hpp"
#include "strata/sycl_runtime/queue_bridge.hpp"
namespace strata::kernels::sycl_kv {
struct Grid {
    size_t x, y, z;
    Grid(size_t a, size_t b = 1, size_t c = 1) : x(a), y(b), z(c) {}
    size_t size() const { return x * y * z; }
};
inline int group_x(sycl::nd_item<1> it, Grid g) { return int(it.get_group(0) % g.x); }
inline int group_y(sycl::nd_item<1> it, Grid g) { return int((it.get_group(0) / g.x) % g.y); }
inline int group_z(sycl::nd_item<1> it, Grid g) { return int(it.get_group(0) / (g.x * g.y)); }
inline int local_x(sycl::nd_item<1> it, Grid b) { return int(it.get_local_id(0) % b.x); }
inline int local_y(sycl::nd_item<1> it, Grid b) { return int((it.get_local_id(0) / b.x) % b.y); }
struct alignas(16) uint4 {
    uint32_t x, y, z, w;
};
struct alignas(8) uint2 {
    uint32_t x, y;
};
struct alignas(16) float4 {
    float x, y, z, w;
};
struct alignas(4) char4 {
    int8_t x, y, z, w;
};
struct alignas(8) ushort4 {
    uint16_t x, y, z, w;
};
template <class T> T xor_lane(sycl::nd_item<1> it, T v, int mask) {
    return sycl::permute_group_by_xor(it.get_sub_group(), v, mask);
}
template <class T> T down_lane(sycl::nd_item<1> it, T v, int off) {
    return sycl::shift_group_left(it.get_sub_group(), v, off);
}
template <class T> T up_lane(sycl::nd_item<1> it, T v, int off) {
    return sycl::shift_group_right(it.get_sub_group(), v, off);
}
inline int atomic_add(int *p, int v) {
    return sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                            sycl::access::address_space::generic_space>(*p)
        .fetch_add(v);
}
inline int atomic_cas(int *p, int expected, int desired) {
    sycl::atomic_ref<int, sycl::memory_order::relaxed, sycl::memory_scope::device,
                     sycl::access::address_space::generic_space>(*p)
        .compare_exchange_strong(expected, desired);
    return expected;
}
} // namespace strata::kernels::sycl_kv
