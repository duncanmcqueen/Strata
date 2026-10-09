// scratch: find the legal INT8 joint_matrix tile on bmg-g31
#include <sycl/sycl.hpp>
#include <sycl/ext/oneapi/matrix/matrix.hpp>
#include <cstdio>
namespace mat = sycl::ext::oneapi::experimental::matrix;

int main() {
    sycl::queue q(sycl::gpu_selector_v);
    constexpr int M = Mv, N = Nv, K = Kv;
    int8_t* a = sycl::malloc_device<int8_t>(M * K, q);
    int8_t* b = sycl::malloc_device<int8_t>(K * N, q);
    int32_t* c = sycl::malloc_device<int32_t>(M * N, q);
    q.parallel_for(sycl::nd_range<1>(16, 16), [=](sycl::nd_item<1> it) [[sycl::reqd_sub_group_size(16)]] {
        auto sg = it.get_sub_group();
        mat::joint_matrix<sycl::sub_group, int8_t, mat::use::a, M, K, mat::layout::row_major> tA;
        mat::joint_matrix<sycl::sub_group, int8_t, mat::use::b, K, N, mat::layout::row_major> tB;
        mat::joint_matrix<sycl::sub_group, int32_t, mat::use::accumulator, M, N> tC;
        mat::joint_matrix_load(sg, tA, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(a), K);
        mat::joint_matrix_load(sg, tB, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(b), N);
        mat::joint_matrix_fill(sg, tC, 0);
        mat::joint_matrix_mad(sg, tC, tA, tB, tC);
        mat::joint_matrix_store(sg, tC, sycl::address_space_cast<sycl::access::address_space::global_space, sycl::access::decorated::no>(c), N, mat::layout::row_major);
    }).wait();
    std::printf("i8 M=%d N=%d K=%d compiled and ran\n", M, N, K);
    return 0;
}
