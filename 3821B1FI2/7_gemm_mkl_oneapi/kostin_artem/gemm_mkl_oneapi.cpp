#include "gemm_mkl_oneapi.h"

#include <oneapi/mkl.hpp>
#include <sycl/sycl.hpp>
#include <vector>

std::vector<float> GemmMklONEAPI(const std::vector<float> a,
                                 const std::vector<float> b,
                                 size_t n,
                                 sycl::device dev) {
    std::vector<float> result(n * n);

    sycl::queue q(dev);

    sycl::buffer<float> A(a.data(), sycl::range(a.size()));
    sycl::buffer<float> B(b.data(), sycl::range(b.size()));
    sycl::buffer<float> C(result.data(), sycl::range(result.size()));

    oneapi::mkl::blas::row_major::gemm(
        q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        n, n, n,
        1.0f,
        A, n,
        B, n,
        0.0f,
        C, n
    );

    return result;
}
