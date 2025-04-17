#include "gemm_mkl_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
    const std::vector<float> a, const std::vector<float> b,
    size_t size, sycl::device device) {
    sycl::queue q(device);
    std::vector<float> c(size * size);
    {
        sycl::buffer<float> a_buf(a.data(), a.size());
        sycl::buffer<float> b_buf(b.data(), b.size());
        sycl::buffer<float> c_buf(c.data(), c.size());

        using oneapi::mkl::blas::row_major::gemm;
        using oneapi::mkl::transpose;

        gemm(q, transpose::nontrans, transpose::nontrans, size, size, size, 1,
            a_buf, size, b_buf, size, 0, c_buf, size);
    }
    return c;
}
