#include "gemm_mkl_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
    const std::vector<float> a, const std::vector<float> b,
    size_t size, sycl::device device) {
    sycl::queue q(device, sycl::property::queue::in_order());
    std::vector<float> result(size * size);

    sycl::buffer<float, 1> matrix_a(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float, 1> matrix_b(b.data(), sycl::range<1>(b.size()));
    sycl::buffer<float, 1> matrix_c(result.data(), sycl::range<1>(result.size()));

    try {
        oneapi::mkl::blas::row_major::gemm(
            q,
            oneapi::mkl::transpose::nontrans,
            oneapi::mkl::transpose::nontrans,
            size,
            size,
            size,
            1.0f,
            matrix_a,
            size,
            matrix_b,
            size,
            0.0f,
            matrix_c,
            size
        );

        q.wait_and_throw();

    }
    catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        throw;
    }

    return result;
}