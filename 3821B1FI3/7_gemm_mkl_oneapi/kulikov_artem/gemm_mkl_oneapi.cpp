#include "gemm_mkl_oneapi.h"

#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
    const std::vector<float> a, const std::vector<float> b,
    size_t size, sycl::device device) {
    sycl::queue q(device);

    float *a_dev = sycl::malloc_device<float>(a.size(), q);
    float *b_dev = sycl::malloc_device<float>(b.size(), q);
    float *c_dev = sycl::malloc_device<float>(size * size, q);

    q.memcpy(a_dev, a.data(), a.size() * sizeof(float)).wait();
    q.memcpy(b_dev, b.data(), b.size() * sizeof(float)).wait();

    oneapi::mkl::blas::row_major::gemm(
        q,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        size, size, size,
        1.0f,
        a_dev, size,
        b_dev, size,
        0.0f,
        c_dev, size)
        .wait();

    std::vector<float> c(size * size);
    q.memcpy(c.data(), c_dev, c.size() * sizeof(float)).wait();

    sycl::free(a_dev, q);
    sycl::free(b_dev, q);
    sycl::free(c_dev, q);

    return c;
}
