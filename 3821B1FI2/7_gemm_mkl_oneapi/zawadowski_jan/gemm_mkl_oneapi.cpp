#include "gemm_mkl_oneapi.h"

#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(
    const std::vector<float> a,
    const std::vector<float> b,
    size_t size,
    sycl::device device)
{
    std::vector<float> c(size * size, 0.0f);
    sycl::queue queue(device);  
    float* a_dev = sycl::malloc_device<float>(size * size, queue);
    float* b_dev = sycl::malloc_device<float>(size * size, queue);
    float* c_dev = sycl::malloc_device<float>(size * size, queue);

    queue.memcpy(a_dev, a.data(), sizeof(float) * size * size).wait();
    queue.memcpy(b_dev, b.data(), sizeof(float) * size * size).wait();
    oneapi::mkl::blas::row_major::gemm(
        queue,
        oneapi::mkl::transpose::nontrans,
        oneapi::mkl::transpose::nontrans,
        size,
        size,
        size,
        1.0f,
        a_dev,
        size,
        b_dev,
        size,
        0.0f,
        c_dev,
        size
    ).wait();
    queue.memcpy(c.data(), c_dev, sizeof(float) * size * size).wait();
    sycl::free(a_dev, queue);
    sycl::free(b_dev, queue);
    sycl::free(c_dev, queue);

    return c;
}
