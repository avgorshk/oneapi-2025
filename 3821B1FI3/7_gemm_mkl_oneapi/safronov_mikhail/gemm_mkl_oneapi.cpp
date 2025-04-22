#include "gemm_mkl_oneapi.h"
#include <oneapi/mkl.hpp>
#include <cassert>

std::vector<float> GemmMKLONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    size_t size, sycl::device device) {
    sycl::queue queue(device);
    std::vector<float> c(size * size, 0.0f);

    float* dev_a = sycl::malloc_device<float>(size * size, queue);
    float* dev_b = sycl::malloc_device<float>(size * size, queue);
    float* dev_c = sycl::malloc_device<float>(size * size, queue);

    queue.memcpy(dev_a, a.data(), size * size * sizeof(float)).wait();
    queue.memcpy(dev_b, b.data(), size * size * sizeof(float)).wait();

    oneapi::mkl::blas::row_major::gemm(
        queue, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans,
        size, size, size, 1.0f, dev_a, size, dev_b, size, 0.0f, dev_c, size);

    queue.memcpy(c.data(), dev_c, size * size * sizeof(float)).wait();

    sycl::free(dev_a, queue);
    sycl::free(dev_b, queue);
    sycl::free(dev_c, queue);

    return c;
}