#include "gemm_mkl_oneapi.h"
#include <oneapi/mkl/blas.hpp>

std::vector<float> GemmMklONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        size_t size, sycl::device device) {
    sycl::queue queue(device);
    
    std::vector<float> c(size * size, 0.0f);
    
    sycl::buffer<float, 1> buf_a(a.data(), a.size());
    sycl::buffer<float, 1> buf_b(b.data(), b.size());
    sycl::buffer<float, 1> buf_c(c.data(), c.size());
    
    float alpha = 1.0f;
    float beta = 0.0f;
    
    oneapi::mkl::blas::column_major::gemm(
        queue,
        oneapi::mkl::transpose::trans,
        oneapi::mkl::transpose::trans,
        size,
        size,
        size,
        alpha,
        buf_b,
        size,
        buf_a,
        size,
        beta,
        buf_c,
        size
    );
    
    queue.wait();
    
    return c;
}