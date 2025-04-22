#include "gemm_mkl_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(const std::vector<float> a,
                                 const std::vector<float> b, size_t size,
                                 sycl::device device) {

  sycl::queue queue(device);
  std::vector<float> result(size * size);

  {
    sycl::buffer<float> buffer_a(a.data(), sycl::range<1>(a.size()));
    sycl::buffer<float> buffer_b(b.data(), sycl::range<1>(b.size()));
    sycl::buffer<float> buffer_c(result.data(), sycl::range<1>(result.size()));

    using oneapi::mkl::transpose;
    using oneapi::mkl::blas::row_major::gemm;

    gemm(queue, transpose::nontrans, transpose::nontrans, size, size, size,
         1.0f, buffer_a, size, buffer_b, size, 0.0f, buffer_c, size);
  }

  return result;
}