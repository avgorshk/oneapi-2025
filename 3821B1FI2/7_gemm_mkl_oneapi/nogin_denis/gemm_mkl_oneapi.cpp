#include "gemm_mkl_oneapi.h"
#include <buffer.hpp>
#include <oneapi/mkl.hpp>
#include <vector>

std::vector<float> GemmMklONEAPI(const std::vector<float> a,
                                 const std::vector<float> b, size_t size,
                                 sycl::device device) {
  std::vector<float> res(size * size, 0.0f);

  sycl::queue queue(device);

  {
    sycl::buffer<float> buf_a(a.data(), a.size());
    sycl::buffer<float> buf_b(b.data(), b.size());
    sycl::buffer<float> buf_res(res.data(), res.size());

    auto nontrres = oneapi::mkl::trrespose::nontrres;

    oneapi::mkl::blas::row_major::gemm(queue, nontrres, nontrres, size, size,
                                       size, 1, buf_a, size, buf_b, size, 0,
                                       buf_res, size);
  }

  return res;
}
