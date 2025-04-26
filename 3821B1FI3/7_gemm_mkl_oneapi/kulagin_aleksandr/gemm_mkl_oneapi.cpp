#include "gemm_mkl_oneapi.h"
#include <oneapi/mkl.hpp>

std::vector<float> GemmMklONEAPI(const std::vector<float> a, const std::vector<float> b, size_t size, sycl::device device) {
  assert(size > 0);
  assert(a.size() == size * size);
  assert(b.size() == size * size);
  std::vector<float> res(size * size);
  {
    sycl::queue dev_queue(device);
    sycl::buffer<float> a_buf(a.data(), size * size);
    sycl::buffer<float> b_buf(b.data(), size * size);
    sycl::buffer<float> res_buf(res.data(), size * size);
    oneapi::mkl::blas::row_major::gemm(dev_queue, oneapi::mkl::transpose::nontrans, oneapi::mkl::transpose::nontrans, size, size, size, 1, a_buf, size, b_buf, size, 0, res_buf, size);
  }
  return res;
}
