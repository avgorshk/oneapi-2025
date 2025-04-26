#include "gemm_mkl_oneapi.h"

#include <buffer.hpp>
#include <oneapi/mkl.hpp>
#include <vector>

std::vector<float> GemmMklONEAPI(const std::vector<float> a,
                                 const std::vector<float> b, size_t size,
                                 sycl::device device) {
  std::vector<float> ans(size * size, 0.0f);

  sycl::queue queue(device);

  {
    sycl::buffer<float> bufA(a.data(), a.size());
    sycl::buffer<float> bufB(b.data(), b.size());
    sycl::buffer<float> bufAns(ans.data(), ans.size());

    auto nontrans = oneapi::mkl::transpose::nontrans;

    oneapi::mkl::blas::row_major::gemm(queue, nontrans, nontrans, size, size,
                                       size, 1, bufA, size, bufB, size, 0,
                                       bufAns, size);
  }

  return ans;
}
