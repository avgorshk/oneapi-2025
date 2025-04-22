#include "gemm_block_oneapi.h"
#include <cassert>

std::vector<float> GemmBlockONEAPI(const std::vector<float> a,
                                   const std::vector<float> b, size_t size,
                                   sycl::device device) {

  constexpr int BS = 16;
  assert(size % BS == 0);

  sycl::queue q(device);
  std::vector<float> result(size * size, 0.0f);

  sycl::buffer<float> bufA(a.data(), a.size());
  sycl::buffer<float> bufB(b.data(), b.size());
  sycl::buffer<float> bufC(result.data(), result.size());

  q.submit([&](sycl::handler &h) {
     sycl::local_accessor<float, 2> localA({BS, BS}, h);
     sycl::local_accessor<float, 2> localB({BS, BS}, h);

     auto accA = bufA.get_access<sycl::access::mode::read>(h);
     auto accB = bufB.get_access<sycl::access::mode::read>(h);
     auto accC = bufC.get_access<sycl::access::mode::write>(h);

     h.parallel_for(
         sycl::nd_range<2>({size, size}, {BS, BS}), [=](sycl::nd_item<2> item) {
           int lid_i = item.get_local_id(0);
           int lid_j = item.get_local_id(1);
           int gid_i = item.get_global_id(0);
           int gid_j = item.get_global_id(1);

           float tmp = 0.0f;

           for (int bk = 0; bk < size; bk += BS) {
             localA[lid_i][lid_j] = accA[gid_i * size + (bk + lid_j)];
             localB[lid_i][lid_j] = accB[(bk + lid_i) * size + gid_j];
             item.barrier(sycl::access::fence_space::local_space);

             for (int k = 0; k < BS; ++k) {
               tmp += localA[lid_i][k] * localB[k][lid_j];
             }

             item.barrier(sycl::access::fence_space::local_space);
           }

           accC[gid_i * size + gid_j] = tmp;
         });
   }).wait();

  return result;
}
