// Copyright (c) 2025 Kulagin Aleksandr
#include "gemm_block_oneapi.h"

std::vector<float> GemmBlockONEAPI(const std::vector<float> a, const std::vector<float> b, size_t size, sycl::device device) {
  constexpr size_t block_size = 32;
  assert(size > 0);
  assert(a.size() == size * size);
  assert(b.size() == size * size);
  assert(size % block_size == 0);
  std::vector<float> res(size * size);
  {
    sycl::queue dev_queue(device);
    sycl::buffer<float> a_buf(a.data(), size * size);
    sycl::buffer<float> b_buf(b.data(), size * size);
    sycl::buffer<float> res_buf(res.data(), size * size);
    dev_queue.submit([&](sycl::handler& handler) {
      sycl::local_accessor<float, 2> a_block(sycl::range<2>(block_size, block_size), handler);
      sycl::local_accessor<float, 2> b_block(sycl::range<2>(block_size, block_size), handler);
      auto in_a = a_buf.get_access<sycl::access::mode::read>(handler);
      auto in_b = b_buf.get_access<sycl::access::mode::read>(handler);
      auto out_res = res_buf.get_access<sycl::access::mode::write>(handler);
      handler.parallel_for(sycl::nd_range<2>(sycl::range<2>(size, size), sycl::range<2>(block_size, block_size)), [=](sycl::nd_item<2> item){
        const size_t ii = item.get_local_id(0);
        const size_t jj = item.get_local_id(1);
        const size_t i = item.get_global_id(0);
        const size_t j = item.get_global_id(1);
        float tmp = 0.0f;
        for (size_t k_block = 0; k_block < size / block_size; k_block++) {
          a_block[ii][jj] = in_a[i * size + (block_size * k_block + jj)];
          b_block[ii][jj] = in_b[(block_size * k_block + ii) * size + j];
          item.barrier(sycl::access::fence_space::local_space);
          for (size_t k = 0; k < block_size; k++) {
            tmp += a_block[ii][k] * b_block[k][jj];
          }
          item.barrier(sycl::access::fence_space::local_space);
        }
        out_res[i * size + j] = tmp;
      });
    }).wait();
  }
  return res;
}
