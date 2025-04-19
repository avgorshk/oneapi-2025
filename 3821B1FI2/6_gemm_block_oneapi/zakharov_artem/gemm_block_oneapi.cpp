#include "gemm_block_oneapi.h"

#include <buffer.hpp>
#include <handler.hpp>
#include <range.hpp>
#include <reduction.hpp>
#include <usm.hpp>
#include <vector>

std::vector<float> GemmBlockONEAPI(const std::vector<float> a,
                                   const std::vector<float> b, size_t size,
                                   sycl::device device) {
  constexpr size_t block_size = 16;
  size_t total_elements = size * size;
  std::vector<float> ans(total_elements, 0.0f);

  sycl::queue queue(device);

  float *dev_a = sycl::malloc_device<float>(total_elements, queue);
  float *dev_b = sycl::malloc_device<float>(total_elements, queue);
  float *dev_c = sycl::malloc_device<float>(total_elements, queue);

  queue.memcpy(dev_a, a.data(), total_elements * sizeof(float)).wait();
  queue.memcpy(dev_b, b.data(), total_elements * sizeof(float)).wait();

  sycl::range<2> global_range((size + block_size - 1) / block_size * block_size,
                              (size + block_size - 1) / block_size * block_size);
  sycl::range<2> local_range(block_size, block_size);

  queue.submit([&](sycl::handler &cgh) {
    sycl::local_accessor<float, 2> block_a(local_range, cgh);
    sycl::local_accessor<float, 2> block_b(local_range, cgh);

    cgh.parallel_for(sycl::nd_range<2>(global_range, local_range),
                     [=](sycl::nd_item<2> item) {
                       int global_y = item.get_global_id(0);
                       int global_x = item.get_global_id(1);
                       int local_y = item.get_local_id(0);
                       int local_x = item.get_local_id(1);

                       int grid_dim = item.get_group_range(0);

                       float cur_value = 0.0f;

                       for (int block_k = 0; block_k < grid_dim; block_k++) {
                         int col = block_k * block_size + local_x;
                         int row = block_k * block_size + local_y;

                         block_a[local_y][local_x] =
                             dev_a[global_y * size + col];

                         block_b[local_y][local_x] =
                             dev_b[row * size + global_x];

                         item.barrier(sycl::access::fence_space::local_space);

                         for (int k = 0; k < block_size; k++) {
                           cur_value += block_a[local_y][k] * block_b[k][local_x];
                         }
                         item.barrier(sycl::access::fence_space::local_space);
                       }
                       dev_c[global_y * size + global_x] = cur_value;
                     });
  });
  queue.wait();

  queue.memcpy(ans.data(), dev_c, total_elements * sizeof(float)).wait();

  free(dev_a, queue);
  free(dev_b, queue);
  free(dev_c, queue);

  return ans;
}
