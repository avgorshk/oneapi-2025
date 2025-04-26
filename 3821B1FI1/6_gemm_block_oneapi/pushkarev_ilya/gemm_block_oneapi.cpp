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
  const size_t total = size * size;
  std::vector<float> result(total, 0.0f);

  sycl::queue q(device);

  float* A = sycl::malloc_device<float>(total, q);
  float* B = sycl::malloc_device<float>(total, q);
  float* C = sycl::malloc_device<float>(total, q);

  q.memcpy(A, a.data(), total * sizeof(float)).wait();
  q.memcpy(B, b.data(), total * sizeof(float)).wait();

  // Округляем глобальный размер под block_size
  sycl::range<2> global((size + block_size - 1) / block_size * block_size,
                        (size + block_size - 1) / block_size * block_size);
  sycl::range<2> local(block_size, block_size);

  q.submit([&](sycl::handler& h) {
    // Локальные тайлы для блоков A и B
    sycl::local_accessor<float, 2> tileA(local, h);
    sycl::local_accessor<float, 2> tileB(local, h);

    h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
      int row = item.get_global_id(0);
      int col = item.get_global_id(1);
      int local_row = item.get_local_id(0);
      int local_col = item.get_local_id(1);

      float sum = 0.0f;

      // Идём по блокам по оси K
      for (int bk = 0; bk < size; bk += block_size) {
        int tiled_col = bk + local_col;
        int tiled_row = bk + local_row;

        // Загружаем блоки в локальную память
        tileA[local_row][local_col] = (tiled_col < size && row < size)
          ? A[row * size + tiled_col] : 0.0f;
        tileB[local_row][local_col] = (tiled_row < size && col < size)
          ? B[tiled_row * size + col] : 0.0f;

        item.barrier(sycl::access::fence_space::local_space);

        // Умножение и накопление
        for (int k = 0; k < block_size; k++) {
          sum += tileA[local_row][k] * tileB[k][local_col];
        }

        item.barrier(sycl::access::fence_space::local_space);
      }

      if (row < size && col < size)
        C[row * size + col] = sum;
    });
  });

  q.wait();
  q.memcpy(result.data(), C, total * sizeof(float)).wait();

  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(C, q);

  return result;
}
