#include "jacobi_dev_oneapi.h"

#include <buffer.hpp>
#include <handler.hpp>
#include <range.hpp>
#include <reduction.hpp>
#include <usm.hpp>
#include <vector>

std::vector<float> JacobiDevONEAPI(const std::vector<float> a,
                                   const std::vector<float> b, float accuracy,
                                   sycl::device device) {
  int size = b.size();
  int step = 0;
  float error = 0.0f;
  std::vector ans(size, 0.0f);

  sycl::queue queue(device);

  float *dev_a = sycl::malloc_device<float>(a.size(), queue);
  float *dev_b = sycl::malloc_device<float>(b.size(), queue);
  float *dev_cur = sycl::malloc_device<float>(size, queue);
  float *dev_prev = sycl::malloc_device<float>(size, queue);
  float *dev_error = sycl::malloc_device<float>(1, queue);

  queue.memcpy(dev_a, a.data(), a.size() * sizeof(float)).wait();
  queue.memcpy(dev_b, b.data(), b.size() * sizeof(float)).wait();
  queue.memset(dev_cur, 0, sizeof(float) * size);
  queue.memset(dev_prev, 0, sizeof(float) * size);
  queue.memset(dev_error, 0, sizeof(float));

  while (step++ < ITERATIONS) {
    auto reduction = sycl::reduction(dev_error, sycl::maximum<>());

    queue.parallel_for(sycl::range<1>(size), reduction,
                       [=](sycl::id<1> id, auto &error) {
                         int i = id.get(0);
                         float cur = dev_b[i];
                         for (int j = 0; j < size; j++) {
                           if (i != j) {
                             cur -= dev_a[i * size + j] * dev_prev[j];
                           }
                         }
                         cur /= dev_a[i * size + i];
                         dev_cur[i] = cur;

                         float diff = sycl::fabs(cur - dev_prev[i]);
                         error.combine(diff);
                       });
    queue.wait();

    queue.memcpy(&error, dev_error, sizeof(float)).wait();
    if (error < accuracy)
      break;
    queue.memset(dev_error, 0, sizeof(float)).wait();

    queue.memcpy(dev_prev, dev_cur, size * sizeof(float)).wait();
  }
  queue.memcpy(ans.data(), dev_cur, size * sizeof(float)).wait();

  sycl::free(dev_a, queue);
  sycl::free(dev_b, queue);
  sycl::free(dev_cur, queue);
  sycl::free(dev_prev, queue);
  sycl::free(dev_error, queue);

  return ans;
}
