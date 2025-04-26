// Copyright (c) 2025 Prokofev Kirill
#include "jacobi_dev_oneapi.h"

std::vector<float> JacobiDevONEAPI(const std::vector<float> a, const std::vector<float> b, float accuracy, sycl::device device) {
  static_assert(ITERATIONS >= 1, "More than 1 iteration");
  const size_t n = b.size();
  assert(n > 0);
  assert(a.size() == n * n);
  assert(accuracy >= 0.0f);
  std::vector<float> res(n, 0.0f);
  std::vector<float> res_prev(res);
  int attempt = 0;
  float error = 0.0f;
  {
    sycl::queue dev_queue(device);
    float* a_dev = sycl::malloc_device<float>(n * n, dev_queue);
    float* b_dev = sycl::malloc_device<float>(n, dev_queue);
    float* res_dev = sycl::malloc_device<float>(n, dev_queue);
    float* res_prev_dev = sycl::malloc_device<float>(n, dev_queue);
    float* error_dev = sycl::malloc_device<float>(1, dev_queue);
    dev_queue.memcpy(a_dev, a.data(), n * n * sizeof(float));
    dev_queue.memcpy(b_dev, b.data(), n * sizeof(float));
    dev_queue.memset(res_dev, 0.0f, n * sizeof(float));
    dev_queue.memset(error_dev, 0.0f, sizeof(float));
    dev_queue.wait();
    while (attempt < ITERATIONS) {
      std::swap(res_dev, res_prev_dev);
      dev_queue.submit([&](sycl::handler& handler) {
        auto reduction = sycl::reduction(error_dev, sycl::maximum<float>());
        handler.parallel_for(sycl::range<1>(n), reduction, [=](sycl::id<1> id, auto& error) {
          const size_t i = id.get(0);
          float g = b_dev[i];
          for (size_t j = 0; j < n; j++) {
            if (i != j) {
              g -= a_dev[i * n + j] * res_prev_dev[j];
            }
          }
          g /= a_dev[i * n + i];
          res_dev[i] = g;
          error.combine(sycl::fabs(g - res_prev_dev[i]));
        });
      }).wait();
      dev_queue.memcpy(&error, error_dev, sizeof(float)).wait();
      if (error < accuracy) {
        break;
      }
      dev_queue.memset(error_dev, 0.0f, sizeof(float)).wait();
      attempt++;
    }
    dev_queue.memcpy(res.data(), res_dev, n * sizeof(float)).wait();
    sycl::free(a_dev, dev_queue);
    sycl::free(b_dev, dev_queue);
    sycl::free(res_dev, dev_queue);
    sycl::free(res_prev_dev, dev_queue);
    sycl::free(error_dev, dev_queue);
  }
  return res;
}
