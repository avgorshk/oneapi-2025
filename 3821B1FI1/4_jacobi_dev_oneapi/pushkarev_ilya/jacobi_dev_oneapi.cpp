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
  const int size = b.size();
  int step = 0;
  float error = 0.0f;
  std::vector<float> ans(size, 0.0f);

  sycl::queue q(device);

  float* d_a = sycl::malloc_device<float>(a.size(), q);
  float* d_b = sycl::malloc_device<float>(b.size(), q);
  float* d_curr = sycl::malloc_device<float>(size, q);
  float* d_prev = sycl::malloc_device<float>(size, q);
  float* d_error = sycl::malloc_device<float>(1, q);

  q.memcpy(d_a, a.data(), a.size() * sizeof(float));
  q.memcpy(d_b, b.data(), b.size() * sizeof(float));
  q.memset(d_curr, 0, sizeof(float) * size);
  q.memset(d_prev, 0, sizeof(float) * size);
  q.memset(d_error, 0, sizeof(float)).wait();

  while (step++ < ITERATIONS) {
    auto red = sycl::reduction(d_error, sycl::maximum<>());

    q.parallel_for(sycl::range<1>(size), red, [=](sycl::id<1> i, auto& err) {
      float sum = d_b[i];
      for (int j = 0; j < size; ++j)
        if (i != j)
          sum -= d_a[i * size + j] * d_prev[j];
      sum /= d_a[i * size + i];
      d_curr[i] = sum;

      err.combine(sycl::fabs(sum - d_prev[i]));
    });
    q.wait();

    q.memcpy(&error, d_error, sizeof(float)).wait();
    if (error < accuracy)
      break;

    q.memset(d_error, 0, sizeof(float));
    q.memcpy(d_prev, d_curr, size * sizeof(float));
  }

  q.memcpy(ans.data(), d_curr, size * sizeof(float)).wait();

  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_curr, q);
  sycl::free(d_prev, q);
  sycl::free(d_error, q);

  return ans;
}
