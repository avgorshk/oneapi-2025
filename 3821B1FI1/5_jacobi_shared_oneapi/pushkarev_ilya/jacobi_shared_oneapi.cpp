#include "jacobi_shared_oneapi.h"
#include <buffer.hpp>
#include <handler.hpp>
#include <range.hpp>
#include <reduction.hpp>
#include <usm.hpp>
#include <vector>

std::vector<float> JacobiSharedONEAPI(const std::vector<float> a,
                                      const std::vector<float> b,
                                      float accuracy, sycl::device device) {
  const int size = b.size();
  int step = 0;
  std::vector<float> ans(size, 0.0f);

  sycl::queue q(device);

  float* a_shared = sycl::malloc_shared<float>(a.size(), q);
  float* b_shared = sycl::malloc_shared<float>(b.size(), q);
  float* curr = sycl::malloc_shared<float>(size, q);
  float* prev = sycl::malloc_shared<float>(size, q);
  float* err = sycl::malloc_shared<float>(1, q);

  q.memcpy(a_shared, a.data(), a.size() * sizeof(float));
  q.memcpy(b_shared, b.data(), b.size() * sizeof(float));
  q.memset(curr, 0, size * sizeof(float));
  q.memset(prev, 0, size * sizeof(float)).wait();
  *err = 0.0f;

  while (step++ < ITERATIONS) {
    auto red = sycl::reduction(err, sycl::maximum<>());

    q.parallel_for(sycl::range<1>(size), red, [=](sycl::id<1> i, auto& max_err) {
      float sum = b_shared[i];
      for (int j = 0; j < size; ++j)
        if (i != j)
          sum -= a_shared[i * size + j] * prev[j];
      sum /= a_shared[i * size + i];

      curr[i] = sum;
      max_err.combine(sycl::fabs(sum - prev[i]));
    });

    q.wait();

    if (*err < accuracy)
      break;

    *err = 0.0f;
    q.memcpy(prev, curr, size * sizeof(float)).wait();
  }

  q.memcpy(ans.data(), curr, size * sizeof(float)).wait();

  sycl::free(a_shared, q);
  sycl::free(b_shared, q);
  sycl::free(curr, q);
  sycl::free(prev, q);
  sycl::free(err, q);

  return ans;
}
