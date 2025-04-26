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
  int size = b.size();
  int step = 0;
  std::vector<float> ans(size, 0.0f);

  sycl::queue queue(device);

  float *shared_a = sycl::malloc_shared<float>(a.size(), queue);
  float *shared_b = sycl::malloc_shared<float>(b.size(), queue);
  float *shared_curr = sycl::malloc_shared<float>(size, queue);
  float *shared_prev = sycl::malloc_shared<float>(size, queue);
  float *shared_error = sycl::malloc_shared<float>(1, queue);

  queue.memcpy(shared_a, a.data(), a.size() * sizeof(float));
  queue.memcpy(shared_b, b.data(), b.size() * sizeof(float));
  queue.memset(shared_curr, 0, sizeof(float) * size);
  queue.memset(shared_prev, 0, sizeof(float) * size);
  *shared_error = 0;

  while (step++ < ITERATIONS) {
    auto reduction = sycl::reduction(shared_error, sycl::maximum<>());

    queue.parallel_for(sycl::range<1>(size), reduction,
                       [=](sycl::id<1> id, auto &error) {
                         int i = id.get(0);
                         float curr = shared_b[i];
                         for (int j = 0; j < size; j++) {
                           if (i != j) {
                             curr -= shared_a[i * size + j] * shared_prev[j];
                           }
                         }
                         curr /= shared_a[i * size + i];
                         shared_curr[i] = curr;

                         float diff = sycl::fabs(curr - shared_prev[i]);
                         error.combine(diff);
                       });

    queue.wait();

    if (*shared_error < accuracy)
      break;
    *shared_error = 0;

    queue.memcpy(shared_prev, shared_curr, size * sizeof(float)).wait();
  }

  queue.memcpy(ans.data(), shared_curr, size * sizeof(float)).wait();

  sycl::free(shared_a, queue);
  sycl::free(shared_b, queue);
  sycl::free(shared_curr, queue);
  sycl::free(shared_prev, queue);
  sycl::free(shared_error, queue);

  return ans;
}