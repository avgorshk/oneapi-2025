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

  std::vector ans(b.size(), 0.0f);

  int size = b.size();
  int it = 0;
  
  sycl::queue queue(device);

  float *sharedA = sycl::malloc_shared<float>(a.size(), queue);
  float *sharedB = sycl::malloc_shared<float>(b.size(), queue);
  float *sharedCurr = sycl::malloc_shared<float>(size, queue);
  float *sharedPrev = sycl::malloc_shared<float>(size, queue);
  float *sharedErr = sycl::malloc_shared<float>(1, queue);

  queue.memcpy(sharedA, a.data(), a.size() * sizeof(float));
  queue.memcpy(sharedB, b.data(), b.size() * sizeof(float));
  queue.memset(sharedCurr, 0, sizeof(float) * size);
  queue.memset(sharedPrev, 0, sizeof(float) * size);
  *sharedErr = 0;

  while (it++ < ITERATIONS) {
    auto reduction = sycl::reduction(sharedErr, sycl::maximum<>());

    queue.parallel_for(sycl::range<1>(size), reduction,
                       [=](sycl::id<1> id, auto &error) {
                         int i = id.get(0);
                         float curr = sharedB[i];
                         for (int j = 0; j < size; j++) {
                           if (i != j) {
                             curr -= sharedA[i * size + j] * sharedPrev[j];
                           }
                         }
                         curr /= sharedA[i * size + i];
                         sharedCurr[i] = curr;

                         float diff = sycl::fabs(curr - sharedPrev[i]);
                         error.combine(diff);
                       });

    queue.wait();

    if (*sharedErr < accuracy)
      break;
    *sharedErr = 0;

    queue.memcpy(sharedPrev, sharedCurr, size * sizeof(float)).wait();
  }

  queue.memcpy(ans.data(), sharedCurr, size * sizeof(float)).wait();

  sycl::free(sharedA, queue);
  sycl::free(sharedB, queue);
  sycl::free(sharedCurr, queue);
  sycl::free(sharedPrev, queue);
  sycl::free(sharedErr, queue);

  return ans;
}
