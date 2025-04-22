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
 
  std::vector ans(b.size(), 0.0f);

  int size = b.size();
  int it = 0;
  float err = 0.0f;

  sycl::queue queue(device);

  float *devA = sycl::malloc_device<float>(a.size(), queue);
  float *devB = sycl::malloc_device<float>(b.size(), queue);
  float *devCurr = sycl::malloc_device<float>(size, queue);
  float *devPrev = sycl::malloc_device<float>(size, queue);
  float *devErr = sycl::malloc_device<float>(1, queue);

  queue.memcpy(devA, a.data(), a.size() * sizeof(float)).wait();
  queue.memcpy(devB, b.data(), b.size() * sizeof(float)).wait();
  queue.memset(devCurr, 0, sizeof(float) * size);
  queue.memset(devPrev, 0, sizeof(float) * size);
  queue.memset(devErr, 0, sizeof(float));

  while (it++ < ITERATIONS) {

    auto reduction = sycl::reduction(devErr, sycl::maximum<>());

    queue.parallel_for(sycl::range<1>(size), reduction,
                       [=](sycl::id<1> id, auto &err) {
                         int i = id.get(0);
                         float curr = devB[i];
                         for (int j = 0; j < size; j++) {
                           if (i != j) {
                             curr -= devA[i * size + j] * devPrev[j];
                           }
                         }
                         curr /= devA[i * size + i];
                         devCurr[i] = curr;

                         float diff = sycl::fabs(curr - devPrev[i]);
                         err.combine(diff);
                       });

    queue.wait();

    queue.memcpy(&err, devErr, sizeof(float)).wait();
    if (err < accuracy)
      break;
    queue.memset(devErr, 0, sizeof(float)).wait();

    queue.memcpy(devPrev, devCurr, size * sizeof(float)).wait();
  }

  queue.memcpy(ans.data(), devCurr, size * sizeof(float)).wait();

  sycl::free(devA, queue);
  sycl::free(devB, queue);
  sycl::free(devCurr, queue);
  sycl::free(devPrev, queue);
  sycl::free(devErr, queue);

  return ans;
}
