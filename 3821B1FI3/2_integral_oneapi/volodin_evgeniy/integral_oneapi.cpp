#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  float step = (end - start) / count;
  float result = 0.0f;

  sycl::queue queue(device);

  {
    sycl::buffer<float> result_buf(&result, 1);
    queue
        .submit([&](sycl::handler &h) {
          auto sum = sycl::reduction(result_buf, h, sycl::plus<>());
          h.parallel_for(sycl::range<2>(count, count), sum,
                         [=](sycl::id<2> idx, auto &partial_sum) {
                           int i = idx[0];
                           int j = idx[1];
                           float x = start + (i + 0.5f) * step;
                           float y = start + (j + 0.5f) * step;
                           partial_sum +=
                               std::sin(x) * std::cos(y) * step * step;
                         });
        })
        .wait();
  }

  return result;
}