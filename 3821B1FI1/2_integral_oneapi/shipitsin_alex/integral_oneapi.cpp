#include "integral_oneapi.h"
#include <range.hpp>
#include <reduction.hpp>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  float result = 0.0f;
  const float delta = (end - start) / count;

  sycl::buffer<float> buf_result(&result, 1);
  sycl::queue queue(device);

  queue.submit([&](sycl::handler &cgh) {
    auto reduction = sycl::reduction(buf_result, cgh, sycl::plus<>());
    
    cgh.parallel_for(sycl::range<2>(count, count), reduction, [=](sycl::id<2> id, auto& sum) {
      const int i = id[0];
      const int j = id[1];

      const float Xi   = start + delta * i;
      const float Xi_1 = Xi + delta;
      const float Yi   = start + delta * j;
      const float Yi_1 = Yi + delta;

      const float mid_x = (Xi + Xi_1) * 0.5f;
      const float mid_y = (Yi + Yi_1) * 0.5f;
      const float area = delta * delta;

      sum += sycl::sin(mid_x) * sycl::cos(mid_y) * area;
    });
  });

  queue.wait();
  return result;
}