#include "integral_oneapi.h"

#include <range.hpp>
#include <reduction.hpp>


float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  float res = 0.0f;
  float interval_len = (end - start) / count;
  {
    sycl::buffer<float> buf_res(&res, 1);

    sycl::queue queue(device);

    queue.submit([&](sycl::handler &cgh) {
      auto reduction = sycl::reduction(buf_res, cgh, sycl::plus<>());

      cgh.parallel_for(sycl::range<2>(count, count), reduction, [=](sycl::id<2> id, auto& sum) {
        float x1 = start + interval_len * id.get(0);
        float x2 = start + interval_len * (id.get(0) + 1);
        float y1 = start + interval_len * id.get(1);
        float y2 = start + interval_len * (id.get(1) + 1);
        sum += sycl::sin((x1 + x2) / 2.0f) * sycl::cos((y1 + y2) / 2.0f) * (x2 - x1) * (y2 - y1);
      });
    });
    queue.wait();
  }
  return res;
}
