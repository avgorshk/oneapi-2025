#include "integral_oneapi.h"

#include <cassert>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  assert(count > 0);
  float res = 0.0;
  const float dx = ((end - start) / static_cast<float>(count));
  {
    sycl::buffer<float> res_buf(&res, 1);
    sycl::queue queue(device);

    queue.submit([&](sycl::handler& handler) {
      auto reduction = sycl::reduction(res_buf, handler, sycl::plus<float>());
      handler.parallel_for(sycl::range<2>(count, count), reduction, [=](sycl::id<2> id, auto& sum){
        const float x_i = start + dx * id.get(0);
        const float y_j = start + dx * id.get(1);
        const float x_i_1 = start + dx * (id.get(0) + 1);
        const float y_j_1 = start + dx * (id.get(1) + 1);

        sum += sycl::sin((x_i + x_i_1) / 2.0f) * sycl::cos((y_j + y_j_1) / 2.0f) * (x_i_1 - x_i) * (y_j_1 - y_j);
      });
    });
    queue.wait();
  }
  return res;
}
