// Copyright (c) 2025 Kulagin Aleksandr
#include "integral_oneapi.h"

#include <cassert>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
  assert(count > 0);
  float res = 0.0f;
  const float dx = (end - start) / static_cast<float>(count);
  {
    sycl::buffer<float> res_buf(&res, 1);
    sycl::queue dev_queue(device);
    dev_queue.submit([&](sycl::handler& handler){
      // https://github.khronos.org/SYCL_Reference/iface/reduction-variables.html
      auto reduction = sycl::reduction(res_buf, handler, sycl::plus<float>());
      handler.parallel_for(sycl::range<2>(count, count), reduction, [=](sycl::id<2> id, auto& sum){
        const float x_i = start + dx * id.get(0);
        const float y_j = start + dx * id.get(1);
        const float x_i_1 = start + dx * (id.get(0) + 1);
        const float y_j_1 = start + dx * (id.get(1) + 1);
        sum += sycl::sin((x_i + x_i_1) / 2.0f) * sycl::cos((y_j + y_j_1) / 2.0f) * (x_i_1 - x_i) * (y_j_1 - y_j);
      });
    });
    dev_queue.wait();
  }
  return res;
}
