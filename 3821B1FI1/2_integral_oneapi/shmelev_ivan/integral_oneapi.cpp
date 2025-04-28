#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    float hx = (end - start) / count;
    float hy = (end - start) / count;
    float integral_sum = 0.0f;

    sycl::queue q(device);

    q.submit([&](sycl::handler& h) {
        sycl::range<2> num_items{count, count};

        h.parallel_for(num_items, [=](sycl::item<2> item) {
            int i = item.get_id(0);
            int j = item.get_id(1);

            float x = start + (i + 0.5f) * hx;
            float y = start + (j + 0.5f) * hy;

            float f_value = std::sin(x) * std::cos(y);

            sycl::atomic<float>(integral_sum) += f_value * hx * hy;
        });
    });

    q.wait();
    return integral_sum;
}
