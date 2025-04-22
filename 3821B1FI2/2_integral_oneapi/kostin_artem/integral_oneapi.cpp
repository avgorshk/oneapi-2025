#include "integral_oneapi.h"

#include <range.hpp>
#include <reduction.hpp>

float IntegralONEAPI(float s, float e, int n, sycl::device d) {
    float r = 0, h = (e - s) / n;
    sycl::buffer<float> b(&r, 1);
    sycl::queue q(d);

    q.submit([&](sycl::handler& c) {
        auto red = sycl::reduction(b, c, sycl::plus<>());
        c.parallel_for(sycl::range<2>(n, n), red, [=](sycl::id<2> id, auto& sum) {
            float x1 = s + h * id[0], x2 = x1 + h;
            float y1 = s + h * id[1], y2 = y1 + h;
            sum += sycl::sin((x1 + x2) * 0.5f) * sycl::cos((y1 + y2) * 0.5f) * h * h;
        });
    });
    q.wait();
    return r;
}
