#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    sycl::queue q(device);
    float res = 0;
    {
        sycl::buffer<float> res_buf(&res, 1);
        sycl::event e = q.submit([&](sycl::handler& h) {
            auto sumReduction = sycl::reduction(res_buf, h, sycl::plus<>());
            h.parallel_for(sycl::range<2>(count, count), sumReduction, [=](sycl::id<2> i, auto& sum) {
                float xL = start + (end - start) / count * i.get(0);
                float xR = start + (end - start) / count * (i.get(0) + 1);
                float yL = start + (end - start) / count * i.get(1);
                float yR = start + (end - start) / count * (i.get(1) + 1);
                sum += sycl::sin((xL + xR) / 2.0f) * sycl::cos((yL + yR) / 2.0f) * (xR - xL) * (yR - yL);
                });
            });
        e.wait();
    }
    return res;
}
