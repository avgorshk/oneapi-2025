#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    const float h = (end - start) / count;
    const float h_sq = h * h;
    float result = 0.0f;

    sycl::queue q(device);
    sycl::buffer<float, 1> result_buf(&result, 1);

    {
        q.submit([&](sycl::handler& h) {
            auto reduction = sycl::reduction(result_buf, h, sycl::plus<>());

            h.parallel_for(sycl::range<2>(count, count), reduction, [=](sycl::id<2> idx, auto& sum) {
                float x = start + h * (idx[0] + 0.5f);
                float y = start + h * (idx[1] + 0.5f);

                sum += sycl::sin(x) * sycl::cos(y) * h_sq;
                });
            }).wait();
    }
    
    return result;
}