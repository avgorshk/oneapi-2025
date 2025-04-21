#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    const float interval_length = end - start;
    const float h = interval_length / count;
    const float h_sq = h * h;
    const float half_point_offset = 0.5f;
    float result = 0.0f;

    {
        sycl::queue q(device);
        sycl::buffer<float, 1> result_buf(&result, sycl::range<1>(1));

        q.submit([&](sycl::handler& cgh) {
            auto reduction = sycl::reduction(result_buf, cgh, sycl::plus<>());

            cgh.parallel_for(
                sycl::range<2>(count, count),
                reduction,
                [=](sycl::id<2> idx, auto& sum) {
                    float x = start + h * (idx[0] + half_point_offset);
                    float y = start + h * (idx[1] + half_point_offset);

                    sum += sycl::sin(x) * sycl::cos(y) * h_sq;
                });
        }).wait();
    }
    
    return result;
}
