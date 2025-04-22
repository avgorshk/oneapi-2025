#include "integral_oneapi.h"
#include <sycl/sycl.hpp>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    float result = 0.0f;
    float step = (end - start) / count;

    try {
        sycl::queue queue(device);
        sycl::buffer<float> result_buf(&result, 1);

        queue.submit([&](sycl::handler& cgh) {
            auto reduction = sycl::reduction(result_buf, cgh, sycl::plus<>());
            cgh.parallel_for(sycl::range<2>(count, count), reduction, [=](sycl::id<2> id, auto& sum) {
                float x_mid = start + (id[0] + 0.5f) * step;
                float y_mid = start + (id[1] + 0.5f) * step;
                sum += sycl::sin(x_mid) * sycl::cos(y_mid) * step * step;
            });
        }).wait();
    } catch (sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
    }

    return result;
}