#include "integral_oneapi.h"
#include <range.hpp>
#include <reduction.hpp>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    float ans = 0.0f;

    {
        sycl::buffer<float> buf_ans(&ans, 1);

        sycl::queue queue(device);

        queue.submit([&](sycl::handler& cgh) {
            auto reduction = sycl::reduction(buf_ans, cgh, sycl::plus<>());

            cgh.parallel_for(sycl::range<2>(count, count), reduction, [=](sycl::id<2> id, auto& sum) {
                float Xi = start + (end - start) / count * id.get(0);
                float Xi_1 = start + (end - start) / count * (id.get(0) + 1);
                float Yi = start + (end - start) / count * id.get(1);
                float Yi_1 = start + (end - start) / count * (id.get(1) + 1);
                sum += sycl::sin((Xi + Xi_1) / 2.0f) * sycl::cos((Yi + Yi_1) / 2.0f) * (Xi_1 - Xi) * (Yi_1 - Yi);
                });
            });

        queue.wait();
    }


    return ans;
}