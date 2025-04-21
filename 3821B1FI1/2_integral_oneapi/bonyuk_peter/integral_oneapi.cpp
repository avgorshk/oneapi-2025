#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    if (count <= 0 || start >= end) {
        return 0.0f;
    }

    const float step = (end - start) / count;
    const float area = step * step;
   
    sycl::queue q(device);

    float* partial_sums = sycl::malloc_shared<float>(count * count, q);

    q.parallel_for(sycl::range<2>(count, count), [=](sycl::id<2> idx) {
        int i = idx[0];
        int j = idx[1];

        float x_mid = start + (i + 0.5f) * step;
        float y_mid = start + (j + 0.5f) * step;

        float f_val = sycl::sin(x_mid) * sycl::cos(y_mid);

        partial_sums[i * count + j] = f_val * area;
    }).wait();

    float total = 0.0f;
    for (int i = 0; i < count * count; ++i) {
        total += partial_sums[i];
    }

    sycl::free(partial_sums, q);

    return total;
}