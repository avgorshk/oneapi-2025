#include "jacobi_acc_oneapi.h"

#include <cmath>
#include <algorithm>

std::vector<float> JacobiAccONEAPI(
    const std::vector<float> a, const std::vector<float> b,
    float accuracy, sycl::device device) {

    size_t n = b.size();
    std::vector<float> x(n, 0.0f);
    std::vector<float> x_new(n, 0.0f);

    sycl::queue queue(device);

    sycl::buffer<float> a_buf(a.data(), sycl::range<1>(n * n));
    sycl::buffer<float> b_buf(b.data(), sycl::range<1>(n));
    sycl::buffer<float> x_buf(x.data(), sycl::range<1>(n));
    sycl::buffer<float> x_new_buf(x_new.data(), sycl::range<1>(n));

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        queue.submit([&](sycl::handler& cgh) {
            auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
            auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_acc = x_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_new_acc = x_new_buf.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for<class JacobiStep>(sycl::range<1>(n), [=](sycl::id<1> i) {
                float sigma = 0.0f;
                for (size_t j = 0; j < n; ++j) {
                    if (j != i) {
                        sigma += a_acc[i * n + j] * x_acc[j];
                    }
                }
                x_new_acc[i] = (b_acc[i] - sigma) / a_acc[i * n + i];
                });
            });

        queue.wait();

        float max_diff = 0.0f;
        {
            auto x_acc = x_buf.get_access<sycl::access::mode::read>();
            auto x_new_acc = x_new_buf.get_access<sycl::access::mode::read>();

            for (size_t i = 0; i < n; ++i) {
                max_diff = std::max(max_diff, std::fabs(x_new_acc[i] - x_acc[i]));
            }
        }

        if (max_diff < accuracy) {
            break;
        }

        queue.submit([&](sycl::handler& cgh) {
            auto x_acc = x_buf.get_access<sycl::access::mode::write>(cgh);
            auto x_new_acc = x_new_buf.get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for<class SwapStep>(sycl::range<1>(n), [=](sycl::id<1> i) {
                x_acc[i] = x_new_acc[i];
                });
            });

        queue.wait();
    }

    auto x_result = std::vector<float>(n);
    {
        auto x_acc = x_new_buf.get_access<sycl::access::mode::read>();
        for (size_t i = 0; i < n; ++i) {
            x_result[i] = x_acc[i];
        }
    }

    return x_result;
}