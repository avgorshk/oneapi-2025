#include "jacobi_shared_oneapi.h"
#include <cmath>
#include <cassert>

std::vector<float> JacobiSharedONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device) {
    const size_t n = b.size();
    assert(a.size() == n * n);

    sycl::queue queue(device);

    float* shared_a = sycl::malloc_shared<float>(n * n, queue);
    float* shared_b = sycl::malloc_shared<float>(n, queue);
    float* shared_x = sycl::malloc_shared<float>(n, queue);
    float* shared_x_new = sycl::malloc_shared<float>(n, queue);
    float* shared_error = sycl::malloc_shared<float>(1, queue);

    queue.memcpy(shared_a, a.data(), n * n * sizeof(float)).wait();
    queue.memcpy(shared_b, b.data(), n * sizeof(float)).wait();
    queue.memset(shared_x, 0, n * sizeof(float)).wait();
    queue.memset(shared_x_new, 0, n * sizeof(float)).wait();
    *shared_error = 0.0f;

    for (int iter = 0; iter < ITERATIONS; ++iter) {
        queue.submit([&](sycl::handler& cgh) {
            auto reduction = sycl::reduction(shared_error, cgh, sycl::maximum<>());

            cgh.parallel_for(sycl::range<1>(n), reduction, [=](sycl::id<1> i, auto& max_error) {
                float sigma = 0.0f;
                for (size_t j = 0; j < n; ++j) {
                    if (j != i) {
                        sigma += shared_a[i * n + j] * shared_x[j];
                    }
                }
                shared_x_new[i] = (shared_b[i] - sigma) / shared_a[i * n + i];
                max_error.combine(std::fabs(shared_x_new[i] - shared_x[i]));
            });
        }).wait();

        if (*shared_error < accuracy) {
            break;
        }

        queue.memcpy(shared_x, shared_x_new, n * sizeof(float)).wait();
        *shared_error = 0.0f;
    }

    std::vector<float> result(n);
    queue.memcpy(result.data(), shared_x_new, n * sizeof(float)).wait();

    sycl::free(shared_a, queue);
    sycl::free(shared_b, queue);
    sycl::free(shared_x, queue);
    sycl::free(shared_x_new, queue);
    sycl::free(shared_error, queue);

    return result;
}