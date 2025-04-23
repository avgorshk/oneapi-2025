#include "jacobi_dev_oneapi.h"

#include <usm.hpp>
#include <reduction.hpp>

std::vector<float> JacobiDevONEAPI(const std::vector<float> a, const std::vector<float> b, float eps, sycl::device dev) {
    int n = b.size(), step = 0;
    float err = 0;
    std::vector<float> x(n, 0.0f);

    sycl::queue q(dev);
    float *A = sycl::malloc_device<float>(a.size(), q),
          *B = sycl::malloc_device<float>(n, q),
          *cur = sycl::malloc_device<float>(n, q),
          *prev = sycl::malloc_device<float>(n, q),
          *errDev = sycl::malloc_device<float>(1, q);

    q.memcpy(A, a.data(), a.size() * sizeof(float)).wait();
    q.memcpy(B, b.data(), n * sizeof(float)).wait();
    q.memset(cur, 0, n * sizeof(float));
    q.memset(prev, 0, n * sizeof(float));
    q.memset(errDev, 0, sizeof(float));

    while (step++ < ITERATIONS) {
        auto red = sycl::reduction(errDev, sycl::maximum<>());

        q.parallel_for(sycl::range<1>(n), red, [=](sycl::id<1> id, auto& maxErr) {
            int i = id[0];
            float xi = B[i];
            for (int j = 0; j < n; j++)
                if (j != i) xi -= A[i * n + j] * prev[j];
            xi /= A[i * n + i];
            cur[i] = xi;
            maxErr.combine(sycl::fabs(xi - prev[i]));
        });
        q.wait();

        q.memcpy(&err, errDev, sizeof(float)).wait();
        if (err < eps) break;
        q.memset(errDev, 0, sizeof(float)).wait();
        q.memcpy(prev, cur, n * sizeof(float)).wait();
    }

    q.memcpy(x.data(), cur, n * sizeof(float)).wait();

    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(cur, q);
    sycl::free(prev, q);
    sycl::free(errDev, q);

    return x;
}
