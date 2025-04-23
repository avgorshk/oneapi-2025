#include "jacobi_shared_oneapi.h"
#include <reduction.hpp>
#include <usm.hpp>

std::vector<float> JacobiSharedONEAPI(const std::vector<float> a, const std::vector<float> b,
                                      float accuracy, sycl::device device) {
    int n = b.size(), step = 0;
    std::vector<float> x(n, 0.0f);

    sycl::queue q(device);

    float *A = sycl::malloc_shared<float>(a.size(), q),
          *B = sycl::malloc_shared<float>(n, q),
          *cur = sycl::malloc_shared<float>(n, q),
          *prev = sycl::malloc_shared<float>(n, q),
          *err = sycl::malloc_shared<float>(1, q);

    q.memcpy(A, a.data(), a.size() * sizeof(float));
    q.memcpy(B, b.data(), n * sizeof(float));
    q.memset(cur, 0, n * sizeof(float));
    q.memset(prev, 0, n * sizeof(float));
    *err = 0;

    while (step++ < ITERATIONS) {
        auto red = sycl::reduction(err, sycl::maximum<>());

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

        if (*err < accuracy) break;
        *err = 0;
        q.memcpy(prev, cur, n * sizeof(float)).wait();
    }

    q.memcpy(x.data(), cur, n * sizeof(float)).wait();

    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(cur, q);
    sycl::free(prev, q);
    sycl::free(err, q);

    return x;
}
