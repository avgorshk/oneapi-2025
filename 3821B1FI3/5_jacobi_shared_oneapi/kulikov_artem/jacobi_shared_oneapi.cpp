#include "jacobi_shared_oneapi.h"

#include <utility>

std::vector<float> JacobiSharedONEAPI(
    const std::vector<float> a, const std::vector<float> b,
    float accuracy, sycl::device device) {
    const auto n = b.size();

    std::vector<float> x_old(n);
    std::vector<float> x_new(n, 0.0f);
    std::vector<float> a_diag(n);
    for (int i = 0; i < n; i++) a_diag[i] = a[i * n + i];

    sycl::queue q(device);

    float *d_a = sycl::malloc_shared<float>(n * n, q);
    float *d_a_diag = sycl::malloc_shared<float>(n, q);
    float *d_b = sycl::malloc_shared<float>(n, q);
    float *d_x_old = sycl::malloc_shared<float>(n, q);
    float *d_x_new = sycl::malloc_shared<float>(n, q);

    q.memcpy(d_a, a.data(), sizeof(float) * n * n).wait();
    q.memcpy(d_a_diag, a_diag.data(), sizeof(float) * n).wait();
    q.memcpy(d_b, b.data(), sizeof(float) * n).wait();

    q.memset(d_x_new, 0, sizeof(float) * n).wait();

    float err = 0.0f;
    int k = ITERATIONS;
    sycl::buffer<float, 1> err_buf(&err, sycl::range<1>(1));
    do {
        {
            auto error = err_buf.get_host_access();
            error[0] = 0.0f;
        }
        std::swap(d_x_old, d_x_new);

        q.submit([&](sycl::handler &cgh) {
             auto reduction = sycl::reduction(err_buf, cgh, sycl::maximum<>());

             cgh.parallel_for(sycl::range<1>(n), reduction,
                              [=](sycl::id<1> i, auto &err_max) {
                                  float d = 0.0f;

                                  int j = 0;
                                  for (; j < i; j++) {
                                      d += d_a[i * n + j] * d_x_old[j];
                                  }
                                  j++;
                                  for (; j < n; j++) {
                                      d += d_a[i * n + j] * d_x_old[j];
                                  }

                                  float x = (d_b[i] - d) / d_a_diag[i];
                                  d_x_new[i] = x;

                                  err_max.combine(sycl::fabs(x - d_x_old[i]));
                              });
         }).wait();

        {
            auto error = err_buf.get_host_access();
            err = error[0];
        }

        k--;
    } while (err >= accuracy && k > 0);

    q.memcpy(x_new.data(), d_x_new, sizeof(float) * n).wait();

    sycl::free(d_a, q);
    sycl::free(d_a_diag, q);
    sycl::free(d_b, q);
    sycl::free(d_x_old, q);
    sycl::free(d_x_new, q);

    return x_new;
}
