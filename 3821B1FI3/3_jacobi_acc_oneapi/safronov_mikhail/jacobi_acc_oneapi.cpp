#include "jacobi_acc.h"
#include <cmath>
#include <cassert>

std::vector<float> JacobiAcc(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device) {
    const size_t n = b.size();
    assert(a.size() == n * n);

    std::vector<float> x_old(n, 0.0f);
    std::vector<float> x_new(n, 0.0f);

    sycl::queue queue(device);

    sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(n * n));
    sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(n));
    sycl::buffer<float, 1> x_old_buf(x_old.data(), sycl::range<1>(n));
    sycl::buffer<float, 1> x_new_buf(x_new.data(), sycl::range<1>(n));

    bool converged = false;
    for (int iter = 0; iter < 1024 && !converged; ++iter) {
        queue.submit([&](sycl::handler& cgh) {
            auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
            auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_old_acc = x_old_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_new_acc = x_new_buf.get_access<sycl::access::mode::write>(cgh);

            cgh.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                float sigma = 0.0f;
                for (size_t j = 0; j < n; ++j) {
                    if (j != i) {
                        sigma += a_acc[i * n + j] * x_old_acc[j];
                    }
                }
                x_new_acc[i] = (b_acc[i] - sigma) / a_acc[i * n + i];
            });
        }).wait();

        converged = true;
        queue.submit([&](sycl::handler& cgh) {
            auto x_old_acc = x_old_buf.get_access<sycl::access::mode::read>(cgh);
            auto x_new_acc = x_new_buf.get_access<sycl::access::mode::read>(cgh);

            cgh.parallel_for(sycl::range<1>(n), [=, &converged](sycl::id<1> i) {
                if (std::fabs(x_new_acc[i] - x_old_acc[i]) > accuracy) {
                    converged = false;
                }
            });
        }).wait();

        std::swap(x_old_buf, x_new_buf);
    }

    sycl::host_accessor x_new_host(x_new_buf, sycl::read_only);
    return std::vector<float>(x_new_host.begin(), x_new_host.end());
}