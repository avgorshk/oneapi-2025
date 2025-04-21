#include "jacobi_acc_oneapi.h"

#include <buffer.hpp>
#include <handler.hpp>
#include <range.hpp>
#include <reduction.hpp>
#include <vector>

std::vector<float> JacobiAccONEAPI(const std::vector<float> a, const std::vector<float> b, float eps, sycl::device dev) {
    int n = b.size(), step = 0;
    std::vector<float> x(n, 0), x_old(n, 0);
    float err = 0;

    sycl::buffer<float> bufA(a.data(), a.size()), bufB(b.data(), n), bufX(x.data(), n), bufOld(x_old.data(), n), bufErr(&err, 1);
    sycl::queue q(dev);

    while (step++ < ITERATIONS) {
        q.submit([&](sycl::handler& h) {
            auto A = bufA.get_access<sycl::access::mode::read>(h);
            auto B = bufB.get_access<sycl::access::mode::read>(h);
            auto X = bufX.get_access<sycl::access::mode::read_write>(h);
            auto Old = bufOld.get_access<sycl::access::mode::read_write>(h);
            auto Red = sycl::reduction(bufErr, h, sycl::maximum<>());

            h.parallel_for(sycl::range<1>(n), Red, [=](sycl::id<1> id, auto& maxErr) {
                int i = id[0];
                float sum = B[i];
                for (int j = 0; j < n; ++j)
                    if (j != i) sum -= A[i * n + j] * Old[j];
                float xi = sum / A[i * n + i];
                X[i] = xi;
                maxErr.combine(sycl::fabs(xi - Old[i]));
            });
        });
        q.wait();

        if (bufErr.get_host_access()[0] < eps) break;
        bufErr.get_host_access()[0] = 0;

        auto xNew = bufX.get_host_access();
        auto xOldHost = bufOld.get_host_access();
        for (int i = 0; i < n; i++) xOldHost[i] = xNew[i];
    }
    return x;
}
