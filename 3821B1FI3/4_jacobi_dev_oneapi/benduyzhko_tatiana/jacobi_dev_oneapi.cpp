#include "jacobi_dev_oneapi.h"

std::vector<float> JacobiDevONEAPI(
    const std::vector<float> a, const std::vector<float> b,
    float accuracy, sycl::device device) {
    int n = b.size();
    sycl::queue q(device);
    float* dev_a = sycl::malloc_device<float>(n * n, q);
    float* dev_b = sycl::malloc_device<float>(n, q);
    q.memcpy(dev_a, a.data(), n * n * sizeof(float)).wait();
    q.memcpy(dev_b, b.data(), n * sizeof(float)).wait();
    float* dev_ans[2];
    dev_ans[0] = sycl::malloc_device<float>(n, q);
    dev_ans[1] = sycl::malloc_device<float>(n, q);
    q.memset(dev_ans[0], 0, n * sizeof(float));
    q.memset(dev_ans[1], 0, n * sizeof(float));
    int* dev_last = sycl::malloc_device<int>(1, q);
    q.memset(dev_last, 0, sizeof(float));
    float* dev_norm2 = sycl::malloc_device<float>(1, q);
    int iterations = 0;
    float host_norm2 = 0;
    do {
        iterations++;
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            int i = id.get(0);
            float res = 0;
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    res += dev_b[j];
                }
                else {
                    res -= dev_a[i * n + j] * dev_ans[*dev_last][j];
                }
            }
            res /= dev_a[i * n + i];
            dev_ans[1 - *dev_last][i] = res;
        }).wait();
        q.single_task([=]() {
            float res = 0;
            for (int i = 0; i < n; i++) {
                float d = dev_ans[0][i] - dev_ans[1][i];
                res += d * d;
            }
            *dev_norm2 = res;
            *dev_last = 1 - *dev_last;
        }).wait();
        q.memcpy(&host_norm2, dev_norm2, sizeof(float)).wait();
    } while (iterations < ITERATIONS && host_norm2 >= accuracy * accuracy);

    std::vector<float> ans(n);
    int host_last;
    q.memcpy(&host_last, dev_last, sizeof(int)).wait();
    q.memcpy(ans.data(), dev_ans[host_last], n * sizeof(float));

    sycl::free(dev_a, q);
    sycl::free(dev_b, q);
    sycl::free(dev_ans[0], q);
    sycl::free(dev_ans[1], q);
    sycl::free(dev_last, q);
    sycl::free(dev_norm2, q);

    return ans;
}
