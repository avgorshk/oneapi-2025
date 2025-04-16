#include "jacobi_shared_oneapi.h"

std::vector<float> JacobiSharedONEAPI(
    const std::vector<float> a, const std::vector<float> b,
    float accuracy, sycl::device device) {
    int n = b.size();
    sycl::queue q(device);
    float* shared_a = sycl::malloc_shared<float>(n * n, q);
    float* shared_b = sycl::malloc_shared<float>(n, q);
    q.memcpy(shared_a, a.data(), n * n * sizeof(float)).wait();
    q.memcpy(shared_b, b.data(), n * sizeof(float)).wait();
    float* shared_ans[2];
    shared_ans[0] = sycl::malloc_shared<float>(n, q);
    shared_ans[1] = sycl::malloc_shared<float>(n, q);
    q.memset(shared_ans[0], 0, n * sizeof(float));
    q.memset(shared_ans[1], 0, n * sizeof(float));
    int* shared_last = sycl::malloc_shared<int>(1, q);
    q.memset(shared_last, 0, sizeof(float));
    float* shared_norm2 = sycl::malloc_shared<float>(1, q);
    int iterations = 0;
    do {
        iterations++;
        q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> id) {
            int i = id.get(0);
            float res = 0;
            for (int j = 0; j < n; j++) {
                if (i == j) {
                    res += shared_b[j];
                }
                else {
                    res -= shared_a[i * n + j] * shared_ans[*shared_last][j];
                }
            }
            res /= shared_a[i * n + i];
            shared_ans[1 - *shared_last][i] = res;
        }).wait();
        q.single_task([=]() {
            float res = 0;
            for (int i = 0; i < n; i++) {
                float d = shared_ans[0][i] - shared_ans[1][i];
                res += d * d;
            }
            *shared_norm2 = res;
            *shared_last = 1 - *shared_last;
        }).wait();
    } while (iterations < ITERATIONS && *shared_norm2 >= accuracy * accuracy);

    std::vector<float> ans(n);
    q.memcpy(ans.data(), shared_ans[*shared_last], n * sizeof(float));

    sycl::free(shared_a, q);
    sycl::free(shared_b, q);
    sycl::free(shared_ans[0], q);
    sycl::free(shared_ans[1], q);
    sycl::free(shared_last, q);
    sycl::free(shared_norm2, q);

    return ans;
}
