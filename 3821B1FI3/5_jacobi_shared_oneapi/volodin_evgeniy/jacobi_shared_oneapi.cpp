#include "jacobi_shared_oneapi.h"

std::vector<float> JacobiSharedONEAPI(const std::vector<float> a,
                                      const std::vector<float> b,
                                      float accuracy, sycl::device device) {
  int n = b.size();
  sycl::queue q(device);

  float *A = sycl::malloc_shared<float>(n * n, q);
  float *B = sycl::malloc_shared<float>(n, q);
  float *x[2];
  x[0] = sycl::malloc_shared<float>(n, q);
  x[1] = sycl::malloc_shared<float>(n, q);

  q.memcpy(A, a.data(), sizeof(float) * n * n).wait();
  q.memcpy(B, b.data(), sizeof(float) * n).wait();
  q.memset(x[0], 0, sizeof(float) * n).wait();
  q.memset(x[1], 0, sizeof(float) * n).wait();

  int *last = sycl::malloc_shared<int>(1, q);
  q.memset(last, 0, sizeof(float));
  float *norm2 = sycl::malloc_shared<float>(1, q);
  int iter = 0;

  do {
    iter++;

    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> idx) {
       int i = idx[0];
       float sigma = 0.0f;
       for (int j = 0; j < n; ++j) {
         if (i != j)
           sigma += A[i * n + j] * x[*last][j];
       }
       x[1 - *last][i] = (B[i] - sigma) / A[i * n + i];
     }).wait();

    q.single_task([=]() {
       float res = 0;
       for (int i = 0; i < n; i++) {
         float diff = x[*last][i] - x[1 - *last][i];
         res += diff * diff;
       }
       *norm2 = res;
       *last = 1 - *last;
     }).wait();

  } while (iter < ITERATIONS && *norm2 >= accuracy * accuracy);

  std::vector<float> result(n);
  q.memcpy(result.data(), x[*last], sizeof(float) * n).wait();

  sycl::free(A, q);
  sycl::free(B, q);
  sycl::free(x[0], q);
  sycl::free(x[1], q);
  sycl::free(last, q);
  sycl::free(norm2, q);

  return result;
}