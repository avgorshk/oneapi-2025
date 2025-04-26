#include "jacobi_dev_oneapi.h"

std::vector<float> JacobiDevONEAPI(const std::vector<float> a,
                                   const std::vector<float> b, float accuracy,
                                   sycl::device device) {

  const int n = b.size();
  sycl::queue q(device);

  float *d_a = sycl::malloc_device<float>(n * n, q);
  float *d_b = sycl::malloc_device<float>(n, q);
  float *d_x[2] = {sycl::malloc_device<float>(n, q),
                   sycl::malloc_device<float>(n, q)};
  float *d_diff = sycl::malloc_device<float>(1, q);

  q.memcpy(d_a, a.data(), sizeof(float) * n * n).wait();
  q.memcpy(d_b, b.data(), sizeof(float) * n).wait();
  q.memset(d_x[0], 0, sizeof(float) * n).wait();
  q.memset(d_x[1], 0, sizeof(float) * n).wait();

  int last = 0;
  float norm2 = 0.0f;
  int iter = 0;

  while (iter < ITERATIONS) {
    iter++;

    q.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
      float sigma = 0.0f;
      for (int j = 0; j < n; j++) {
        if (j != i[0])
          sigma += d_a[i[0] * n + j] * d_x[last][j];
      }
      d_x[1 - last][i] = (d_b[i] - sigma) / d_a[i[0] * n + i[0]];
    });

    q.submit([&](sycl::handler &h) {
      h.single_task([=]() {
        float sum = 0.0f;
        for (int i = 0; i < n; ++i) {
          float diff = d_x[1 - last][i] - d_x[last][i];
          sum += diff * diff;
        }
        *d_diff = sum;
      });
    });

    q.memcpy(&norm2, d_diff, sizeof(float)).wait();

    if (norm2 < accuracy * accuracy)
      break;

    last = 1 - last;
  }

  std::vector<float> result(n);
  q.memcpy(result.data(), d_x[last], sizeof(float) * n).wait();

  sycl::free(d_a, q);
  sycl::free(d_b, q);
  sycl::free(d_x[0], q);
  sycl::free(d_x[1], q);
  sycl::free(d_diff, q);

  return result;
}
