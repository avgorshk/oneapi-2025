#include "jacobi_dev_oneapi.h"

std::vector<float> JacobiDevONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device) {
  static_assert(ITERATIONS >= 1, "More than 1 iteration");

  const size_t n = b.size();

  assert(n > 0);
  assert(a.size() == n * n);
  assert(accuracy >= 0.0f);

  std::vector<float> res(n, 0.0f);
  std::vector<float> res_prev(res);

  int attempt = 0;
  float error = 0.0f;
  {
    sycl::queue queue(device);
    float* a_dev 		= sycl::malloc_device<float>(n * n, queue),
		   b_dev 	    = sycl::malloc_device<float>(n, queue),
		   res_dev      = sycl::malloc_device<float>(n, queue),
		   res_prev_dev = sycl::malloc_device<float>(n, queue),
		   error_dev    = sycl::malloc_device<float>(1, queue);

    queue.memcpy(a_dev, a.data(), n * n * sizeof(float));
    queue.memcpy(b_dev, b.data(), n * sizeof(float));

    queue.memset(res_dev, 0.0f, n * sizeof(float));
    queue.memset(error_dev, 0.0f, sizeof(float));

    queue.wait();
    while (attempt < ITERATIONS) {
      std::swap(res_dev, res_prev_dev);

      queue.submit([&](sycl::handler& handler) {
        auto reduction = sycl::reduction(error_dev, sycl::maximum<float>());

        handler.parallel_for(sycl::range<1>(n), reduction,
							 [=](sycl::id<1> id, auto& error) {
          const size_t i = id.get(0);
          float x = b_dev[i];

          for (size_t j = 0; j < n; j++)
            if (i != j)
              x -= a_dev[i * n + j] * res_prev_dev[j];

          x /= a_dev[i * n + i];
          res_dev[i] = x;
          error.combine(sycl::fabs(x - res_prev_dev[i]));
        });
      }).wait();

      queue.memcpy(&error, error_dev, sizeof(float)).wait();
      if (error < accuracy)
        break;

      queue.memset(error_dev, 0.0f, sizeof(float)).wait();
      ++attempt;
    }

    queue.memcpy(res.data(), res_dev, n * sizeof(float)).wait();

    sycl::free(a_dev, queue);
    sycl::free(b_dev, queue);
    sycl::free(res_dev, queue);
    sycl::free(res_prev_dev, queue);
    sycl::free(error_dev, queue);
  }
  return res;
}