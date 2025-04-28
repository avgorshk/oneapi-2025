#include "jacobi_shared_oneapi.h"

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device) {
  static_assert(ITERATIONS >= 1, "More than 1 iteration");

  const size_t n = b.size();

  assert(n > 0);
  assert(a.size() == n * n);
  assert(accuracy >= 0.0f);

  int attempt = 0;
  std::vector<float> res(n);
  {
    sycl::queue queue(device);
    float* a_shared 	   = sycl::malloc_shared<float>(n * n, queue),
		   b_shared 	   = sycl::malloc_shared<float>(n, queue),
		   res_shared 	   = sycl::malloc_shared<float>(n, queue),
		   res_prev_shared = sycl::malloc_shared<float>(n, queue),
		   error_shared    = sycl::malloc_shared<float>(1, queue);

    queue.memcpy(a_shared, a.data(), n * n * sizeof(float));
    queue.memcpy(b_shared, b.data(), n * sizeof(float));
    queue.memset(res_shared, 0.0f, n * sizeof(float));
    queue.memset(error_shared, 0.0f, sizeof(float));
    queue.wait();

    while (attempt < ITERATIONS) {
      std::swap(res_shared, res_prev_shared);
      queue.submit([&](sycl::handler& handler) {
        auto reduction = sycl::reduction(error_shared, sycl::maximum<float>());

        handler.parallel_for(sycl::range<1>(n), reduction,
							 [=](sycl::id<1> id, auto& error) {
          const size_t i = id.get(0);
          float x = b_shared[i];
          for (size_t j = 0; j < n; j++)
            if (i != j)
              x -= a_shared[i * n + j] * res_prev_shared[j];
          x /= a_shared[i * n + i];
          res_shared[i] = x;
          error.combine(sycl::fabs(x - res_prev_shared[i]));
        });
      }).wait();

      if (*error_shared < accuracy)
        break;

      *error_shared = 0.0f;
      ++attempt;
    }

    queue.memcpy(res.data(), res_shared, n * sizeof(float)).wait();

    sycl::free(a_shared, queue);
    sycl::free(b_shared, queue);
    sycl::free(res_shared, queue);
    sycl::free(res_prev_shared, queue);
    sycl::free(error_shared, queue);
  }
  return res;
}