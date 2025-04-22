#include "jacobi_acc_oneapi.h"

std::vector<float> JacobiAccONEAPI(const std::vector<float> a,
                                   const std::vector<float> b, float accuracy,
                                   sycl::device device) {
  size_t n = b.size();
  std::vector<float> x(n, 0.0f);
  std::vector<float> x_new(n, 0.0f);

  sycl::queue queue(device);

  sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(n * n));
  sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(n));
  sycl::buffer<float, 1> x_buf(x.data(), sycl::range<1>(n));
  sycl::buffer<float, 1> x_new_buf(x_new.data(), sycl::range<1>(n));
  sycl::buffer<float, 1> max_diff_buf(sycl::range<1>(1));

  x_buf.set_final_data(nullptr);
  x_new_buf.set_final_data(nullptr);
  max_diff_buf.set_final_data(nullptr);

  float max_diff = accuracy + 1.0f;
  size_t iter = 0;

  while (max_diff > accuracy && iter < ITERATIONS) {
    ++iter;

    queue.submit([&](sycl::handler &cgh) {
      auto a_acc = a_buf.get_access<sycl::access_mode::read>(cgh);
      auto b_acc = b_buf.get_access<sycl::access_mode::read>(cgh);
      auto x_acc = x_buf.get_access<sycl::access_mode::read>(cgh);
      auto x_new_acc = x_new_buf.get_access<sycl::access_mode::write>(cgh);
      auto diff_acc = max_diff_buf.get_access<sycl::access_mode::write>(cgh);

      cgh.parallel_for<class JacobiKernel>(
          sycl::range<1>(n), [=](sycl::id<1> idx) {
            int i = idx[0];
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
              if (j != i)
                sum += a_acc[i * n + j] * x_acc[j];
            }
            float new_value = (b_acc[i] - sum) / a_acc[i * n + i];
            x_new_acc[i] = new_value;
            diff_acc[i] = sycl::fabs(new_value - x_acc[i]);
          });
    });

    queue.wait();

    {
      auto diff_host = max_diff_buf.get_access<sycl::access_mode::read>();
      max_diff = 0.0f;
      for (size_t i = 0; i < n; ++i) {
        max_diff = sycl::max(max_diff, diff_host[i]);
      }
    }

    std::swap(x_buf, x_new_buf);
  }

  std::vector<float> result(n);
  {
    sycl::host_accessor final_x(x_buf, sycl::read_only);
    for (size_t i = 0; i < n; ++i) {
      result[i] = final_x[i];
    }
  }
  return result;
}
