#include "jacobi_acc_oneapi.h"

std::vector<float> JacobiAccONEAPI(const std::vector<float> a,
                                   const std::vector<float> b, float accuracy,
                                   sycl::device device) {
  size_t n = b.size();
  std::vector<float> x(n, 0.0f);
  std::vector<float> x_new(n, 0.0f);
  float max_diff = accuracy + 1.0f;
  size_t iter = 0;

  sycl::queue queue(device);

  sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(n * n));
  sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(n));
  sycl::buffer<float, 1> x_buf(x.data(), sycl::range<1>(n));
  sycl::buffer<float, 1> x_new_buf(x_new.data(), sycl::range<1>(n));
  sycl::buffer<float, 1> max_diff_buf(&max_diff, sycl::range<1>(1));

  x_buf.set_final_data(nullptr);
  x_new_buf.set_final_data(nullptr);
  max_diff_buf.set_final_data(nullptr);

  while (max_diff > accuracy && iter < ITERATIONS) {
    iter++;
    max_diff = 0.0f;

    queue.submit([&](sycl::handler &cgh) {
      auto a_acc = a_buf.get_access<sycl::access_mode::read>(cgh);
      auto b_acc = b_buf.get_access<sycl::access_mode::read>(cgh);
      auto x_acc = x_buf.get_access<sycl::access_mode::read>(cgh);
      auto x_new_acc = x_new_buf.get_access<sycl::access_mode::write>(cgh);

      cgh.parallel_for<class JacobiKernelOpt>(
          sycl::range<1>(n), [=](sycl::id<1> idx) {
            int i = idx[0];
            float sum = 0.0f;
            for (int j = 0; j < n; j++) {
              if (j != i)
                sum += a_acc[i * n + j] * x_acc[j];
            }
            x_new_acc[i] = (b_acc[i] - sum) / a_acc[i * n + i];
          });
    });

    queue.submit([&](sycl::handler &cgh) {
      auto x_acc = x_buf.get_access<sycl::access_mode::read>(cgh);
      auto x_new_acc = x_new_buf.get_access<sycl::access_mode::read>(cgh);
      auto max_diff_acc =
          max_diff_buf.get_access<sycl::access_mode::read_write>(cgh);

      cgh.parallel_for<class MaxDiffKernelOpt>(
          sycl::range<1>(n), [=](sycl::id<1> idx) {
            int i = idx[0];
            float diff = sycl::fabs(x_new_acc[i] - x_acc[i]);
            sycl::atomic_ref<float, sycl::memory_order::relaxed,
                             sycl::memory_scope::device,
                             sycl::access::address_space::global_space>
                atomic_max(max_diff_acc[0]);
            atomic_max.fetch_max(diff);
          });
    });

    queue.wait_and_throw();
    {
      auto host_acc = max_diff_buf.get_access<sycl::access_mode::read>();
      max_diff = host_acc[0];
    }

    queue.submit([&](sycl::handler &cgh) {
      auto x_acc = x_buf.get_access<sycl::access_mode::write>(cgh);
      auto x_new_acc = x_new_buf.get_access<sycl::access_mode::read>(cgh);

      cgh.parallel_for<class UpdateKernelOpt>(sycl::range<1>(n),
                                              [=](sycl::id<1> idx) {
                                                int i = idx[0];
                                                x_acc[i] = x_new_acc[i];
                                              });
    });
  }

  queue.wait_and_throw();
  std::vector<float> result(n);
  {
    sycl::host_accessor x_new_host(x_new_buf, sycl::read_only);
    for (size_t i = 0; i < n; ++i)
      result[i] = x_new_host[i];
  }

  return result;
}
