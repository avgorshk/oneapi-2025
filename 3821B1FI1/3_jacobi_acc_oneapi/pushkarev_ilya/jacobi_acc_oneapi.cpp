#include "jacobi_acc_oneapi.h"
#include <algorithm>
#include <buffer.hpp>
#include <handler.hpp>
#include <range.hpp>
#include <reduction.hpp>
#include <vector>

std::vector<float> JacobiAccONEAPI(const std::vector<float> a,
                                   const std::vector<float> b, float accuracy,
                                   sycl::device device) {
  const int size = b.size();
  std::vector<float> curr(size, 0.0f);
  std::vector<float> prev(size, 0.0f);
  float max_error = 0.0f;
  int step = 0;

  sycl::buffer<float> buf_a(a.data(), a.size());
  sycl::buffer<float> buf_b(b.data(), b.size());
  sycl::buffer<float> buf_curr(curr.data(), curr.size());
  sycl::buffer<float> buf_prev(prev.data(), prev.size());
  sycl::buffer<float> buf_error(&max_error, 1);

  sycl::queue q(device);

  while (step++ < ITERATIONS) {
    q.submit([&](sycl::handler& h) {
      auto A = buf_a.get_access<sycl::access::mode::read>(h);
      auto B = buf_b.get_access<sycl::access::mode::read>(h);
      auto Prev = buf_prev.get_access<sycl::access::mode::read_write>(h);
      auto Curr = buf_curr.get_access<sycl::access::mode::read_write>(h);
      auto red = sycl::reduction(buf_error, h, sycl::maximum<>());

      h.parallel_for(sycl::range<1>(size), red, [=](sycl::id<1> i, auto& err) {
        float sum = B[i];
        for (int j = 0; j < size; ++j) {
          if (i != j)
            sum -= A[i * size + j] * Prev[j];
        }
        sum /= A[i * size + i];
        Curr[i] = sum;
        err.combine(sycl::fabs(sum - Prev[i]));
      });
    });

    q.wait();

    auto err = buf_error.get_host_access();
    if (err[0] < accuracy)
      break;
    err[0] = 0.0f;

    // Обновляем prev <- curr
    auto curr_acc = buf_curr.get_host_access();
    auto prev_acc = buf_prev.get_host_access();
    std::copy(curr_acc.begin(), curr_acc.end(), prev_acc.begin());
  }

  return curr;
}
