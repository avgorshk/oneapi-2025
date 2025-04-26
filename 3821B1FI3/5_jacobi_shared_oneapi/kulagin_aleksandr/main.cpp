// Copyright (c) 2025 Kulagin Aleksandr
#include "jacobi_shared_oneapi.h"
#include <iostream>
#include <iomanip>
#include <stdexcept>

static std::vector<float> jacobi(const std::vector<float> a, const std::vector<float> b, float accuracy) {
  static_assert(ITERATIONS >= 1, "Must be more than 1 iteration");
  const size_t n = b.size();
  assert(n > 0);
  assert(a.size() == n * n);
  assert(accuracy >= 0.0f);
  std::vector<float> res(n, 0.0f);
  std::vector<float> res_prev(res);
  int attempt = 0;
  float error = 0.0f;
  while(attempt < ITERATIONS) {
    std::swap(res_prev, res);
    for (size_t i = 0; i < n; i++) {
      float g = b[i];
      for (size_t j = 0; j < n; j++) {
        if (i != j) {
          g -= a[i * n + j] * res_prev[j];
        }
      }
      g /= a[i * n + i];
      res[i] = g;
      error = std::max(error, std::abs(g - res_prev[i]));
    }
    if (error < accuracy) {
      break;
    }
    error = 0.0f;
    attempt++;
  }
  return res;
}

static std::vector<sycl::device> getDevices() {
  std::vector<sycl::platform> platforms = sycl::platform::get_platforms();
  std::vector<sycl::device> res;
  for (const auto& platform : platforms) {
    std::cout << "Platform: " << platform.get_info<sycl::info::platform::name>() << '\n';
    std::vector<sycl::device> devices = platform.get_devices();
    for (const auto& device : devices) {
      std::cout << "-- Device: " << device.get_info<sycl::info::device::name>() << '\n';
      res.push_back(device);
    }
  }
  return res;
}

static sycl::device getFirstDevice() {
  std::vector<sycl::device> devices = getDevices();
  if (devices.size() == 0) {
    throw std::runtime_error("Error no devices found");
  }
  sycl::device res = devices.at(0);
  std::cout << "Selected device: " << res.get_info<sycl::info::device::name>() << '\n';
  return res;
}


int main() {
  std::vector<float> a = {2, 1, 5, 7};
  std::vector<float> b = {11, 13};
  const float acc = 0.009f;
  const size_t n = b.size();
  assert(a.size() == n * n);
  std::vector<float> real = jacobi(a, b, acc);
  std::vector<float> test = JacobiSharedONEAPI(a, b, acc, getFirstDevice());
  std::cout << std::setprecision(16);
  for (size_t i = 0; i < n; i++) {
    float error = std::abs(real[i] - test[i]);
    std::cout << real[i] << ' ' << test[i] << ' ' << error << '\n';
    bool is_ok = error < acc;
    assert(is_ok);
    if (!is_ok) {
      throw std::runtime_error("Test failed, difference exceeds " + std::to_string(acc));
    }
  }
  return 0;
}
