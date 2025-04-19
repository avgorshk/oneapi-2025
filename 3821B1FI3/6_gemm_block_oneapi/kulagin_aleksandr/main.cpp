// Copyright (c) 2025 Kulagin Aleksandr
#include "gemm_block_oneapi.h"
#define _USE_MATH_DEFINES
#include <iostream>
#include <iomanip>
#include <stdexcept>
#include <cmath>

static std::vector<float> matrix_mul(const std::vector<float> &a, const std::vector<float> &b, size_t n) {
  assert(n > 0);
  assert(a.size() == n * n);
  assert(b.size() == n * n);
  std::vector<float> res(n * n, 0.0f);
  for (size_t i = 0; i < n; i++) {
    for (size_t j = 0; j < n; j++) {
      float& res_ij = res[i * n + j];
      for (size_t k = 0; k < n; k++) {
        res_ij += a[i * n + k] * b[k * n + j];
      }
    }
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
  constexpr float acc = 0.002f;
  const size_t n = 64;
  const size_t sz = n * n;
  std::vector<float> a(sz);
  std::vector<float> b(sz);
  for (int i = 0; i < sz; i++) {
    a[i] = (float)((i + 1) % n) / M_PIf; // meh
    b[i] = (float)((i - 1) % n) / M_PIf;
  }
  std::vector<float> real = matrix_mul(a, b, n);
  std::vector<float> test = GemmBlockONEAPI(a, b, n, getFirstDevice());
  std::cout << std::setprecision(16);
  for (size_t i = 0; i < sz; i++) {
    float error = std::abs(real[i] - test[i]);
    std::cout << real[i] << ' ' << test[i] << ' ' << error << '\n';
    bool is_ok = error < acc;
    assert(is_ok && "Test failed, difference exceeds defined accuracy");
    if (!is_ok) {
      throw std::runtime_error("Test failed, difference exceeds " + std::to_string(acc));
    }
  }
  return 0;
}
