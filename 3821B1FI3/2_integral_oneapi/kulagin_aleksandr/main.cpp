// Copyright (c) 2025 Kulagin Aleksandr
#include "integral_oneapi.h"

#include <cmath>
#include <iostream>
#include <stdexcept>
#include <vector>
#include <cassert>
#include <iomanip>

static float integral_test(float start, float end, int count) {
  assert(count > 0);
  float dx = (end - start) / static_cast<float>(count);
  float sum = 0.0f;
  for (int j = 0; j < count; j++) {
    for (int i = 0; i < count; i++) {
      const float x_i = start + dx * i;
      const float y_j = start + dx * j;
      const float x_i_1 = start + dx * (i + 1);
      const float y_j_1 = start + dx * (j + 1);
      sum += std::sin((x_i + x_i_1) / 2.0f) * std::cos((y_j + y_j_1) / 2.0f) * (x_i_1 - x_i) * (y_j_1 - y_j);
    }
  }
  return sum;
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
  const float test_count = static_cast<int>(std::sqrt(65536));
  float real = integral_test(0, 1, test_count);
  float test = IntegralONEAPI(0, 1, test_count, getFirstDevice());
  std::cout << std::setprecision(16) << real << ' ' << test << '\n';
  return 0;
}
