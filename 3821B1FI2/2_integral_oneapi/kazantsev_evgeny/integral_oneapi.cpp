#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {

  float sum_integral = 0.0f;
  float dx = (end - start) / count;
  float dy = (end - start) / count;
  try {
    sycl::queue device_queue{device};
    sycl::buffer<float> buf_sum_integral(&sum_integral, sycl::range<1>(1));

    device_queue
        .submit([&](sycl::handler &cgh) {
          auto sumReduction =
              sycl::reduction(buf_sum_integral, cgh, sycl::plus<>());
          cgh.parallel_for<class Rieman_Integral_kernel>(
              sycl::range<2>(count, count), sumReduction,
              [=](sycl::item<2> item, auto &sum_integral) {
                float x_mid = start + (item[0] + 0.5f) * dx;
                float y_mid = start + (item[1] + 0.5f) * dy;

                sum_integral += sycl::sin(x_mid) * sycl::cos(y_mid) * dx * dy;
              });
        })
        .wait();
  } catch (sycl::exception e) {
    std::cerr << e.what() << std::endl;
  } catch (...) {
    std::cerr << "Global Error!" << std::endl;
  }

  return sum_integral;
}