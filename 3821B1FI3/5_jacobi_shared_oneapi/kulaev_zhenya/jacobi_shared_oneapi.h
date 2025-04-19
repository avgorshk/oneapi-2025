#ifndef __JACOBI_SHARED_ONEAPI_H
#define __JACOBI_SHARED_ONEAPI_H

#include <sycl/sycl.hpp>
#include <vector>

constexpr auto ITERATIONS = 1024;

std::vector<float> JacobiSharedONEAPI(const std::vector<float> a,
                                      const std::vector<float> b,
                                      float accuracy, sycl::device device);

#endif  // __JACOBI_SHARED_ONEAPI_H