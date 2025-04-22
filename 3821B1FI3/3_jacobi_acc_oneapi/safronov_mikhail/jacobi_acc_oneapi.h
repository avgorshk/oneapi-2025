#ifndef __JACOBI_ACC_H
#define __JACOBI_ACC_H

#include <vector>
#include <sycl/sycl.hpp>

std::vector<float> JacobiAcc(
    const std::vector<float>& a, const std::vector<float>& b,
    float accuracy, sycl::device device);

#endif  // __JACOBI_ACC_H