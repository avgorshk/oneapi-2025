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
                                       
  std::vector<float> ansCurr(b.size(), 0.0f);
  std::vector<float> ansPrev(b.size(), 0.0f);

  int size = b.size();
  int it = 0;
  float err = 0.0f;

  {
    sycl::buffer<float, 1> bufA(a.data(), a.size());
    sycl::buffer<float, 1> bufB(b.data(), b.size());
    sycl::buffer<float, 1> bufCurr(ansCurr.data(), ansCurr.size());
    sycl::buffer<float, 1> bufPrev(ansPrev.data(), ansPrev.size());
    sycl::buffer<float, 1> bufErr(&err, 1);

    while (it++ < ITERATIONS) {

      sycl::queue queue(device);

      queue.submit([&](sycl::handler &cgh) {

        auto inA = bufA.get_access<sycl::access::mode::read>(cgh);
        auto inB = bufB.get_access<sycl::access::mode::read>(cgh);
        auto inPrev = bufPrev.get_access<sycl::access::mode::read_write>(cgh);
        auto inCurr = bufCurr.get_access<sycl::access::mode::read_write>(cgh);

        auto reduction = sycl::reduction(bufErr, cgh, sycl::maximum<>());

        cgh.parallel_for(sycl::range<1>(size), reduction,
                         [=](sycl::id<1> id, auto &err) {
                           int i = id.get(0);
                           float curr = inB[i];
                           for (int j = 0; j < size; j++) {
                             if (i != j) {
                               curr -= inA[i * size + j] * inPrev[j];
                             }
                           }
                           curr /= inA[i * size + i];
                           inCurr[i] = curr;

                           float diff = sycl::fabs(curr - inPrev[i]);
                           err.combine(diff);
                         });
      });

      queue.wait();

      {
        auto err = bufErr.get_host_access();
        if (err[0] < accuracy)
          break;
        err[0] = 0.0f;
      }

      {
        auto hostCurr = bufCurr.get_host_access();
        auto hostPrev = bufPrev.get_host_access();
        for (int i = 0; i < size; i++)
          hostPrev[i] = hostCurr[i];
      }
    }
  }

  return ansCurr;
}