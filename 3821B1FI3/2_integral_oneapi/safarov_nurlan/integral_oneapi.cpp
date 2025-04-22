#include "integral_oneapi.h"
#include <range.hpp>
#include <reduction.hpp>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    float integralResult = 0.0f;
    float stepSize = (end - start) / count;
    
    {
        sycl::buffer<float> resultBuffer(&integralResult, 1);
    
        sycl::queue computationQueue(device);
    
        computationQueue.submit([&](sycl::handler &commandGroupHandler) {
            auto reductionOperation = sycl::reduction(resultBuffer, commandGroupHandler, sycl::plus<>());
    
            commandGroupHandler.parallel_for(
                sycl::range<2>(count, count), reductionOperation,
                [=](sycl::id<2> index, auto &reductionSum) {
                    float xStart = start + stepSize * index.get(0);
                    float xEnd = start + stepSize * (index.get(0) + 1);
                    float yStart = start + stepSize * index.get(1);
                    float yEnd = start + stepSize * (index.get(1) + 1);
                    reductionSum += sycl::sin((xStart + xEnd) * 0.5f) *
                                   sycl::cos((yStart + yEnd) * 0.5f) * (xEnd - xStart) * (yEnd - yStart);
                });
        });
    
        computationQueue.wait();
    }

  return ans;
}