#include "gemm_block_oneapi.h"

#include <sycl/sycl.hpp>
#include <vector>

std::vector<float> GemmBlockONEAPI(const std::vector<float>& a,
                                  const std::vector<float>& b, 
                                  size_t size, 
                                  sycl::device device) {
                                                                                         
    constexpr size_t block_size = 32;
    size_t total_elements = size * size;
    std::vector<float> result(total_elements, 0.0f);

    sycl::queue queue(device, sycl::property::queue::in_order{});

    float *devA = sycl::malloc_device<float>(total_elements, queue);
    float *devB = sycl::malloc_device<float>(total_elements, queue);
    float *devC = sycl::malloc_device<float>(total_elements, queue);

    auto copyA = queue.memcpy(devA, a.data(), total_elements * sizeof(float));
    auto copyB = queue.memcpy(devB, b.data(), total_elements * sizeof(float));
    
    copyA.wait();
    copyB.wait();

    sycl::range<2> global_range(size, size);
    sycl::range<2> local_range(block_size, block_size);

    queue.submit([&](sycl::handler &cgh) {
        sycl::local_accessor<float, 2> block_a(sycl::range<2>(block_size, block_size), cgh);
        sycl::local_accessor<float, 2> block_b(sycl::range<2>(block_size, block_size), cgh);

        cgh.parallel_for<class GemmKernel>(
            sycl::nd_range<2>(global_range, local_range),
            [=](sycl::nd_item<2> item) {
                int global_row = item.get_global_id(0);
                int global_col = item.get_global_id(1);
                int local_row = item.get_local_id(0);
                int local_col = item.get_local_id(1);
                
                float sum = 0.0f;
                
                int num_blocks = size / block_size;
                
                for (int block_idx = 0; block_idx < num_blocks; ++block_idx) {

                    int a_col = block_idx * block_size + local_col;
                    block_a[local_row][local_col] = 
                        (global_row < size && a_col < size) 
                        ? devA[global_row * size + a_col] 
                        : 0.0f;
                    
                    int b_row = block_idx * block_size + local_row;
                    block_b[local_row][local_col] = 
                        (b_row < size && global_col < size) 
                        ? devB[b_row * size + global_col] 
                        : 0.0f;
                    
                    item.barrier(sycl::access::fence_space::local_space);
                    
                    for (int k = 0; k < block_size; ++k) {
                        sum += block_a[local_row][k] * block_b[k][local_col];
                    }
                    
                    item.barrier(sycl::access::fence_space::local_space);
                }
                
                // Сохранение результата
                if (global_row < size && global_col < size) {
                    devC[global_row * size + global_col] = sum;
                }
            });
    });

    queue.memcpy(result.data(), devC, total_elements * sizeof(float)).wait();

    sycl::free(devA, queue);
    sycl::free(devB, queue);
    sycl::free(devC, queue);

    return result;
}