#include "gemm_block_oneapi.h"

constexpr size_t BLOCK_SIZE = 16;

std::vector<float> GemmBlockONEAPI(
    const std::vector<float> a, const std::vector<float> b,
    size_t size, sycl::device device) {
    
    std::vector<float> c(size * size, 0.0f);
    sycl::queue queue(device);

    float* a_dev = sycl::malloc_device<float>(size * size, queue);
    float* b_dev = sycl::malloc_device<float>(size * size, queue);
    float* c_dev = sycl::malloc_device<float>(size * size, queue);

    queue.memcpy(a_dev, a.data(), sizeof(float) * size * size).wait();
    queue.memcpy(b_dev, b.data(), sizeof(float) * size * size).wait();
    queue.memset(c_dev, 0, sizeof(float) * size * size).wait();

    sycl::range<2> global(size, size);
    sycl::range<2> local(BLOCK_SIZE, BLOCK_SIZE);   

    queue.submit([&](sycl::handler& h) {
        sycl::local_accessor<float, 2> a_tile({ BLOCK_SIZE, BLOCK_SIZE }, h);
        sycl::local_accessor<float, 2> b_tile({ BLOCK_SIZE, BLOCK_SIZE }, h);

        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            size_t row = item.get_global_id(0);
            size_t col = item.get_global_id(1);

            float sum = 0.0f;
            for (size_t block = 0; block < size / BLOCK_SIZE; ++block) {

                a_tile[item.get_local_id(0)][item.get_local_id(1)] =
                    a_dev[row * size + block * BLOCK_SIZE + item.get_local_id(1)];

                b_tile[item.get_local_id(0)][item.get_local_id(1)] =
                    b_dev[(block * BLOCK_SIZE + item.get_local_id(0)) * size + col];

                item.barrier(sycl::access::fence_space::local_space);

                for (size_t k = 0; k < BLOCK_SIZE; ++k) {
                    sum += a_tile[item.get_local_id(0)][k] * b_tile[k][item.get_local_id(1)];
                }

                item.barrier(sycl::access::fence_space::local_space);
            }

            c_dev[row * size + col] = sum;
            });
        }).wait();

        queue.memcpy(c.data(), c_dev, sizeof(float) * size * size).wait();

        sycl::free(a_dev, queue);
        sycl::free(b_dev, queue);
        sycl::free(c_dev, queue);

        return c;
}