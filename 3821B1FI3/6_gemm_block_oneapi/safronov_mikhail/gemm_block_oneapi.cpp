#include "gemm_block_oneapi.h"
#include <cassert>

std::vector<float> GemmBlockONEAPI(
    const std::vector<float>& a, const std::vector<float>& b,
    size_t size, sycl::device device) {
    constexpr size_t BLOCK_SIZE = 16;
    assert(size % BLOCK_SIZE == 0);

    sycl::queue queue(device);
    std::vector<float> c(size * size, 0.0f);

    float* dev_a = sycl::malloc_device<float>(size * size, queue);
    float* dev_b = sycl::malloc_device<float>(size * size, queue);
    float* dev_c = sycl::malloc_device<float>(size * size, queue);

    queue.memcpy(dev_a, a.data(), size * size * sizeof(float)).wait();
    queue.memcpy(dev_b, b.data(), size * size * sizeof(float)).wait();

    sycl::range<2> global_range(size, size);
    sycl::range<2> local_range(BLOCK_SIZE, BLOCK_SIZE);

    queue.submit([&](sycl::handler& cgh) {
        sycl::local_accessor<float, 2> block_a({BLOCK_SIZE, BLOCK_SIZE}, cgh);
        sycl::local_accessor<float, 2> block_b({BLOCK_SIZE, BLOCK_SIZE}, cgh);

        cgh.parallel_for(sycl::nd_range<2>(global_range, local_range),
                         [=](sycl::nd_item<2> item) {
                             size_t global_row = item.get_global_id(0);
                             size_t global_col = item.get_global_id(1);
                             size_t local_row = item.get_local_id(0);
                             size_t local_col = item.get_local_id(1);

                             float sum = 0.0f;

                             for (size_t block = 0; block < size / BLOCK_SIZE; ++block) {
                                 block_a[local_row][local_col] =
                                     dev_a[global_row * size + block * BLOCK_SIZE + local_col];
                                 block_b[local_row][local_col] =
                                     dev_b[(block * BLOCK_SIZE + local_row) * size + global_col];

                                 item.barrier(sycl::access::fence_space::local_space);

                                 for (size_t k = 0; k < BLOCK_SIZE; ++k) {
                                     sum += block_a[local_row][k] * block_b[k][local_col];
                                 }

                                 item.barrier(sycl::access::fence_space::local_space);
                             }

                             dev_c[global_row * size + global_col] = sum;
                         });
    }).wait();

    queue.memcpy(c.data(), dev_c, size * size * sizeof(float)).wait();

    sycl::free(dev_a, queue);
    sycl::free(dev_b, queue);
    sycl::free(dev_c, queue);

    return c;
}