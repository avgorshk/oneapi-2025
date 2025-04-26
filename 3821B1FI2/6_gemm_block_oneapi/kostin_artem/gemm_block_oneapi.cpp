#include "gemm_block_oneapi.h"
#include <usm.hpp>
#include <vector>

std::vector<float> GemmBlockONEAPI(const std::vector<float> a, const std::vector<float> b,
                                   size_t size, sycl::device device) {
    constexpr size_t BLOCK_SIZE = 16;
    size_t total = size * size;
    std::vector<float> result(total, 0.0f);

    sycl::queue q(device);

    float *A = sycl::malloc_device<float>(total, q);
    float *B = sycl::malloc_device<float>(total, q);
    float *C = sycl::malloc_device<float>(total, q);

    q.memcpy(A, a.data(), total * sizeof(float));
    q.memcpy(B, b.data(), total * sizeof(float));

    sycl::range<2> global{(size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE,
                          (size + BLOCK_SIZE - 1) / BLOCK_SIZE * BLOCK_SIZE};
    sycl::range<2> local{BLOCK_SIZE, BLOCK_SIZE};

    q.submit([&](sycl::handler &h) {
        sycl::local_accessor<float, 2> A_block({BLOCK_SIZE, BLOCK_SIZE}, h);
        sycl::local_accessor<float, 2> B_block({BLOCK_SIZE, BLOCK_SIZE}, h);

        h.parallel_for(sycl::nd_range<2>(global, local), [=](sycl::nd_item<2> item) {
            int gy = item.get_global_id(0);
            int gx = item.get_global_id(1);
            int ly = item.get_local_id(0);
            int lx = item.get_local_id(1);
            int group_count = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;

            float sum = 0.0f;

            for (int block = 0; block < group_count; ++block) {
                int a_col = block * BLOCK_SIZE + lx;
                int b_row = block * BLOCK_SIZE + ly;

                A_block[ly][lx] = (gy < size && a_col < size) ? A[gy * size + a_col] : 0.0f;
                B_block[ly][lx] = (b_row < size && gx < size) ? B[b_row * size + gx] : 0.0f;

                item.barrier(sycl::access::fence_space::local_space);

                for (int k = 0; k < BLOCK_SIZE; ++k)
                    sum += A_block[ly][k] * B_block[k][lx];

                item.barrier(sycl::access::fence_space::local_space);
            }

            if (gy < size && gx < size)
                C[gy * size + gx] = sum;
        });
    });
    q.wait();

    q.memcpy(result.data(), C, total * sizeof(float)).wait();

    sycl::free(A, q);
    sycl::free(B, q);
    sycl::free(C, q);

    return result;
}
