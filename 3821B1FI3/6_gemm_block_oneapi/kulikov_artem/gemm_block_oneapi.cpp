#include "gemm_block_oneapi.h"

std::vector<float> GemmBlockONEAPI(
    const std::vector<float> a, const std::vector<float> b,
    size_t size, sycl::device device) {
    const int BLOCK_SIZE = 16;
    size_t C_size = size * size;
    std::vector<float> c(C_size);

    sycl::queue q(device);

    float *d_a = sycl::malloc_device<float>(C_size, q);
    float *d_b = sycl::malloc_device<float>(C_size, q);
    float *d_c = sycl::malloc_device<float>(C_size, q);

    q.memcpy(d_a, a.data(), sizeof(float) * C_size);
    q.memcpy(d_b, b.data(), sizeof(float) * C_size).wait();

    size_t num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
    sycl::range<2> global_range(num_blocks * BLOCK_SIZE, num_blocks * BLOCK_SIZE);
    sycl::range<2> local_range(BLOCK_SIZE, BLOCK_SIZE);

    q.submit([&](sycl::handler &cgh) {
         sycl::local_accessor<float, 2> aShared(local_range, cgh);
         sycl::local_accessor<float, 2> bShared(local_range, cgh);

         cgh.parallel_for(
             sycl::nd_range<2>(global_range, local_range),
             [=](sycl::nd_item<2> nd_item) {
                 size_t iGlob = nd_item.get_global_id(1);
                 size_t jGlob = nd_item.get_global_id(0);
                 size_t iLoc = nd_item.get_local_id(1);
                 size_t jLoc = nd_item.get_local_id(0);

                 float resCell = 0.0f;

                 for (int i = 0; i < num_blocks; ++i) {
                     aShared[jLoc][iLoc] = d_a[jGlob * size + i * BLOCK_SIZE + iLoc];
                     bShared[jLoc][iLoc] = d_b[(i * BLOCK_SIZE + jLoc) * size + iGlob];

                     nd_item.barrier(sycl::access::fence_space::local_space);

                     for (int j = 0; j < BLOCK_SIZE; ++j) {
                         resCell += aShared[jLoc][j] * bShared[j][iLoc];
                     }

                     nd_item.barrier(sycl::access::fence_space::local_space);
                 }

                 d_c[jGlob * size + iGlob] = resCell;
             });
     }).wait();

    q.memcpy(c.data(), d_c, sizeof(float) * C_size).wait();

    sycl::free(d_a, q);
    sycl::free(d_b, q);
    sycl::free(d_c, q);

    return c;
}
