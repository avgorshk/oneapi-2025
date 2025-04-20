#include "gemm_block_oneapi.h"

std::vector<float> GemmBlockONEAPI(
    const std::vector<float> a, const std::vector<float> b,
    size_t size, sycl::device device) {
    const int BLOCK_SIZE = 16;
    std::vector<float> c(size * size);
    sycl::queue q(device);

    {
        sycl::buffer<float, 1> a_buf(a.data(), sycl::range<1>(a.size()));
        sycl::buffer<float, 1> b_buf(b.data(), sycl::range<1>(b.size()));
        sycl::buffer<float, 1> c_buf(c.data(), sycl::range<1>(c.size()));

        q.submit([&](sycl::handler &cgh) {
             auto a_acc = a_buf.get_access<sycl::access::mode::read>(cgh);
             auto b_acc = b_buf.get_access<sycl::access::mode::read>(cgh);
             auto c_acc = c_buf.get_access<sycl::access::mode::write>(cgh);
             sycl::local_accessor<float, 1> aShared(BLOCK_SIZE * BLOCK_SIZE, cgh);
             sycl::local_accessor<float, 1> bShared(BLOCK_SIZE * BLOCK_SIZE, cgh);

             size_t num_blocks = (size + BLOCK_SIZE - 1) / BLOCK_SIZE;
             sycl::range<2> global_range(num_blocks * BLOCK_SIZE, num_blocks * BLOCK_SIZE);
             sycl::range<2> local_range(BLOCK_SIZE, BLOCK_SIZE);
             cgh.parallel_for(
                 sycl::nd_range<2>(global_range, local_range),
                 [=](sycl::nd_item<2> nd_item) {
                     size_t iGlob = nd_item.get_group(1) * BLOCK_SIZE + nd_item.get_local_id(1);
                     size_t jGlob = nd_item.get_group(0) * BLOCK_SIZE + nd_item.get_local_id(0);
                     size_t iLoc = nd_item.get_local_id(1);
                     size_t jLoc = nd_item.get_local_id(0);

                     int numBlocks = nd_item.get_group_range(0);
                     float resCell = 0.0f;

                     for (int i = 0; i < numBlocks; ++i) {
                         aShared[iLoc * BLOCK_SIZE + jLoc] = a_acc[iGlob * size + i * BLOCK_SIZE + jLoc];
                         bShared[iLoc * BLOCK_SIZE + jLoc] = b_acc[(i * BLOCK_SIZE + iLoc) * size + jGlob];

                         nd_item.barrier(sycl::access::fence_space::local_space);

                         for (int j = 0; j < BLOCK_SIZE; ++j) {
                             resCell += aShared[iLoc * BLOCK_SIZE + j] * bShared[j * BLOCK_SIZE + jLoc];
                         }

                         nd_item.barrier(sycl::access::fence_space::local_space);
                     }

                     c_acc[iGlob * size + jGlob] = resCell;
                 });
         }).wait();
    }

    return c;
}
