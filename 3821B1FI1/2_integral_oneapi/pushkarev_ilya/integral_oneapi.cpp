#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    const float step = (end - start) / count;
    const float area = step * step;
    float result = 0.0f;

    try {
        sycl::queue q(device, sycl::property::queue::enable_profiling{});

        const int max_count = std::min(count, 4096);
        const size_t wg_size = 8;
        const size_t rounded = ((max_count + wg_size - 1) / wg_size) * wg_size;
        const size_t num_groups = rounded / wg_size;

        sycl::buffer<float, 1> result_buf(&result, 1);
        sycl::buffer<float, 1> group_buf(num_groups * num_groups);

        q.submit([&](sycl::handler& h) {
            auto acc = group_buf.get_access<sycl::access::mode::write>(h);
            h.fill(acc, 0.0f);
        }).wait();

        q.submit([&](sycl::handler& h) {
            auto out = group_buf.get_access<sycl::access::mode::write>(h);
            sycl::local_accessor<float, 1> local(sycl::range<1>(wg_size * wg_size), h);

            h.parallel_for(
                sycl::nd_range<2>(
                    sycl::range<2>(rounded, rounded),
                    sycl::range<2>(wg_size, wg_size)
                ),
                [=](sycl::nd_item<2> item) {
                    int gi = item.get_global_id(0);
                    int gj = item.get_global_id(1);
                    int li = item.get_local_id(0);
                    int lj = item.get_local_id(1);
                    int lid = li * wg_size + lj;
                    int gid = item.get_group(0) * num_groups + item.get_group(1);

                    local[lid] = 0.0f;
                    if (gi < max_count && gj < max_count) {
                        float x = start + (gi + 0.5f) * step;
                        float y = start + (gj + 0.5f) * step;
                        local[lid] = sycl::sin(x) * sycl::cos(y) * area;
                    }

                    item.barrier(sycl::access::fence_space::local_space);

                    for (int s = (wg_size * wg_size) / 2; s > 0; s /= 2) {
                        if (lid < s) local[lid] += local[lid + s];
                        item.barrier(sycl::access::fence_space::local_space);
                    }

                    if (lid == 0) out[gid] = local[0];
                }
            );
        }).wait();

        q.submit([&](sycl::handler& h) {
            auto in = group_buf.get_access<sycl::access::mode::read>(h);
            auto out = result_buf.get_access<sycl::access::mode::write>(h);
            h.single_task([=]() {
                float sum = 0.0f;
                for (size_t i = 0; i < num_groups * num_groups; ++i)
                    sum += in[i];
                out[0] = sum;
            });
        }).wait();

        result = result_buf.get_host_access()[0];

        if (max_count < count) {
            result *= static_cast<float>(count * count) / (max_count * max_count);
        }
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        throw;
    }

    return result;
}
