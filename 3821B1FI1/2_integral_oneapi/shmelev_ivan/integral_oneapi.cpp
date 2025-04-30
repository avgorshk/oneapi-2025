#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    const float step = (end - start) / count;
    const float area = step * step;
    float result = 0.0f;

    try {
        sycl::queue q(device, sycl::property::queue::enable_profiling {});

        const int max_count = std::min(count, 4096);

        const int work_group_size = 8;

        const int rounded_count = ((max_count + work_group_size - 1) / work_group_size) * work_group_size;

        const int num_groups = rounded_count / work_group_size;

        sycl::buffer<float, 1> result_buf(&result, 1);

        sycl::buffer<float, 1> group_results(sycl::range<1>(num_groups * num_groups));

        q.submit([&](sycl::handler& h) {
            auto acc = group_results.get_access<sycl::access::mode::write>(h);
            h.fill(acc, 0.0f);
            }).wait();

        q.submit([&](sycl::handler& h) {
            auto group_results_acc = group_results.get_access<sycl::access::mode::write>(h);

            sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local>
                local_sum(sycl::range<1>(work_group_size * work_group_size), h);

            h.parallel_for(sycl::nd_range<2>(
                sycl::range<2>(rounded_count, rounded_count),
                sycl::range<2>(work_group_size, work_group_size)
            ), [=](sycl::nd_item<2> item) {
                const int global_i = item.get_global_id(0);
                const int global_j = item.get_global_id(1);
                const int local_i = item.get_local_id(0);
                const int local_j = item.get_local_id(1);
                const int local_idx = local_i * work_group_size + local_j;
                const int group_i = item.get_group(0);
                const int group_j = item.get_group(1);
                const int group_idx = group_i * num_groups + group_j;

                local_sum[local_idx] = 0.0f;

                if (global_i < max_count && global_j < max_count) {
                    const float x = start + (global_i + 0.5f) * step;
                    const float y = start + (global_j + 0.5f) * step;
                    local_sum[local_idx] = sycl::sin(x) * sycl::cos(y) * area;
                }

                item.barrier(sycl::access::fence_space::local_space);

                for (int stride = work_group_size * work_group_size / 2; stride > 0; stride /= 2) {
                    if (local_idx < stride) {
                        local_sum[local_idx] += local_sum[local_idx + stride];
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                }

                if (local_idx == 0) {
                    group_results_acc[group_idx] = local_sum[0];
                }
                });
            }).wait();

        q.submit([&](sycl::handler& h) {
            auto group_results_acc = group_results.get_access<sycl::access::mode::read>(h);
            auto result_acc = result_buf.get_access<sycl::access::mode::write>(h);

            h.single_task([=]() {
                float sum = 0.0f;
                for (int i = 0; i < num_groups * num_groups; i++) {
                    sum += group_results_acc[i];
                }
                result_acc[0] = sum;
                });
            }).wait();

        auto result_acc = result_buf.get_access<sycl::access::mode::read>();
        result = result_acc[0];

        if (max_count < count) {
            float scale_factor = static_cast<float>(count * count) / (max_count * max_count);
            result *= scale_factor;
        }

    }
    catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        throw;
    }

    return result;
}
