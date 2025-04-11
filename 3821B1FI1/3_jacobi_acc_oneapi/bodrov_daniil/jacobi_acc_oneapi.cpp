#include "jacobi_acc_oneapi.h"
#include <cmath>

std::vector<float> JacobiAccONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device) {
    // Определяем размер системы
    int n = b.size();
    
    // Создаем начальное приближение x и временное хранилище x_new
    std::vector<float> x(n, 0.0f);
    std::vector<float> x_new(n);
    
    // Создаем очередь SYCL на указанном устройстве
    sycl::queue queue(device);
    
    // Создаем буферы для данных
    sycl::buffer<float, 1> buf_a(a.data(), a.size());
    sycl::buffer<float, 1> buf_b(b.data(), b.size());
    sycl::buffer<float, 1> buf_x(x.data(), x.size());
    sycl::buffer<float, 1> buf_x_new(x_new.data(), x_new.size());
    sycl::buffer<float, 1> buf_diff(1);
    
    // Выполняем итерации метода Якоби
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // Вычисляем новое приближение
        queue.submit([&](sycl::handler& h) {
            auto acc_a = buf_a.get_access<sycl::access::mode::read>(h);
            auto acc_b = buf_b.get_access<sycl::access::mode::read>(h);
            auto acc_x = buf_x.get_access<sycl::access::mode::read>(h);
            auto acc_x_new = buf_x_new.get_access<sycl::access::mode::write>(h);
            
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                int row = i[0];
                float diagonal = acc_a[row * n + row];
                float sum = 0.0f;
                
                // Вычисляем сумму произведений элементов вне диагонали
                for (int j = 0; j < n; j++) {
                    if (j != row) {
                        sum += acc_a[row * n + j] * acc_x[j];
                    }
                }
                
                // Вычисляем новое значение по формуле Якоби
                acc_x_new[row] = (acc_b[row] - sum) / diagonal;
            });
        }).wait();
        
        // Вычисляем максимальную разницу между новым и старым приближением
        float max_diff = 0.0f;
        {
            queue.submit([&](sycl::handler& h) {
                auto acc_x = buf_x.get_access<sycl::access::mode::read>(h);
                auto acc_x_new = buf_x_new.get_access<sycl::access::mode::read>(h);
                auto acc_diff = buf_diff.get_access<sycl::access::mode::write>(h);
                
                // Создаем локальную память для редукции
                sycl::accessor<float, 1, sycl::access::mode::read_write, sycl::access::target::local> 
                    local_diff(sycl::range<1>(256), h);
                
                h.parallel_for(
                    sycl::nd_range<1>(sycl::range<1>(256), sycl::range<1>(256)),
                    [=](sycl::nd_item<1> item) {
                        int local_id = item.get_local_id(0);
                        local_diff[local_id] = 0.0f;
                        
                        // Каждый поток обрабатывает несколько элементов
                        for (int i = local_id; i < n; i += 256) {
                            float diff = std::abs(acc_x_new[i] - acc_x[i]);
                            local_diff[local_id] = std::max(local_diff[local_id], diff);
                        }
                        
                        item.barrier(sycl::access::fence_space::local_space);
                        
                        // Редукция для нахождения максимальной разницы
                        for (int stride = 128; stride > 0; stride >>= 1) {
                            if (local_id < stride) {
                                local_diff[local_id] = std::max(local_diff[local_id], local_diff[local_id + stride]);
                            }
                            item.barrier(sycl::access::fence_space::local_space);
                        }
                        
                        // Сохраняем результат
                        if (local_id == 0) {
                            acc_diff[0] = local_diff[0];
                        }
                    }
                );
            }).wait();
            
            auto host_diff = buf_diff.get_access<sycl::access::mode::read>();
            max_diff = host_diff[0];
        }
        
        // Копируем новое приближение в старое
        queue.submit([&](sycl::handler& h) {
            auto acc_x = buf_x.get_access<sycl::access::mode::write>(h);
            auto acc_x_new = buf_x_new.get_access<sycl::access::mode::read>(h);
            
            h.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
                acc_x[i] = acc_x_new[i];
            });
        }).wait();
        
        // Проверяем условие остановки по точности
        if (max_diff < accuracy) {
            break;
        }
    }
    
    // Копируем результат обратно в вектор x
    {
        auto host_x = buf_x.get_access<sycl::access::mode::read>();
        for (int i = 0; i < n; i++) {
            x[i] = host_x[i];
        }
    }
    
    return x;
}