#include "jacobi_dev_oneapi.h"
#include <cmath>

std::vector<float> JacobiDevONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device) {
    // Определяем размер системы
    int n = b.size();
    
    // Создаем очередь SYCL на указанном устройстве
    sycl::queue queue(device, sycl::property::queue::in_order());
    
    // Создаем память для данных на устройстве с использованием USM
    float* dev_a = sycl::malloc_device<float>(n * n, queue);
    float* dev_b = sycl::malloc_device<float>(n, queue);
    float* dev_x = sycl::malloc_device<float>(n, queue);
    float* dev_x_new = sycl::malloc_device<float>(n, queue);
    float* dev_diff = sycl::malloc_device<float>(1, queue);
    
    // Создаем временную память для редукции
    const int block_size = 256;
    float* dev_block_diffs = sycl::malloc_device<float>(block_size, queue);
    
    // Копируем исходные данные на устройство
    queue.memcpy(dev_a, a.data(), n * n * sizeof(float));
    queue.memcpy(dev_b, b.data(), n * sizeof(float));
    
    // Инициализируем начальное приближение нулями
    queue.fill(dev_x, 0.0f, n);
    queue.wait();
    
    // Создаем вектор для результата
    std::vector<float> x(n, 0.0f);
    
    // Выполняем итерации метода Якоби
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // Вычисляем новое приближение
        queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            int row = i[0];
            float diagonal = dev_a[row * n + row];
            float sum = 0.0f;
            
            // Вычисляем сумму произведений элементов вне диагонали
            for (int j = 0; j < n; j++) {
                if (j != row) {
                    sum += dev_a[row * n + j] * dev_x[j];
                }
            }
            
            // Вычисляем новое значение по формуле Якоби
            dev_x_new[row] = (dev_b[row] - sum) / diagonal;
        }).wait();
        
        // Инициализируем переменные для максимальной разницы
        queue.fill(dev_diff, 0.0f, 1).wait();
        queue.fill(dev_block_diffs, 0.0f, block_size).wait();
        
        // Вычисляем максимальную разницу между новым и старым приближением в блоках
        queue.parallel_for(sycl::range<1>(block_size), [=](sycl::id<1> idx) {
            int lid = idx[0];
            float local_max = 0.0f;
            
            // Каждый поток обрабатывает свой набор элементов
            for (int i = lid; i < n; i += block_size) {
                float diff = std::abs(dev_x_new[i] - dev_x[i]);
                local_max = std::max(local_max, diff);
            }
            
            // Сохраняем локальный максимум каждого потока
            dev_block_diffs[lid] = local_max;
        }).wait();
        
        // Находим глобальный максимум в отдельном потоке
        queue.single_task([=]() {
            float max_diff = 0.0f;
            for (int i = 0; i < block_size; i++) {
                max_diff = std::max(max_diff, dev_block_diffs[i]);
            }
            dev_diff[0] = max_diff;
        }).wait();
        
        // Копируем максимальную разницу с устройства на хост
        float host_diff;
        queue.memcpy(&host_diff, dev_diff, sizeof(float)).wait();
        
        // Копируем новое приближение в старое
        queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            dev_x[i] = dev_x_new[i];
        }).wait();
        
        // Проверяем условие остановки по точности
        if (host_diff < accuracy) {
            break;
        }
    }
    
    // Копируем результат с устройства на хост
    queue.memcpy(x.data(), dev_x, n * sizeof(float)).wait();
    
    // Освобождаем память на устройстве
    sycl::free(dev_a, queue);
    sycl::free(dev_b, queue);
    sycl::free(dev_x, queue);
    sycl::free(dev_x_new, queue);
    sycl::free(dev_diff, queue);
    sycl::free(dev_block_diffs, queue);
    
    return x;
}