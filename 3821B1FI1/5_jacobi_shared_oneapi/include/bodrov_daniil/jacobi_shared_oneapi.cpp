#include "jacobi_shared_oneapi.h"
#include <cmath>

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device) {
    // Определяем размер системы
    int n = b.size();
    
    // Создаем очередь SYCL на указанном устройстве
    sycl::queue queue(device, sycl::property::queue::in_order());
    
    // Создаем shared память, доступную и хосту, и устройству
    float* shared_a = sycl::malloc_shared<float>(n * n, queue);
    float* shared_b = sycl::malloc_shared<float>(n, queue);
    float* shared_x = sycl::malloc_shared<float>(n, queue);
    float* shared_x_new = sycl::malloc_shared<float>(n, queue);
    float* shared_diff = sycl::malloc_shared<float>(1, queue);
    
    // Инициализируем данные на хосте
    for (int i = 0; i < n * n; i++) {
        shared_a[i] = a[i];
    }
    
    for (int i = 0; i < n; i++) {
        shared_b[i] = b[i];
        shared_x[i] = 0.0f; // Начальное приближение - нули
    }
    
    *shared_diff = 0.0f;
    
    // Выполняем итерации метода Якоби
    for (int iter = 0; iter < ITERATIONS; iter++) {
        // Вычисляем новое приближение
        queue.parallel_for(sycl::range<1>(n), [=](sycl::id<1> i) {
            int row = i[0];
            float diagonal = shared_a[row * n + row];
            float sum = 0.0f;
            
            // Вычисляем сумму произведений элементов вне диагонали
            for (int j = 0; j < n; j++) {
                if (j != row) {
                    sum += shared_a[row * n + j] * shared_x[j];
                }
            }
            
            // Вычисляем новое значение по формуле Якоби
            shared_x_new[row] = (shared_b[row] - sum) / diagonal;
        }).wait();
        
        // Вычисляем максимальную разницу прямо на хосте
        *shared_diff = 0.0f;
        for (int i = 0; i < n; i++) {
            float diff = std::abs(shared_x_new[i] - shared_x[i]);
            *shared_diff = std::max(*shared_diff, diff);
        }
        
        // Копируем новое приближение в старое
        for (int i = 0; i < n; i++) {
            shared_x[i] = shared_x_new[i];
        }
        
        // Проверяем условие остановки по точности
        if (*shared_diff < accuracy) {
            break;
        }
    }
    
    // Создаем вектор для результата
    std::vector<float> x(n);
    
    // Копируем результат в вектор x
    for (int i = 0; i < n; i++) {
        x[i] = shared_x[i];
    }
    
    // Освобождаем shared память
    sycl::free(shared_a, queue);
    sycl::free(shared_b, queue);
    sycl::free(shared_x, queue);
    sycl::free(shared_x_new, queue);
    sycl::free(shared_diff, queue);
    
    return x;
}