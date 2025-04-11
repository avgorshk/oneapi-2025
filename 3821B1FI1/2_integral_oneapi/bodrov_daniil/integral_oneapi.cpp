#include "integral_oneapi.h"
#include <cmath>

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    // Шаг интегрирования
    float step = (end - start) / count;
    float area = step * step;
    
    // Создаем очередь на указанном устройстве
    sycl::queue queue(device);
    
    // Общее количество прямоугольников
    int total_points = count * count;
    
    // Буфер для результатов
    float* partial_results = new float[total_points];
    for (int i = 0; i < total_points; i++) {
        partial_results[i] = 0.0f;
    }
    
    {
        // Создаем буфер для результатов вычислений
        sycl::buffer<float, 1> result_buf(partial_results, total_points);
        
        // Вычисляем значения функции во всех точках
        queue.submit([&](sycl::handler& h) {
            auto result_accessor = result_buf.get_access<sycl::access::mode::write>(h);
            
            h.parallel_for(sycl::range<1>(total_points), [=](sycl::id<1> idx) {
                // Получаем индексы (i,j) из линейного индекса
                int i = idx[0] % count;
                int j = idx[0] / count;
                
                // Вычисляем координаты средней точки прямоугольника
                float x_mid = start + (i + 0.5f) * step;
                float y_mid = start + (j + 0.5f) * step;
                
                // Вычисляем значение функции и умножаем на площадь прямоугольника
                result_accessor[idx] = sin(x_mid) * cos(y_mid) * area;
            });
        }).wait();
    }
    
    // Суммируем результаты на хосте
    float result = 0.0f;
    for (int i = 0; i < total_points; i++) {
        result += partial_results[i];
    }
    
    // Освобождаем память
    delete[] partial_results;
    
    return result;
}