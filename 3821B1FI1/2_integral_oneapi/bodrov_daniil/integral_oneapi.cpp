#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    const float step = (end - start) / count;
    const float area = step * step;
    float result = 0.0f;

    try {
        sycl::queue q(device);
        
        // Округляем размер до ближайшего числа, кратного 16
        const int rounded_count = ((count + 15) / 16) * 16;
        
        // Создаем буфер для результата
        sycl::buffer<float, 1> result_buf(&result, 1);
        
        // Вычисляем интеграл
        q.submit([&](sycl::handler& h) {
            auto result_acc = result_buf.get_access<sycl::access::mode::write>(h);
            
            h.parallel_for(sycl::nd_range<2>(
                sycl::range<2>(rounded_count, rounded_count),
                sycl::range<2>(16, 16)
            ), [=](sycl::nd_item<2> item) {
                const int i = item.get_global_id(0);
                const int j = item.get_global_id(1);
                
                // Проверяем, не вышли ли мы за пределы исходного диапазона
                if (i < count && j < count) {
                    const float x = start + (i + 0.5f) * step;
                    const float y = start + (j + 0.5f) * step;
                    
                    sycl::atomic_ref<float, sycl::memory_order::relaxed,
                                   sycl::memory_scope::device,
                                   sycl::access::address_space::global_space>
                        result_ref(result_acc[0]);
                    result_ref.fetch_add(sycl::sin(x) * sycl::cos(y) * area);
                }
            });
        }).wait();
        
        auto result_acc = result_buf.get_access<sycl::access::mode::read>();
        result = result_acc[0];
        
    } catch (const sycl::exception& e) {
        std::cerr << "SYCL Exception: " << e.what() << std::endl;
        throw;
    }
    
    return result;
}