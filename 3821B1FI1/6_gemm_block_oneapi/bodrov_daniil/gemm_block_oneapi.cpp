#include "gemm_block_oneapi.h"
#include <cmath>

std::vector<float> GemmBlockONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        size_t size, sycl::device device) {
    // Определяем оптимальный размер блока
    // Для простоты берем 16, но в реальных приложениях это значение 
    // следует подбирать экспериментально для конкретной архитектуры
    const size_t BLOCK_SIZE = 16;
    
    // Создаем очередь SYCL на указанном устройстве
    sycl::queue queue(device);
    
    // Создаем вектор для результата
    std::vector<float> c(size * size, 0.0f);
    
    // Создаем буферы для работы с данными
    sycl::buffer<float, 1> buf_a(a.data(), a.size());
    sycl::buffer<float, 1> buf_b(b.data(), b.size());
    sycl::buffer<float, 1> buf_c(c.data(), c.size());
    
    // Количество блоков в одном измерении матрицы
    size_t num_blocks = size / BLOCK_SIZE;
    
    // Выполняем блочное умножение матриц
    for (size_t block_i = 0; block_i < num_blocks; block_i++) {
        for (size_t block_j = 0; block_j < num_blocks; block_j++) {
            // Для каждого блока результирующей матрицы C(I,J)
            // выполняем суммирование произведений блоков A(I,K) и B(K,J)
            for (size_t block_k = 0; block_k < num_blocks; block_k++) {
                queue.submit([&](sycl::handler& h) {
                    auto a_acc = buf_a.get_access<sycl::access::mode::read>(h);
                    auto b_acc = buf_b.get_access<sycl::access::mode::read>(h);
                    auto c_acc = buf_c.get_access<sycl::access::mode::read_write>(h);
                    
                    // Используем локальную память для хранения блоков
                    sycl::accessor<float, 2, sycl::access::mode::read_write, sycl::access::target::local>
                        a_local(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), h);
                    sycl::accessor<float, 2, sycl::access::mode::read_write, sycl::access::target::local>
                        b_local(sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE), h);
                    
                    h.parallel_for(
                        sycl::nd_range<2>(
                            sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE),
                            sycl::range<2>(BLOCK_SIZE, BLOCK_SIZE)
                        ),
                        [=](sycl::nd_item<2> item) {
                            // Локальные индексы внутри блока
                            size_t local_i = item.get_local_id(0);
                            size_t local_j = item.get_local_id(1);
                            
                            // Глобальные индексы для блока результата C(I,J)
                            size_t global_i = block_i * BLOCK_SIZE + local_i;
                            size_t global_j = block_j * BLOCK_SIZE + local_j;
                            
                            // Загружаем блоки A(I,K) и B(K,J) в локальную память
                            a_local[local_i][local_j] = a_acc[global_i * size + (block_k * BLOCK_SIZE + local_j)];
                            b_local[local_i][local_j] = b_acc[(block_k * BLOCK_SIZE + local_i) * size + global_j];
                            
                            // Ждем, чтобы убедиться, что все данные загружены
                            item.barrier(sycl::access::fence_space::local_space);
                            
                            // Выполняем умножение блока A(I,K) на блок B(K,J)
                            float sum = 0.0f;
                            for (size_t k = 0; k < BLOCK_SIZE; k++) {
                                sum += a_local[local_i][k] * b_local[k][local_j];
                            }
                            
                            // Прибавляем результат к блоку C(I,J)
                            c_acc[global_i * size + global_j] += sum;
                        }
                    );
                });
            }
        }
    }
    
    // Ждем завершения всех операций
    queue.wait();
    
    return c;
} 