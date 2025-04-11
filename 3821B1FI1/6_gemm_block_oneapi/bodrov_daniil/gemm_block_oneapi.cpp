#include "gemm_block_oneapi.h"
#include <vector>

std::vector<float> GemmBlockONEAPI(
    const std::vector<float> a, const std::vector<float> b,
    size_t size, sycl::device device) {
    constexpr size_t TILE_SIZE = 64;
    constexpr size_t SUBTILE_SIZE = 16;
    
    std::vector<float> result(size * size, 0.0f);
    sycl::queue q(device, sycl::property::queue::in_order());
    
    // Создаем буферы для матриц
    sycl::buffer<float, 2> buf_a(a.data(), sycl::range<2>(size, size));
    sycl::buffer<float, 2> buf_b(b.data(), sycl::range<2>(size, size));
    sycl::buffer<float, 2> buf_c(result.data(), sycl::range<2>(size, size));
    
    // Вычисляем количество блоков
    const size_t num_tiles = (size + TILE_SIZE - 1) / TILE_SIZE;
    
    q.submit([&](sycl::handler& h) {
        auto a_acc = buf_a.get_access<sycl::access::mode::read>(h);
        auto b_acc = buf_b.get_access<sycl::access::mode::read>(h);
        auto c_acc = buf_c.get_access<sycl::access::mode::write>(h);
        
        // Выделяем локальную память для подблоков
        sycl::local_accessor<float, 2> local_a(sycl::range<2>(TILE_SIZE, SUBTILE_SIZE), h);
        sycl::local_accessor<float, 2> local_b(sycl::range<2>(SUBTILE_SIZE, TILE_SIZE), h);
        
        h.parallel_for(sycl::nd_range<2>(
            sycl::range<2>(num_tiles * TILE_SIZE, num_tiles * TILE_SIZE),
            sycl::range<2>(TILE_SIZE, TILE_SIZE)
        ), [=](sycl::nd_item<2> item) {
            const size_t local_row = item.get_local_id(0);
            const size_t local_col = item.get_local_id(1);
            const size_t global_row = item.get_global_id(0);
            const size_t global_col = item.get_global_id(1);
            
            if (global_row >= size || global_col >= size) return;
            
            float sum = 0.0f;
            
            // Обрабатываем матрицу блоками
            for (size_t tile = 0; tile < num_tiles; ++tile) {
                // Загружаем подблоки в локальную память
                for (size_t subtile = 0; subtile < TILE_SIZE; subtile += SUBTILE_SIZE) {
                    // Загружаем подблок A
                    if (local_col < SUBTILE_SIZE && 
                        global_row < size && 
                        tile * TILE_SIZE + subtile + local_col < size) {
                        local_a[local_row][local_col] = 
                            a_acc[global_row][tile * TILE_SIZE + subtile + local_col];
                    }
                    
                    // Загружаем подблок B
                    if (local_row < SUBTILE_SIZE && 
                        tile * TILE_SIZE + subtile + local_row < size && 
                        global_col < size) {
                        local_b[local_row][local_col] = 
                            b_acc[tile * TILE_SIZE + subtile + local_row][global_col];
                    }
                    
                    item.barrier(sycl::access::fence_space::local_space);
                    
                    // Умножаем подблоки с разверткой цикла
                    #pragma unroll
                    for (size_t k = 0; k < SUBTILE_SIZE; ++k) {
                        sum += local_a[local_row][k] * local_b[k][local_col];
                    }
                    
                    item.barrier(sycl::access::fence_space::local_space);
                }
            }
            
            // Записываем результат
            c_acc[global_row][global_col] = sum;
        });
    });
    
    q.wait();
    return result;
} 