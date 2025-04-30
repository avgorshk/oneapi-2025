# Content
- [How To](#how-to)
- [Configuration](#configuration)
- [Time Measurement](#time-measurement)
- [Tasks](#tasks)
- [Results](#results)

# How To
1. Create [github](https://github.com/) account (if not exists);
2. Make sure SSH clone & commit is working ([Connecting to GitHub with SSH](https://docs.github.com/en/authentication/connecting-to-github-with-ssh));
3. Fork this repo (just click **Fork** button on the top of the page, detailed instructions [here](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project))
4. Clone your forked repo into your local machine, use your user instead of `username`:
```sh
git clone git@github.com:username/oneapi-2025.git
cd oneapi-2025
```
5. Go to your group folder, e.g.:
```sh
cd 3821B1FI1
```
6. Go to needed task folder, e.g.:
```sh
cd 1_integral_oneapi
```
7. Create new folder with your surname and name (**make sure it's the same for all tasks**), e.g.:
```sh
mkdir petrov_ivan
```
8. Copy your task source/header files (including main program) into this folder (use `copy` instead of `cp` on Windows), e.g.:
```sh
cd petrov_ivan
cp /home/usr/lab/*.cpp .
cp /home/usr/lab/*.h .
```
8. Push your sources to github repo, e.g.:
```sh
cd ..
git add .
git commit -m "1_integral_oneapi task"
git push
```
9. Go to your repo in browser, click **Contribute** button on the top of page, then **Open pull request**. Provide meaningfull request title and description, then **Create pull request** (see details [here](https://docs.github.com/en/get-started/exploring-projects-on-github/contributing-to-a-project)).
10. Go to Pull Requests [page](https://github.com/avgorshk/oneapi-2025/pulls) in course repo, find your pull request and check if there are no any merge conflicts occur. If merge conflicts happen - resolve it following the instruction provided by github.

# Time Measurement
The following scheme is used to measure task execution time:
```cpp
int main() {
    // ...

    // Warming-up
    Task(input, size / 8);

    // Performance Measuring
    auto start = std::chrono::high_resolution_clock::now();
    auto c = Task(input, size);
    auto end = std::chrono::high_resolution_clock::now();

    // ...
}
```

# Configuration
- CPU: Intel Core i5 12600K (4 cores, 4 threads)
- RAM: 16 GB
- GPU: Intel UHD Graphics 770 (8 GB)
- Host Compiler: GCC 11.4.0
- oneAPI: 2025.0

# Tasks
## Task #1: Permutations
To train modern C++11 skills, the following task is suggested.

There is a set of strings, each string is unique and contains small English letters only. The goal is to make a dictionary of permutations - for each string in a set one should find all other strings from the same set that are permutations of this string.

The following function should be implemented:
```cpp
using dictionary_t = std::map<std::string, std::vector<std::string>>;
void Permutations(dictionary_t& dictionary);
```
Initially, dictionary will contain key strings only (all vectors will be empty). After function completion, the same dictionary should additionally keep the lists of permutations for each key string. Each list of permutations should be sorted in reverse alphabetical order.

The following example will illustrate the idea.
Let's consider the following set of strings as an input:
```
aaa
acb
acd
ad
adc
bac
bc
bcc
bd
bda
bdc
caa
cad
cb
cc
ccb
cd
dac
db
dc
dca
dcb
dcc
dd
```
As a result, one should get the following:
```
aaa :
acb : bac
acd : dca dac cad adc
ad :
adc : dca dac cad acd
bac : acb
bc : cb
bcc : ccb
bd : db
bda :
bdc : dcb
caa :
cad : dca dac adc acd
cb : bc
cc :
ccb : bcc
cd : dc
dac : dca cad adc acd
db : bd
dc : cd
dca : dac cad adc acd
dcb : bdc
dcc :
dd :
```
Two files are expected to be uploaded:
- permutations_cxx.h
```cpp
#ifndef __PERMUTATIONS_CXX_H
#define __PERMUTATIONS_CXX_H

#include <map>
#include <string>
#include <vector>

using dictionary_t = std::map<std::string, std::vector<std::string>>;

void Permutations(dictionary_t& dictionary);

#endif  // __PERMUTATIONS_CXX_H
```
- permutations_cxx.cpp
```cpp
#include "permutations_cxx.h"

void Permutations(dictionary_t& dictionary) {
    // Place your implementation here
}
```

## Task #2: Double Integral Computation
In many cases there is no analytical solution for the integral, but one may use approximation of any kind. One of such approximations could be retrieved with the help of [Riemann Sum](https://en.wikipedia.org/wiki/Riemann_sum).

E.g., for double integral the following formula could be used to get its approximate value:

$\int_a^b\int_c^df(x,y)dxdy=\sum_{j=0}^{n-1}\sum_{i=0}^{n-1}f(\frac{x_i+x_{i+1}}2, \frac{y_j+y_{j+1}}2)(x_{i+1}-x_i)(y_{j+1}-y_j)$

The task goal is to compute the following integral using **Middle Riemann Sum**:

$\int_{start}^{end}\int_{start}^{end}sin(x)cos(y)dxdy$

**Hint**: for $start=0$ and $end=1$ it should be equal to $0.3868223$.

Implement the function in SYCL with the following interface:
```cpp
float IntegralONEAPI(float start, float end, int count, sycl::device device);
```
$Count$ means how many intervals one should use to split integration space (the same for $x$ and $y$). E.g. if $count=10$, one will have $10*10=100$ rectangles in total.

Two files are expected to be uploaded:
- integral_oneapi.h
```cpp
#ifndef __INTEGRAL_ONEAPI_H
#define __INTEGRAL_ONEAPI_H

#include <sycl/sycl.hpp>

float IntegralONEAPI(float start, float end, int count, sycl::device device);

#endif  // __INTEGRAL_ONEAPI_H
```
- integral_oneapi.cpp
```cpp
#include "integral_oneapi.h"

float IntegralONEAPI(float start, float end, int count, sycl::device device) {
    // Place your implementation here
}
```

## Task #3: Jacobi Method (Accessors)
Systems of linear equations are basic apparatus (as part of mathematical model) for variety of problems in physics, chemistry, economics, etc.

There two methods of their solving - direct and iterative. Direct methods (like Gaussian Elimination) are able to get accurate solution, while iterative ones can only provide approximate results.
At the same time, in practice iterative methods may be preferable, e.g. in case of huge matrices that have to stored in RAM while using direct approaches, or if one has close enough initial estimation for final result.

One of such iterative methods is [Jacobi Method](https://en.wikipedia.org/wiki/Jacobi_method) that allows to get accurate enough solution using the following formula:

$x_i^{(k+1)}=\frac{1}a_{ii}(b_i-\sum_{j \neq i}a_{ij}x_j^{(k)})$

Here $x^{(k+1)}$ is the next approximation of system solution, computed from the previous  $x^{(k)}$. First approximation $x^{(0)}$ could be all zeros.

There are two ways to stop computations:
1. After $N$ iterations, where $N$ is some predefined constant;
2. If $|x^{(k+1)}-x^{(k)}|<Eps$, where $Eps$ is target accuracy.

**Note:** to ensure method convergence one should use it only for [strictly diagonally dominant](https://en.wikipedia.org/wiki/Diagonally_dominant_matrix) system, that means:

$|a_{ii}|>\sum_{j \neq i}|a_{ij}|$ for any $i$.

To complete this task, one should implement the function that computes the solution for the system of linear equations using Jacobi method:
```cpp
std::vector<float> JacobiAccONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device);
```
One should implement both stop methods at the same time. Use $N=1024$ as maximum iterations count and $accuracy$ argument as $Eps$. computations have to be stopped when $|x^{(k+1)}-x^{(k)}|<accuracy$ first, and if it's not happening, when after 1024 iterations.

Matix $a$ is stored by rows. One should implement the algorithm using SYCL buffers & accessors approach.

Two files are expected to be uploaded:
- jacobi_acc_oneapi.h
```cpp
#ifndef __JACOBI_ACC_ONEAPI_H
#define __JACOBI_ACC_ONEAPI_H

#include <vector>

#include <sycl/sycl.hpp>

#define ITERATIONS 1024

std::vector<float> JacobiAccONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device);

#endif  // __JACOBI_ACC_ONEAPI_H
```
- jacobi_acc_oneapi.cpp
```cpp
#include "jacobi_acc_oneapi.h"

std::vector<float> JacobiAccONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device) {
    // Place your implementation here
}
```

## Task #4: Jacobi Method (Device Memory)
This task assumes Jacobi method implementation using SYCL device memory approach (see all the details in Task #2).

Two files are expected to be uploaded:
- jacobi_dev_oneapi.h
```cpp
#ifndef __JACOBI_DEV_ONEAPI_H
#define __JACOBI_DEV_ONEAPI_H

#include <vector>

#include <sycl/sycl.hpp>

#define ITERATIONS 1024

std::vector<float> JacobiDevONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device);

#endif  // __JACOBI_DEV_ONEAPI_H
```
- jacobi_dev_oneapi.cpp
```cpp
#include "jacobi_dev_oneapi.h"

std::vector<float> JacobiDevONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device) {
    // Place your implementation here
}
```

## Task #5: Jacobi Method (Shared Memory)
This task assumes Jacobi method implementation using SYCL shared memory approach (see all the details in Task #2).

Two files are expected to be uploaded:
- jacobi_shared_oneapi.h
```cpp
#ifndef __JACOBI_SHARED_ONEAPI_H
#define __JACOBI_SHARED_ONEAPI_H

#include <vector>

#include <sycl/sycl.hpp>

#define ITERATIONS 1024

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device);

#endif  // __JACOBI_SHARED_ONEAPI_H
```
- jacobi_shared_oneapi.cpp
```cpp
#include "jacobi_shared_oneapi.h"

std::vector<float> JacobiSharedONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        float accuracy, sycl::device device) {
    // Place your implementation here
}
```

## Task #6: Block Matrix Multiplication
General matrix multiplication (GEMM) is a very basic and broadly used linear algebra operation applied in high performance computing (HPC), statistics, deep learning and other domains. There are a lot of GEMM algorithms with different mathematical complexity form $O(n^3)$ for naive and block approaches to $O(n^{2.371552})$ for the method descibed by Williams et al. in 2024 [[1](https://epubs.siam.org/doi/10.1137/1.9781611977912.134)]. But despite a variety of algorithms with low complexity, block matrix multiplication remains the most used implementation in practice since it fits to modern HW better.

In real applications block-based approach for matrix multiplication can get multiple times faster execution comparing with naive version due to cache friendly approach.

In block version, algorithm could be divided into three stages:
1. Split matricies into blocks (block size normally affects performance significantly so choose it consciously);
2. Multiply two blocks to get partial result;
3. Replay step 2 for all row/column blocks accumulating values into a single result block.

From math perspective, block matrix multiplication could be described by the following formula, where $C_{IJ}$, $A_{IK}$ and $B_{KJ}$ are sub-matricies with the size $block\_size*block\_size$:

$C_{IJ}=\sum_{k=1}^{block_count}A_{IK}B_{KJ}$

Each matrix must be stored in a linear array by rows, so that `a.size()==size*size`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2 and all matricies are square.

Two files are expected to be uploaded:
- gemm_block_oneapi.h:
```cpp
#ifndef __GEMM_BLOCK_ONEAPI_H
#define __GEMM_BLOCK_ONEAPI_H

#include <vector>

#include <sycl/sycl.hpp>

std::vector<float> GemmBlockONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        size_t size, sycl::device device);

#endif  // __GEMM_BLOCK_ONEAPI_H
```
- gemm_block_oneapi.cpp:
```cpp
#include "gemm_block_oneapi.h"

std::vector<float> GemmBlockONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        size_t size, sycl::device device) {
    // Place your implementation here
}
```

## Task #7: Matrix Multiplication Using oneMKL
The most performant way to multiply two matrices on particular hardware is to use vendor-provided library for this purpose. In SYCL it's [oneMKL](https://oneapi-spec.uxlfoundation.org/specifications/oneapi/v1.3-rev-1/elements/onemkl/source/). Try to use oneMKL BLAS API to implement general matrix multiplication in most performant way.

Each matrix must be stored in a linear array by rows, so that `a.size()==size*suze`. Function takes two matricies and their size as inputs, and returns result matrix also stored by rows.

For simplicity, let's consider matrix size is always power of 2 and all matricies are square.

**Note**, that in oneMKL BLAS API matrix is expected to be stored by columns, so additional transpose (or slightly different API) may be required.

Two files are expected to be uploaded:
- gemm_mkl_oneapi.h:
```cpp
#ifndef __GEMM_MKL_ONEAPI_H
#define __GEMM_MKL_ONEAPI_H

#include <vector>

#include <sycl/sycl.hpp>

std::vector<float> GemmMklONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        size_t size, sycl::device device);

#endif  // __GEMM_MKL_ONEAPI_H
```
- gemm_mkl_oneapi.cpp:
```cpp
#include "gemm_mkl_oneapi.h"

std::vector<float> GemmMklONEAPI(
        const std::vector<float> a, const std::vector<float> b,
        size_t size, sycl::device device) {
    // Place your implementation here
}
```

# Results
## 1_permutations_cxx (10240 elements)
|Group|Name|Result|
|-----|----|------|
|3821B1FI2|isaev_dmitrii|0.0033|
|3821B1FI1|mirzakhmedov_alexander|0.0038|
|3821B1FI2|petrov_maksim|0.0039|
|3821B1FI3|durandin_vladimir|0.0039|
|3821B1FI1|borovkov_sergey|0.0049|
|3821B1FI1|shipitsin_alex|0.0053|
|3821B1FI1|bodrov_daniil|0.0054|
|3821B1FI3|pivovarov_alexey|0.0058|
|3821B1FI3|prokofev_kirill|0.0059|
|3821B1FI2|nogin_denis|0.0060|
|3821B1FI2|kostanyan_arsen|0.0060|
|3821B1FI1|kashirin_alexander|0.0060|
|3821B1FI3|kulagin_aleksandr|0.0060|
|3821B1FI3|savotina_valeria|0.0063|
|3821B1FI2|zakharov_artem|0.0067|
|3821B1FI1|balyasov_ivan|0.0067|
|3821B1FI3|simonyan_suren|0.0067|
|3821B1FI3|kuznetsov_artyom|0.0067|
|3821B1FI1|lysanova_julia|0.0069|
|3821B1FI2|soloninko_andrey|0.0070|
|3821B1FI3|sharapov_georgiy|0.0070|
|3821B1FI1|chernyshev_nikita|0.0073|
|3821B1FI3|korablev_nikita|0.0073|
|3821B1FI3|kulaev_zhenya|0.0073|
|3821B1FI1|bonyuk_peter|0.0074|
|3821B1FI3|safarov_nurlan|0.0075|
|3821B1FI3|ryabkov_vladislav|0.0076|
|3821B1FI2|zawadowski_jan|0.0078|
|3821B1FI2|loginov_maxim|0.0078|
|3821B1FI2|kostin_artem|0.0079|
|3821B1FI3|ivanov_nikita|0.0079|
|3821B1FI3|vasilev_ivan|0.0079|
|3821B1FI2|kazantsev_evgeny|0.0080|
|3821B1FI3|kulikov_artem|0.0080|
|3821B1FI3|safronov_mikhail|0.0081|
|3821B1FI3|volodin_evgeniy|0.0091|
|3821B1FI3|sadikov_damir|0.0093|
|3821B1FI3|benduyzhko_tatiana|0.0093|
|3821B1FI1|pushkarev_ilya|0.0094|
|3821B1FI3|tyulkina_olga|0.0095|
|3821B1FI1|shmelev_ivan|0.0099|
|3821B1FI2|travin_maksim|0.0104|
|3821B1FI3|polozov_vladislav|0.0107|
|**REF**|**REF**|**0.8951**|
|3821B1FI2|kanakov_roman|TEST FAILED|

## 2_integral_oneapi (65536 elements)
|Group|Name|Result|
|-----|----|------|
|3821B1FI1|shipitsin_alex|0.0334|
|3821B1FI1|bonyuk_peter|0.0356|
|3821B1FI1|borovkov_sergey|0.0356|
|3821B1FI1|bodrov_daniil|0.0366|
|3821B1FI1|shmelev_ivan|0.0367|
|3821B1FI3|pivovarov_alexey|0.0367|
|3821B1FI1|kashirin_alexander|0.0370|
|3821B1FI1|pushkarev_ilya|0.0827|
|3821B1FI2|petrov_maksim|0.4503|
|3821B1FI3|volodin_evgeniy|0.4515|
|**REF**|**REF**|**0.4682**|
|3821B1FI2|nogin_denis|0.9784|
|3821B1FI3|durandin_vladimir|0.9917|
|3821B1FI3|kulikov_artem|0.9977|
|3821B1FI3|safronov_mikhail|0.9996|
|3821B1FI2|travin_maksim|1.0000|
|3821B1FI3|ivanov_nikita|1.0015|
|3821B1FI2|zawadowski_jan|1.0018|
|3821B1FI3|tyulkina_olga|1.0048|
|3821B1FI3|korablev_nikita|1.0086|
|3821B1FI3|kulagin_aleksandr|1.2853|
|3821B1FI3|simonyan_suren|1.2950|
|3821B1FI2|kostin_artem|1.3396|
|3821B1FI2|kazantsev_evgeny|1.3527|
|3821B1FI3|savotina_valeria|1.3929|
|3821B1FI3|ryabkov_vladislav|1.3933|
|3821B1FI1|lysanova_julia|1.3934|
|3821B1FI3|prokofev_kirill|1.4003|
|3821B1FI2|isaev_dmitrii|1.4004|
|3821B1FI3|vasilev_ivan|1.4037|
|3821B1FI2|loginov_maxim|1.4044|
|3821B1FI2|kanakov_roman|1.4046|
|3821B1FI1|mirzakhmedov_alexander|1.4051|
|3821B1FI2|zakharov_artem|1.4051|
|3821B1FI1|balyasov_ivan|1.4059|
|3821B1FI2|soloninko_andrey|1.4059|
|3821B1FI3|safarov_nurlan|1.4079|
|3821B1FI1|chernyshev_nikita|1.4119|
|3821B1FI3|kuznetsov_artyom|1.4179|
|3821B1FI2|kostanyan_arsen|1.4367|
|3821B1FI3|sharapov_georgiy|1.4472|
|3821B1FI3|polozov_vladislav|1.4476|
|3821B1FI3|sadikov_damir|1.4479|
|3821B1FI3|kulaev_zhenya|1.4486|
|3821B1FI3|benduyzhko_tatiana|1.6310|

## 3_jacobi_acc_oneapi (4096 elements)
|Group|Name|Result|
|-----|----|------|
|3821B1FI2|kanakov_roman|0.2187|
|3821B1FI1|mirzakhmedov_alexander|0.2204|
|3821B1FI2|petrov_maksim|0.2334|
|3821B1FI2|kostin_artem|0.2355|
|3821B1FI1|borovkov_sergey|0.2373|
|3821B1FI3|safronov_mikhail|0.2539|
|3821B1FI1|shmelev_ivan|0.2551|
|3821B1FI1|shipitsin_alex|0.2593|
|3821B1FI3|safarov_nurlan|0.2594|
|3821B1FI2|soloninko_andrey|0.2597|
|3821B1FI3|prokofev_kirill|0.2649|
|3821B1FI1|balyasov_ivan|0.2651|
|3821B1FI2|nogin_denis|0.2655|
|3821B1FI1|bonyuk_peter|0.2655|
|3821B1FI1|lysanova_julia|0.2671|
|3821B1FI3|kulagin_aleksandr|0.2671|
|3821B1FI3|savotina_valeria|0.2673|
|3821B1FI1|pushkarev_ilya|0.2676|
|3821B1FI2|loginov_maxim|0.2677|
|3821B1FI3|pivovarov_alexey|0.2686|
|3821B1FI2|kostanyan_arsen|0.2692|
|3821B1FI3|sharapov_georgiy|0.2747|
|3821B1FI2|isaev_dmitrii|0.2791|
|3821B1FI3|kulaev_zhenya|0.2825|
|3821B1FI1|chernyshev_nikita|0.2838|
|3821B1FI3|durandin_vladimir|0.2841|
|3821B1FI2|zakharov_artem|0.2855|
|3821B1FI3|kuznetsov_artyom|0.2886|
|3821B1FI3|ryabkov_vladislav|0.2902|
|3821B1FI3|ivanov_nikita|0.2926|
|3821B1FI3|vasilev_ivan|0.2932|
|3821B1FI1|bodrov_daniil|0.3032|
|3821B1FI3|korablev_nikita|0.3170|
|3821B1FI2|kazantsev_evgeny|0.3232|
|3821B1FI3|volodin_evgeniy|0.3332|
|3821B1FI2|travin_maksim|0.3474|
|3821B1FI2|zawadowski_jan|0.3533|
|3821B1FI1|kashirin_alexander|0.3647|
|3821B1FI3|kulikov_artem|0.3709|
|3821B1FI3|benduyzhko_tatiana|0.4267|
|3821B1FI3|tyulkina_olga|0.5634|
|3821B1FI3|polozov_vladislav|0.5856|
|3821B1FI3|sadikov_damir|0.6114|
|**REF**|**REF**|**0.6595**|
|3821B1FI3|simonyan_suren|0.9227|

## 4_jacobi_dev_oneapi (4096 elements)
|Group|Name|Result|
|-----|----|------|
|3821B1FI3|safronov_mikhail|0.1772|
|3821B1FI1|bonyuk_peter|0.1829|
|3821B1FI1|borovkov_sergey|0.1830|
|3821B1FI3|volodin_evgeniy|0.1848|
|3821B1FI1|shipitsin_alex|0.1855|
|3821B1FI1|shmelev_ivan|0.1874|
|3821B1FI2|nogin_denis|0.1921|
|3821B1FI3|simonyan_suren|0.1952|
|3821B1FI2|soloninko_andrey|0.1982|
|3821B1FI3|savotina_valeria|0.1986|
|3821B1FI3|kuznetsov_artyom|0.1987|
|3821B1FI1|balyasov_ivan|0.1988|
|3821B1FI2|zakharov_artem|0.1993|
|3821B1FI3|durandin_vladimir|0.1998|
|3821B1FI2|loginov_maxim|0.1999|
|3821B1FI2|petrov_maksim|0.2010|
|3821B1FI2|kostin_artem|0.2025|
|3821B1FI3|pivovarov_alexey|0.2033|
|3821B1FI2|kostanyan_arsen|0.2042|
|3821B1FI1|lysanova_julia|0.2052|
|3821B1FI3|sharapov_georgiy|0.2053|
|3821B1FI3|vasilev_ivan|0.2054|
|3821B1FI3|kulaev_zhenya|0.2069|
|3821B1FI3|safarov_nurlan|0.2079|
|3821B1FI1|chernyshev_nikita|0.2090|
|3821B1FI1|mirzakhmedov_alexander|0.2098|
|3821B1FI3|prokofev_kirill|0.2100|
|3821B1FI2|isaev_dmitrii|0.2148|
|3821B1FI3|ryabkov_vladislav|0.2151|
|3821B1FI3|kulagin_aleksandr|0.2270|
|3821B1FI3|kulikov_artem|0.2559|
|3821B1FI2|kazantsev_evgeny|0.2707|
|3821B1FI3|sadikov_damir|0.2827|
|3821B1FI3|polozov_vladislav|0.2890|
|3821B1FI3|korablev_nikita|0.3057|
|3821B1FI2|travin_maksim|0.3224|
|3821B1FI3|tyulkina_olga|0.3359|
|3821B1FI3|benduyzhko_tatiana|0.3370|
|3821B1FI1|bodrov_daniil|0.3388|
|3821B1FI2|zawadowski_jan|0.3598|
|3821B1FI3|ivanov_nikita|0.3635|
|3821B1FI1|kashirin_alexander|0.3994|
|**REF**|**REF**|**0.6662**|
|3821B1FI1|pushkarev_ilya|0.6729|
|3821B1FI2|kanakov_roman|BUILD FAILED|

## 5_jacobi_shared_oneapi (4096 elements)
|Group|Name|Result|
|-----|----|------|
|3821B1FI3|sharapov_georgiy|0.1545|
|3821B1FI2|zakharov_artem|0.1564|
|3821B1FI3|durandin_vladimir|0.1606|
|3821B1FI2|soloninko_andrey|0.1675|
|3821B1FI1|borovkov_sergey|0.1680|
|3821B1FI2|kostin_artem|0.1704|
|3821B1FI1|pushkarev_ilya|0.1750|
|3821B1FI1|shmelev_ivan|0.1770|
|3821B1FI3|kuznetsov_artyom|0.1792|
|3821B1FI3|tyulkina_olga|0.1798|
|3821B1FI3|safarov_nurlan|0.1873|
|3821B1FI3|safronov_mikhail|0.1876|
|3821B1FI3|benduyzhko_tatiana|0.1879|
|3821B1FI2|loginov_maxim|0.1883|
|3821B1FI3|savotina_valeria|0.1897|
|3821B1FI1|chernyshev_nikita|0.1898|
|3821B1FI1|lysanova_julia|0.1906|
|3821B1FI1|balyasov_ivan|0.1912|
|3821B1FI2|nogin_denis|0.1925|
|3821B1FI2|petrov_maksim|0.1926|
|3821B1FI3|pivovarov_alexey|0.1932|
|3821B1FI1|bonyuk_peter|0.1937|
|3821B1FI2|isaev_dmitrii|0.1963|
|3821B1FI3|prokofev_kirill|0.1971|
|3821B1FI3|vasilev_ivan|0.2001|
|3821B1FI3|ryabkov_vladislav|0.2002|
|3821B1FI1|shipitsin_alex|0.2007|
|3821B1FI3|kulaev_zhenya|0.2021|
|3821B1FI3|simonyan_suren|0.2047|
|3821B1FI3|kulagin_aleksandr|0.2055|
|3821B1FI1|mirzakhmedov_alexander|0.2076|
|3821B1FI3|korablev_nikita|0.2633|
|3821B1FI3|kulikov_artem|0.2676|
|3821B1FI3|ivanov_nikita|0.2710|
|3821B1FI3|polozov_vladislav|0.2711|
|3821B1FI2|kazantsev_evgeny|0.2721|
|3821B1FI3|sadikov_damir|0.2724|
|3821B1FI1|bodrov_daniil|0.3076|
|3821B1FI3|volodin_evgeniy|0.3227|
|3821B1FI2|travin_maksim|0.3601|
|3821B1FI2|zawadowski_jan|0.3608|
|3821B1FI1|kashirin_alexander|0.3695|
|**REF**|**REF**|**0.6341**|
|3821B1FI2|kostanyan_arsen|0.8133|
|3821B1FI2|kanakov_roman|BUILD FAILED|

## 6_gemm_block_oneapi (3072 elements)
|Group|Name|Result|
|-----|----|------|
|3821B1FI1|borovkov_sergey|0.8200|
|3821B1FI2|kostanyan_arsen|0.8220|
|3821B1FI3|tyulkina_olga|0.8278|
|3821B1FI3|simonyan_suren|0.8326|
|3821B1FI2|travin_maksim|0.8332|
|3821B1FI2|soloninko_andrey|0.8369|
|3821B1FI2|kazantsev_evgeny|0.8484|
|3821B1FI1|shipitsin_alex|0.8530|
|3821B1FI3|kulagin_aleksandr|0.8541|
|3821B1FI1|pushkarev_ilya|0.8616|
|**REF**|**REF**|**0.8759**|
|3821B1FI2|nogin_denis|0.8866|
|3821B1FI3|safronov_mikhail|0.8912|
|3821B1FI1|lysanova_julia|0.8928|
|3821B1FI1|shmelev_ivan|0.8931|
|3821B1FI2|loginov_maxim|0.8935|
|3821B1FI2|zakharov_artem|0.8939|
|3821B1FI3|ryabkov_vladislav|0.8944|
|3821B1FI3|benduyzhko_tatiana|0.8946|
|3821B1FI1|chernyshev_nikita|0.8964|
|3821B1FI3|kulikov_artem|0.8975|
|3821B1FI3|ivanov_nikita|0.8976|
|3821B1FI3|safarov_nurlan|0.8977|
|3821B1FI1|mirzakhmedov_alexander|0.8982|
|3821B1FI3|sharapov_georgiy|0.8987|
|3821B1FI3|durandin_vladimir|0.8999|
|3821B1FI3|kulaev_zhenya|0.9010|
|3821B1FI3|pivovarov_alexey|0.9012|
|3821B1FI1|balyasov_ivan|0.9016|
|3821B1FI2|isaev_dmitrii|0.9018|
|3821B1FI3|vasilev_ivan|0.9022|
|3821B1FI2|zawadowski_jan|0.9025|
|3821B1FI3|korablev_nikita|0.9030|
|3821B1FI3|savotina_valeria|0.9033|
|3821B1FI3|kuznetsov_artyom|0.9044|
|3821B1FI2|petrov_maksim|0.9142|
|3821B1FI3|polozov_vladislav|0.9244|
|3821B1FI2|kostin_artem|0.9246|
|3821B1FI3|volodin_evgeniy|0.9248|
|3821B1FI3|sadikov_damir|0.9324|
|3821B1FI3|prokofev_kirill|0.9383|
|3821B1FI1|kashirin_alexander|0.9719|
|3821B1FI1|bonyuk_peter|0.9923|
|3821B1FI1|bodrov_daniil|1.0162|
|3821B1FI2|kanakov_roman|BUILD FAILED|

## 7_gemm_mkl_oneapi (3072 elements)
|Group|Name|Result|
|-----|----|------|
|**REF**|**REF**|**0.2893**|
|3821B1FI3|sadikov_damir|0.4335|
|3821B1FI2|isaev_dmitrii|0.4872|
|3821B1FI2|kazantsev_evgeny|0.6009|
|3821B1FI2|nogin_denis|0.6340|
|3821B1FI2|loginov_maxim|0.6507|
|3821B1FI3|safronov_mikhail|0.6646|
|3821B1FI2|zawadowski_jan|0.6787|
|3821B1FI3|kuznetsov_artyom|0.6912|
|3821B1FI3|pivovarov_alexey|0.7002|
|3821B1FI3|tyulkina_olga|0.7044|
|3821B1FI2|soloninko_andrey|0.7344|
|3821B1FI3|simonyan_suren|0.7454|
|3821B1FI1|shipitsin_alex|0.7713|
|3821B1FI1|borovkov_sergey|0.7744|
|3821B1FI3|benduyzhko_tatiana|0.7849|
|3821B1FI1|bonyuk_peter|0.7913|
|3821B1FI3|kulaev_zhenya|0.7945|
|3821B1FI3|kulikov_artem|0.7959|
|3821B1FI2|travin_maksim|0.7961|
|3821B1FI1|pushkarev_ilya|0.7966|
|3821B1FI3|prokofev_kirill|0.8055|
|3821B1FI3|savotina_valeria|0.8079|
|3821B1FI1|kashirin_alexander|0.8137|
|3821B1FI3|sharapov_georgiy|0.8169|
|3821B1FI1|mirzakhmedov_alexander|0.8302|
|3821B1FI3|ivanov_nikita|0.8374|
|3821B1FI2|kostanyan_arsen|0.8474|
|3821B1FI3|volodin_evgeniy|0.8689|
|3821B1FI2|kostin_artem|0.9371|
|3821B1FI1|shmelev_ivan|0.9394|
|3821B1FI3|ryabkov_vladislav|0.9400|
|3821B1FI2|kanakov_roman|0.9404|
|3821B1FI3|durandin_vladimir|0.9522|
|3821B1FI2|zakharov_artem|0.9576|
|3821B1FI2|petrov_maksim|0.9633|
|3821B1FI3|kulagin_aleksandr|0.9653|
|3821B1FI3|vasilev_ivan|0.9656|
|3821B1FI1|lysanova_julia|0.9668|
|3821B1FI3|safarov_nurlan|0.9733|
|3821B1FI1|balyasov_ivan|0.9790|
|3821B1FI1|chernyshev_nikita|0.9790|
|3821B1FI3|korablev_nikita|0.9860|
|3821B1FI1|bodrov_daniil|0.9925|
|3821B1FI3|polozov_vladislav|1.0043|

# Tasks Done
## 3821B1FI1
|Group|Name|Passed|
|-----|----|------|
|3821B1FI1|balyasov_ivan|**7/7**|
|3821B1FI1|bodrov_daniil|**7/7**|
|3821B1FI1|bonyuk_peter|**7/7**|
|3821B1FI1|borovkov_sergey|**7/7**|
|3821B1FI1|chernyshev_nikita|**7/7**|
|3821B1FI1|kashirin_alexander|**7/7**|
|3821B1FI1|lysanova_julia|**7/7**|
|3821B1FI1|mirzakhmedov_alexander|**7/7**|
|3821B1FI1|pushkarev_ilya|**7/7**|
|3821B1FI1|shipitsin_alex|**7/7**|
|3821B1FI1|shmelev_ivan|**7/7**|

Passed: 11

## 3821B1FI2
|Group|Name|Passed|
|-----|----|------|
|3821B1FI2|isaev_dmitrii|**7/7**|
|3821B1FI2|kanakov_roman|3/7|
|3821B1FI2|kazantsev_evgeny|**7/7**|
|3821B1FI2|kostanyan_arsen|**7/7**|
|3821B1FI2|kostin_artem|**7/7**|
|3821B1FI2|loginov_maxim|**7/7**|
|3821B1FI2|nogin_denis|**7/7**|
|3821B1FI2|petrov_maksim|**7/7**|
|3821B1FI2|soloninko_andrey|**7/7**|
|3821B1FI2|travin_maksim|**7/7**|
|3821B1FI2|zakharov_artem|**7/7**|
|3821B1FI2|zawadowski_jan|**7/7**|

Passed: 11

## 3821B1FI3
|Group|Name|Passed|
|-----|----|------|
|3821B1FI3|benduyzhko_tatiana|**7/7**|
|3821B1FI3|durandin_vladimir|**7/7**|
|3821B1FI3|ivanov_nikita|**7/7**|
|3821B1FI3|korablev_nikita|**7/7**|
|3821B1FI3|kulaev_zhenya|**7/7**|
|3821B1FI3|kulagin_aleksandr|**7/7**|
|3821B1FI3|kulikov_artem|**7/7**|
|3821B1FI3|kuznetsov_artyom|**7/7**|
|3821B1FI3|pivovarov_alexey|**7/7**|
|3821B1FI3|polozov_vladislav|**7/7**|
|3821B1FI3|prokofev_kirill|**7/7**|
|3821B1FI3|ryabkov_vladislav|**7/7**|
|3821B1FI3|sadikov_damir|**7/7**|
|3821B1FI3|safarov_nurlan|**7/7**|
|3821B1FI3|safronov_mikhail|**7/7**|
|3821B1FI3|savotina_valeria|**7/7**|
|3821B1FI3|sharapov_georgiy|**7/7**|
|3821B1FI3|simonyan_suren|**7/7**|
|3821B1FI3|tyulkina_olga|**7/7**|
|3821B1FI3|vasilev_ivan|**7/7**|
|3821B1FI3|volodin_evgeniy|**7/7**|

Passed: 21

**Total Passed: 43**
