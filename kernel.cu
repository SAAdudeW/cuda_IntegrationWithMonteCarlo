
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <cuda.h>
#include <curand_kernel.h>

    // определение параметров GPU
#define THREADS 128
#define BLOCKS 32
#define GENS 32*128

    // определение входных параметров
#define SEED 1234
#define N 1000 // максимальное число случайных точек на одном отрезке


    // проверка ошибок CUDA
#define cudaCheck(cudaf) { cudaCheckInner((cudaf), __FILE__, __LINE__); }
int cudaCheckInner(cudaError_t code, const char* file, int line) {

    if (code != cudaSuccess) {

        fprintf(stderr, "CUDA failed: %s %s %d\n", cudaGetErrorString(code), file, line);
        return 1;

    }
    else return 0;

}


    // инициализация генератора
__global__ void initfGENS(curandStatePhilox4_32_10_t* d_gen) {

    for (int id = threadIdx.x + blockIdx.x * blockDim.x; id < GENS; id += GENS) {
        curand_init(SEED, id, 0, d_gen + id);
    }

}


    // исследуемая функция
__device__ float f(float x) {
    return x*x;
}


    // параллельный Монте-Карло
/* прямоугольник, каждая сторона которого содержит точку графика
(вертикальные ассимптоты: x = id / R, x = (id + 1) / R)
(горизонтальные ассимптоты y = 0, y = max) */
/* бросаем N точек
K точек попало под график
integral = K / N * Sпрямоуг */
__global__ void parallel_monte_carlo_integrate(curandStatePhilox4_32_10_t* d_gen, int R, int n, float* d_result) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    for (; id < R; id += GENS) {

        float segmentMax = f(((float)id + (float)1) / R); // максимальное значение монотонно-возрастающей функции в отрезке
        int k = 0; // счетчик точек, попавших под график

        for (int i = 0; i < n; i++) {

            float x = ((float)curand(&d_gen [id]) / UINT_MAX + (float)id) / R; // генерация абсциссы в полосе
            float y = (float)curand(&d_gen [id]) * segmentMax / UINT_MAX;// генерация ординаты в полосе

            if (y <= f(x)) k++; // если точка под графиком, счетчик увеличивается
        }

        float square = segmentMax / R; // площадь прямоугольника в полосе
        float S = (float)k / n * square; // рассчет интеграла в полосе
        atomicAdd(&d_result[0], S); // рассчет интеграла на области определения
    }

}


int main() {

        // инициализация генератора
    curandStatePhilox4_32_10_t* d_gen;
    cudaCheck(cudaMalloc((void**) &d_gen, GENS * sizeof(curandStatePhilox4_32_10_t)));
    initfGENS << < BLOCKS, THREADS >> > (d_gen);
    cudaCheck(cudaGetLastError());

        // Монте-Карло CUDA begining
    int R; printf(" Enter number of integration lines: R = "); scanf("%d", &R);
    float ans = 0.3333; // известный ответ
    float integral[1];
    float* d_result; cudaCheck(cudaMalloc((void**) &d_result, 1 * sizeof(float)));

        // запись ошибок beginning
    FILE* out1 = fopen("output1.txt", "w");
    FILE* out2 = fopen("output2.txt", "w");
    if ((out1 != NULL) && (out2 != NULL)) {

        for (int n = 10; n <= N; n += 10) {

                // Монте-Карло CUDA continuation
            integral[0] = 0;
            cudaCheck(cudaMemset(d_result, 0, 1 * sizeof(float)));
            parallel_monte_carlo_integrate << <BLOCKS, THREADS >> > (d_gen, R, n, d_result);
            cudaCheck(cudaGetLastError());
            cudaCheck(cudaMemcpy(integral, d_result, 1 * sizeof(float), cudaMemcpyDeviceToHost));

                // запись ошибок continuation
            float mis1 = abs(integral [0] - ans);
            float mis2 = (integral[0] - ans) * (integral[0] - ans);
            fprintf(out1, "%f", mis1); fprintf(out1, " ");
            fprintf(out2, "%f", mis2); fprintf(out2, " ");

        }
            // Монте-Карло CUDA ending
        printf("Integral = %f", integral[0]);

            // запись ошибок ending
        fclose(out1);
        fclose(out2);
    }
    else printf("Не удалось открыть файл(ы)");

// cudaDeviceSynchronize waits for the kernel to finish, and returns
// any errors encountered during the launch.
    cudaCheck(cudaDeviceSynchronize());

        //освобождение памяти 
    cudaCheck(cudaFree(d_gen));
    cudaCheck(cudaFree(d_result));

    return 0;
}


/*
    // параллельный Монте-Карло
__global__ void parallel_monte_carlo_integrate(curandStatePhilox4_32_10_t* d_gen, int R, int n, float *d_result) {

    int id = threadIdx.x + blockIdx.x * blockDim.x;

    for (; id < R; id += GENS) {

        float segmentIntegral = 0;

        for (int i = 0; i < n; i++) {
            segmentIntegral += f((curand_uniform(&d_gen [id]) + id) / R) / (n * R); // интеграл по одному отрезку
            // segmentIntegral += f(((float)curand(&d_gen [id]) / UINT_MAX + id) / R) / (n * R);
        }

        atomicAdd(&d_result[0], segmentIntegral); // интеграл по области определения
    }

}
*/