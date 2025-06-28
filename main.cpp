#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include "gpu_integral.h"
#include "cpu_integral.h"

int main(int argc, char** argv) {
    bool run_gpu = false;
    bool run_cpu = false;
    int n = 100, m = 100;

    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-g") == 0) run_gpu = true;
        else if (strcmp(argv[i], "-c") == 0) run_cpu = true;
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) m = atoi(argv[++i]);
    }

    float* result = new float[m];
    float* x = new float[m];
    for (int i = 0; i < m; ++i) x[i] = static_cast<float>(i + 1);

    if (run_gpu) {
        std::cout << "Running GPU version..." << std::endl;
        int n_m[2] = {n, m};
        auto start = std::chrono::high_resolution_clock::now();
        launchFloatKernel(n_m, result, x, m);
        cudaDeviceSynchronize();
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "GPU float time: " << duration.count() << "s" << std::endl;
    }

    if (run_cpu) {
        std::cout << "Running CPU version..." << std::endl;
        auto start = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < m; ++i) {
            result[i] = computeFloatEI(n, static_cast<float>(i + 1));
        }
        auto end = std::chrono::high_resolution_clock::now();
        std::chrono::duration<double> duration = end - start;
        std::cout << "CPU float time: " << duration.count() << "s" << std::endl;
    }

    delete[] result;
    delete[] x;
    return 0;
}
