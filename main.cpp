#include <iostream>
#include <cuda_runtime.h>
#include <cstdlib>
#include <cstring>
#include <chrono>
#include <cmath>
#include "gpu_integral.h"
#include "cpu_integral.h"

int main(int argc, char** argv) {
    bool run_cpu = false;
    int n = 100, m = 100;

    // Argument parsing
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "-c") == 0) run_cpu = true;
        else if (strcmp(argv[i], "-n") == 0 && i + 1 < argc) n = atoi(argv[++i]);
        else if (strcmp(argv[i], "-m") == 0 && i + 1 < argc) m = atoi(argv[++i]);
    }

    // Allocate host memory
    float* x_f = new float[m];
    float* cpu_result_f = new float[m];
    float* gpu_result_f = new float[m];

    double* x_d = new double[m];
    double* cpu_result_d = new double[m];
    double* gpu_result_d = new double[m];

    for (int i = 0; i < m; ++i) {
        x_f[i] = static_cast<float>(i + 1);
        x_d[i] = static_cast<double>(i + 1);
    }

    // GPU float
    std::cout << "Running GPU float..." << std::endl;
    auto gpu_start_f = std::chrono::high_resolution_clock::now();
    launchFloatKernel(&n, gpu_result_f, x_f, m);
    cudaDeviceSynchronize();
    auto gpu_end_f = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time_f = gpu_end_f - gpu_start_f;
    std::cout << "GPU float time: " << gpu_time_f.count() << "s" << std::endl;

    // GPU double
    std::cout << "Running GPU double..." << std::endl;
    auto gpu_start_d = std::chrono::high_resolution_clock::now();
    launchDoubleKernel(&n, gpu_result_d, x_d, m);
    cudaDeviceSynchronize();
    auto gpu_end_d = std::chrono::high_resolution_clock::now();
    std::chrono::duration<double> gpu_time_d = gpu_end_d - gpu_start_d;
    std::cout << "GPU double time: " << gpu_time_d.count() << "s" << std::endl;

    std::chrono::duration<double> cpu_time_f(0);
    std::chrono::duration<double> cpu_time_d(0);

    if (run_cpu) {
        std::cout << "Running CPU float..." << std::endl;
        auto cpu_start_f = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < m; ++i) {
            cpu_result_f[i] = computeFloatEI(n, x_f[i]);
        }
        auto cpu_end_f = std::chrono::high_resolution_clock::now();
        cpu_time_f = cpu_end_f - cpu_start_f;
        std::cout << "CPU float time: " << cpu_time_f.count() << "s" << std::endl;

        std::cout << "Running CPU double..." << std::endl;
        auto cpu_start_d = std::chrono::high_resolution_clock::now();
        for (int i = 0; i < m; ++i) {
            cpu_result_d[i] = computeDoubleEI(n, x_d[i]);
        }
        auto cpu_end_d = std::chrono::high_resolution_clock::now();
        cpu_time_d = cpu_end_d - cpu_start_d;
        std::cout << "CPU double time: " << cpu_time_d.count() << "s" << std::endl;

        // --- Numerical validation ---
        std::cout << "\nValidating float results..." << std::endl;
        int float_mismatches = 0;
        float max_error_f = 0.0f;
        for (int i = 0; i < m; ++i) {
            float diff = fabs(cpu_result_f[i] - gpu_result_f[i]);
            if (diff > 1e-5f) float_mismatches++;
            if (diff > max_error_f) max_error_f = diff;
        }
        std::cout << "Float mismatches: " << float_mismatches << ", Max error: " << max_error_f << std::endl;

        std::cout << "Validating double results..." << std::endl;
        int double_mismatches = 0;
        double max_error_d = 0.0;
        for (int i = 0; i < m; ++i) {
            double diff = fabs(cpu_result_d[i] - gpu_result_d[i]);
            if (diff > 1e-5) double_mismatches++;
            if (diff > max_error_d) max_error_d = diff;
        }
        std::cout << "Double mismatches: " << double_mismatches << ", Max error: " << max_error_d << std::endl;

        // --- Speedup ---
        std::cout << "\nSpeedup (float): " << cpu_time_f.count() / gpu_time_f.count() << "x" << std::endl;
        std::cout << "Speedup (double): " << cpu_time_d.count() / gpu_time_d.count() << "x" << std::endl;
    }

    // Cleanup
    delete[] x_f;
    delete[] cpu_result_f;
    delete[] gpu_result_f;
    delete[] x_d;
    delete[] cpu_result_d;
    delete[] gpu_result_d;

    return 0;
}

