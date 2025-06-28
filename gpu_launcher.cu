#include "gpu_integral.h"
#include <cuda_runtime.h>
#include <iostream>

// Declare kernels
__global__ void exponentialIntegralFloatKernel(float* results, int n, int m);
__global__ void exponentialIntegralDoubleKernel(double* results, int n, int m);

void launchFloatKernel(int* n_m, float* result, float* host_x, int size) {
    float* d_result;
    cudaMalloc(&d_result, size * sizeof(float));

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    exponentialIntegralFloatKernel<<<numBlocks, blockSize>>>(d_result, n_m[0], size);

    cudaMemcpy(result, d_result, size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}

void launchDoubleKernel(int* n_m, double* result, double* host_x, int size) {
    double* d_result;
    cudaMalloc(&d_result, size * sizeof(double));

    int blockSize = 256;
    int numBlocks = (size + blockSize - 1) / blockSize;
    exponentialIntegralDoubleKernel<<<numBlocks, blockSize>>>(d_result, n_m[0], size);

    cudaMemcpy(result, d_result, size * sizeof(double), cudaMemcpyDeviceToHost);
    cudaFree(d_result);
}
