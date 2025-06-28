#include "gpu_integral.h"
#include <cuda_runtime.h>

// Renamed to avoid conflict with CPU versions
__device__ float computeFloatEI_GPU(int n, float x) {
    float sum = 0.0f;
    float term = x;
    for (int k = 1; k <= 100; ++k) {
        term *= x / (n + k);
        sum += term;
    }
    return term + sum;
}

__device__ double computeDoubleEI_GPU(int n, double x) {
    double sum = 0.0;
    double term = x;
    for (int k = 1; k <= 100; ++k) {
        term *= x / (n + k);
        sum += term;
    }
    return term + sum;
}

__global__ void exponentialIntegralFloatKernel(float* results, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        float x = (float)(idx + 1);
        results[idx] = computeFloatEI_GPU(n, x);
    }
}

__global__ void exponentialIntegralDoubleKernel(double* results, int n, int m) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < m) {
        double x = (double)(idx + 1);
        results[idx] = computeDoubleEI_GPU(n, x);
    }
}

