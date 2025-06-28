#pragma once

#ifdef __cplusplus
extern "C" {
#endif

void launchFloatKernel(int* n_m, float* result, float* host_x, int size);
void launchDoubleKernel(int* n_m, double* result, double* host_x, int size);

#ifdef __cplusplus
}
#endif
