#include "cpu_integral.h"
#include <cmath>

float computeFloatEI(int n, float x) {
    float sum = 0.0f;
    float term = x;
    for (int k = 1; k <= 100; ++k) {
        term *= x / (n + k);
        sum += term;
    }
    return term + sum;
}

double computeDoubleEI(int n, double x) {
    double sum = 0.0;
    double term = x;
    for (int k = 1; k <= 100; ++k) {
        term *= x / (n + k);
        sum += term;
    }
    return term + sum;
}
