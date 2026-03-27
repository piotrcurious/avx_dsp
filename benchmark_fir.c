#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <stdint.h>
#include "avx_dsp.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

volatile float dummy_res;

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void scalar_fir_filter(const float* x, size_t n, const float* h, size_t h_size, float* y) {
    for (size_t i = 0; i < n; i++) {
        y[i] = 0;
        size_t taps = (i < h_size) ? i + 1 : h_size;
        for (size_t j = 0; j < taps; j++) y[i] += x[i - j] * h[j];
    }
}

void benchmark_fir(size_t n, size_t h_size) {
    float *x = avx_malloc(n);
    float *h = avx_malloc(h_size);
    float *y = avx_malloc(n);
    for (size_t i = 0; i < n; i++) x[i] = 1.0f;
    for (size_t i = 0; i < h_size; i++) h[i] = 1.0f;

    double start = get_time();
    for (int i = 0; i < 1000; i++) scalar_fir_filter(x, n, h, h_size, y);
    double scalar_time = get_time() - start;

    start = get_time();
    for (int i = 0; i < 1000; i++) avx_fir_filter(x, n, h, h_size, y);
    double avx_time = get_time() - start;

    printf("FIR (%zu signal, %zu taps): Scalar=%.6fs, AVX=%.6fs, Speedup=%.2fx\n",
           n, h_size, scalar_time, avx_time, scalar_time/avx_time);

    avx_free(x); avx_free(h); avx_free(y);
}

int main() {
    benchmark_fir(4096, 128);
    return 0;
}
