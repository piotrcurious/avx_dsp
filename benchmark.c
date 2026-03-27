#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include "avx_dsp.h"

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Prevent optimization of scalar loop by adding a dummy global result
volatile float dummy_res;

float scalar_dot_product(const float* a, const float* b, size_t size) {
    float sum = 0;
    for (size_t i = 0; i < size; i++) sum += a[i] * b[i];
    return sum;
}

void scalar_convolution(const float* x, size_t x_size, const float* h, size_t h_size, float* y) {
    for (size_t i = 0; i < x_size + h_size - 1; i++) {
        y[i] = 0;
        size_t start = (i >= h_size - 1) ? i - (h_size - 1) : 0;
        size_t end = (i < x_size) ? i : x_size - 1;
        for (size_t j = start; j <= end; j++) y[i] += x[j] * h[i - j];
    }
}

double get_time() {
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return ts.tv_sec + ts.tv_nsec * 1e-9;
}

void benchmark_dot_product(size_t size) {
    float *a = malloc(size * sizeof(float));
    float *b = malloc(size * sizeof(float));
    for (size_t i = 0; i < size; i++) { a[i] = 1.0f; b[i] = 2.0f; }

    double start = get_time();
    for (int i = 0; i < 100000; i++) dummy_res = scalar_dot_product(a, b, size);
    double scalar_time = get_time() - start;

    start = get_time();
    for (int i = 0; i < 100000; i++) dummy_res = avx_dot_product_array(a, b, size);
    double avx_time = get_time() - start;

    printf("Dot Product (%zu elements): Scalar=%.6fs, AVX=%.6fs, Speedup=%.2fx\n", size, scalar_time, avx_time, scalar_time/avx_time);
    free(a); free(b);
}

void benchmark_convolution(size_t x_size, size_t h_size) {
    float *x = malloc(x_size * sizeof(float));
    float *h = malloc(h_size * sizeof(float));
    float *y = malloc((x_size + h_size - 1) * sizeof(float));
    for (size_t i = 0; i < x_size; i++) x[i] = 1.0f;
    for (size_t i = 0; i < h_size; i++) h[i] = 1.0f;

    double start = get_time();
    for (int i = 0; i < 100; i++) scalar_convolution(x, x_size, h, h_size, y);
    double scalar_time = get_time() - start;

    start = get_time();
    for (int i = 0; i < 100; i++) avx_convolution_array(x, x_size, h, h_size, y);
    double avx_time = get_time() - start;

    printf("Convolution (%zu x %zu): Scalar=%.6fs, AVX=%.6fs, Speedup=%.2fx\n", x_size, h_size, scalar_time, avx_time, scalar_time/avx_time);
    free(x); free(h); free(y);
}

void benchmark_fft(size_t size) {
    float *x = malloc(2 * size * sizeof(float));
    float *out = malloc(2 * size * sizeof(float));
    for (size_t i = 0; i < 2 * size; i++) x[i] = (float)rand() / RAND_MAX;

    double start = get_time();
    for (int i = 0; i < 100; i++) avx_dft_array(x, size, out);
    double dft_time = get_time() - start;

    start = get_time();
    for (int i = 0; i < 100; i++) avx_fft_array(x, size);
    double fft_time = get_time() - start;

    printf("FFT (%zu complex points): DFT=%.6fs, Radix-2 FFT=%.6fs, Speedup=%.2fx\n", size, dft_time, fft_time, dft_time/fft_time);
    free(x); free(out);
}

int main() {
    benchmark_dot_product(1024);
    benchmark_convolution(1024, 128);
    benchmark_fft(1024);
    return 0;
}
