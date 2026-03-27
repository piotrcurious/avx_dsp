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

void benchmark_dot_product(size_t size) {
    float *a = avx_malloc(size);
    float *b = avx_malloc(size);
    for (size_t i = 0; i < size; i++) { a[i] = 1.0f; b[i] = 2.0f; }

    double start = get_time();
    for (int i = 0; i < 100000; i++) {
        float sum = 0;
        for (size_t j = 0; j < size; j++) sum += a[j] * b[j];
        dummy_res = sum;
    }
    double scalar_time = get_time() - start;

    start = get_time();
    for (int i = 0; i < 100000; i++) dummy_res = avx_dot_product_array(a, b, size);
    double avx_time = get_time() - start;

    printf("Dot Product (%zu elements): Scalar=%.6fs, AVX=%.6fs, Speedup=%.2fx\n", size, scalar_time, avx_time, scalar_time/avx_time);
    avx_free(a); avx_free(b);
}

void benchmark_convolution(size_t x_size, size_t h_size) {
    float *x = avx_malloc(x_size);
    float *h = avx_malloc(h_size);
    float *y = avx_malloc(x_size + h_size - 1);
    for (size_t i = 0; i < x_size; i++) x[i] = 1.0f;
    for (size_t i = 0; i < h_size; i++) h[i] = 1.0f;

    double start = get_time();
    for (int i = 0; i < 100; i++) {
        for (size_t j = 0; j < x_size + h_size - 1; j++) {
            float sum = 0;
            size_t start_s = (j >= h_size - 1) ? j - (h_size - 1) : 0;
            size_t end_s = (j < x_size) ? j : x_size - 1;
            for (size_t k = start_s; k <= end_s; k++) sum += x[k] * h[j - k];
            y[j] = sum;
        }
    }
    double scalar_time = get_time() - start;

    start = get_time();
    for (int i = 0; i < 100; i++) avx_convolution_array(x, x_size, h, h_size, y);
    double avx_time = get_time() - start;

    printf("Convolution (%zu x %zu): Scalar=%.6fs, AVX=%.6fs, Speedup=%.2fx\n", x_size, h_size, scalar_time, avx_time, scalar_time/avx_time);
    avx_free(x); avx_free(h); avx_free(y);
}

void benchmark_fft(size_t size) {
    float *x = avx_malloc(2 * size);
    float *out = avx_malloc(2 * size);
    for (size_t i = 0; i < 2 * size; i++) x[i] = (float)rand() / RAND_MAX;

    double start = get_time();
    for (int i = 0; i < 100; i++) avx_dft_array(x, size, out);
    double dft_time = get_time() - start;

    start = get_time();
    for (int i = 0; i < 100; i++) avx_fft_array(x, size);
    double fft_time = get_time() - start;

    printf("FFT (%zu complex points): DFT=%.6fs, Radix-2 FFT=%.6fs, Speedup=%.2fx\n", size, dft_time, fft_time, dft_time/fft_time);
    avx_free(x); avx_free(out);
}

void benchmark_windowing(size_t size) {
    float *x = avx_malloc(size);
    for (size_t i = 0; i < size; i++) x[i] = 1.0f;

    double start = get_time();
    for (int i = 0; i < 10000; i++) {
        for (size_t j = 0; j < size; j++) {
            x[j] *= 0.5f * (1.0f - cosf(2.0f * (float)M_PI * j / (size - 1)));
        }
    }
    double scalar_time = get_time() - start;

    start = get_time();
    for (int i = 0; i < 10000; i++) avx_window_hann(x, size);
    double avx_time = get_time() - start;

    printf("Windowing (Hann, %zu elements): Scalar=%.6fs, AVX=%.6fs, Speedup=%.2fx\n", size, scalar_time, avx_time, scalar_time/avx_time);
    avx_free(x);
}

void benchmark_magnitude(size_t size) {
    float *x = avx_malloc(2 * size);
    float *mag = avx_malloc(size);
    for (size_t i = 0; i < 2 * size; i++) x[i] = 1.0f;

    double start = get_time();
    for (int i = 0; i < 10000; i++) {
        for (size_t j = 0; j < size; j++) {
            mag[j] = sqrtf(x[2*j]*x[2*j] + x[2*j+1]*x[2*j+1]);
        }
    }
    double scalar_time = get_time() - start;

    start = get_time();
    for (int i = 0; i < 10000; i++) avx_vector_magnitude(x, size, mag);
    double avx_time = get_time() - start;

    printf("Magnitude (%zu complex points): Scalar=%.6fs, AVX=%.6fs, Speedup=%.2fx\n", size, scalar_time, avx_time, scalar_time/avx_time);
    avx_free(x); avx_free(mag);
}

int main() {
    benchmark_dot_product(1024);
    benchmark_convolution(1024, 128);
    benchmark_fft(1024);
    benchmark_windowing(1024);
    benchmark_magnitude(1024);
    return 0;
}
