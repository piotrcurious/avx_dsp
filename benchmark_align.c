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

float scalar_dot_product(const float* a, const float* b, size_t size) {
    float sum = 0;
    for (size_t i = 0; i < size; i++) sum += a[i] * b[i];
    return sum;
}

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
    for (int i = 0; i < 100000; i++) dummy_res = scalar_dot_product(a, b, size);
    double scalar_time = get_time() - start;

    start = get_time();
    for (int i = 0; i < 100000; i++) dummy_res = avx_dot_product_array(a, b, size);
    double avx_time = get_time() - start;

    printf("Dot Product (Aligned, %zu elements): Scalar=%.6fs, AVX=%.6fs, Speedup=%.2fx\n", size, scalar_time, avx_time, scalar_time/avx_time);

    // Test unaligned
    float *a_un = malloc((size + 1) * sizeof(float));
    float *a_u = (float*)((uintptr_t)a_un | 4); // Misalign
    float *b_un = malloc((size + 1) * sizeof(float));
    float *b_u = (float*)((uintptr_t)b_un | 4);

    start = get_time();
    for (int i = 0; i < 100000; i++) dummy_res = avx_dot_product_array(a_u, b_u, size);
    double avx_un_time = get_time() - start;
    printf("Dot Product (Unaligned, %zu elements): AVX=%.6fs, Speedup over Scalar=%.2fx\n", size, avx_un_time, scalar_time/avx_un_time);

    avx_free(a); avx_free(b);
    free(a_un); free(b_un);
}

int main() {
    benchmark_dot_product(1024);
    return 0;
}
