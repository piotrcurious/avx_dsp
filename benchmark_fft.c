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

void benchmark_fft(size_t size) {
    float *x = avx_malloc(2 * size);
    float *out = avx_malloc(2 * size);
    for (size_t i = 0; i < 2 * size; i++) x[i] = (float)rand() / RAND_MAX;

    double start = get_time();
    for (int i = 0; i < 1000; i++) avx_fft_array(x, size);
    double on_the_fly_time = get_time() - start;

    avx_fft_ctx_t *ctx = avx_fft_init(size);
    start = get_time();
    for (int i = 0; i < 1000; i++) avx_fft_execute(ctx, x);
    double precomputed_time = get_time() - start;

    printf("FFT (%zu points): On-the-fly=%.6fs, Precomputed=%.6fs, Speedup=%.2fx\n",
           size, on_the_fly_time, precomputed_time, on_the_fly_time/precomputed_time);

    avx_fft_free(ctx);
    avx_free(x); avx_free(out);
}

int main() {
    benchmark_fft(1024);
    benchmark_fft(4096);
    return 0;
}
