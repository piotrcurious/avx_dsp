#include <stdio.h>
#include <assert.h>
#include <math.h>
#include "avx_dsp.h"

#define EPS 1e-5

void test_dot_product() {
    float a[8] = {1, 2, 3, 4, 5, 6, 7, 8};
    float b[8] = {1, 1, 1, 1, 1, 1, 1, 1};
    __m256 x = _mm256_loadu_ps(a);
    __m256 y = _mm256_loadu_ps(b);
    __m256 z = avx_dot_product(x, y);
    float c[8];
    _mm256_storeu_ps(c, z);
    for(int i=0; i<8; i++) {
        assert(fabs(c[i] - 36.0f) < EPS);
    }
    printf("test_dot_product passed\n");
}

void test_convolution() {
    float a[8] = {1, 2, 3, 4, 0, 0, 0, 0};
    float b[8] = {1, 0, 0, 0, 0, 0, 0, 0};
    __m256 x = _mm256_loadu_ps(a);
    __m256 y = _mm256_loadu_ps(b);
    __m256 z = avx_convolution(x, y);
    float c[8];
    _mm256_storeu_ps(c, z);
    assert(fabs(c[0] - 1.0f) < EPS);
    assert(fabs(c[1] - 2.0f) < EPS);
    assert(fabs(c[2] - 3.0f) < EPS);
    assert(fabs(c[3] - 4.0f) < EPS);
    printf("test_convolution passed\n");
}

void test_fft() {
    float a[8] = {1, 0, 1, 0, 1, 0, 1, 0};
    __m256 x = _mm256_loadu_ps(a);
    __m256 z = avx_fft(x);
    float c[8];
    _mm256_storeu_ps(c, z);
    // 4 points of (1,0) should result in (4,0) at k=0, and (0,0) at others
    assert(fabs(c[0] - 4.0f) < EPS);
    assert(fabs(c[1] - 0.0f) < EPS);
    assert(fabs(c[2] - 0.0f) < EPS);
    assert(fabs(c[3] - 0.0f) < EPS);
    assert(fabs(c[4] - 0.0f) < EPS);
    assert(fabs(c[5] - 0.0f) < EPS);
    assert(fabs(c[6] - 0.0f) < EPS);
    assert(fabs(c[7] - 0.0f) < EPS);
    printf("test_fft passed\n");
}

int main() {
    test_dot_product();
    test_convolution();
    test_fft();
    return 0;
}
