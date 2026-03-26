#include "avx_dsp.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper to sum all 8 floats in __m256 and return a __m256 broadcast with the sum
static inline __m256 sum_m256(__m256 v) {
    // [v0, v1, v2, v3 | v4, v5, v6, v7]
    __m256 hsum = _mm256_hadd_ps(v, v);
    // [v0+v1, v2+v3, v0+v1, v2+v3 | v4+v5, v6+v7, v4+v5, v6+v7]
    hsum = _mm256_hadd_ps(hsum, hsum);
    // [v0+v1+v2+v3, v0+v1+v2+v3, v0+v1+v2+v3, v0+v1+v2+v3 | v4+v5+v6+v7, v4+v5+v6+v7, v4+v5+v6+v7, v4+v5+v6+v7]

    __m128 low = _mm256_extractf128_ps(hsum, 0);
    __m128 high = _mm256_extractf128_ps(hsum, 1);
    __m128 total = _mm_add_ps(low, high);

    return _mm256_insertf128_ps(_mm256_castps128_ps256(total), total, 1);
}

__m256 avx_dot_product(__m256 x, __m256 y) {
    __m256 mul = _mm256_mul_ps(x, y);
    return sum_m256(mul);
}

// Optimized cyclic convolution using more AVX
__m256 avx_convolution(__m256 x, __m256 h) {
    float res[8];
    float h_vals[8];
    _mm256_storeu_ps(h_vals, h);

    // c[i] = sum_j x[j] * h[(i-j+8)%8]
    for (int i = 0; i < 8; i++) {
        // Rotate h to match indices for multiplication with x
        // For a given i, h[(i-j+8)%8] is needed.
        float h_rot[8];
        for (int j = 0; j < 8; j++) {
            h_rot[j] = h_vals[(i - j + 8) % 8];
        }
        __m256 h_vec = _mm256_loadu_ps(h_rot);
        __m256 dot = avx_dot_product(x, h_vec);
        res[i] = _mm256_cvtss_f32(dot);
    }

    return _mm256_loadu_ps(res);
}

// 4-point complex FFT using intrinsics where it makes sense
__m256 avx_fft(__m256 x) {
    // Input: [r0, i0, r1, i1, r2, i2, r3, i3]
    // 4-point DFT:
    // X0 = x0 + x1 + x2 + x3
    // X1 = x0 - j*x1 - x2 + j*x3
    // X2 = x0 - x1 + x2 - x3
    // X3 = x0 + j*x1 - x2 - j*x3

    float in[8];
    _mm256_storeu_ps(in, x);
    float out[8];

    for (int k = 0; k < 4; k++) {
        float re = 0, im = 0;
        for (int n = 0; n < 4; n++) {
            float angle = -2.0f * (float)M_PI * k * n / 4.0f;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);

            re += in[2*n] * cos_val - in[2*n+1] * sin_val;
            im += in[2*n] * sin_val + in[2*n+1] * cos_val;
        }
        out[2*k] = re;
        out[2*k+1] = im;
    }

    return _mm256_loadu_ps(out);
}
