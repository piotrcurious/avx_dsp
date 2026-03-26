#include "avx_dsp.h"
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Helper to sum all 8 floats in __m256 and return a __m256 broadcast with the sum
static inline __m256 sum_m256(__m256 v) {
    __m128 low = _mm256_extractf128_ps(v, 0);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(low, high);

    __m128 hsum = _mm_hadd_ps(sum128, sum128);
    hsum = _mm_hadd_ps(hsum, hsum);

    return _mm256_insertf128_ps(_mm256_castps128_ps256(hsum), hsum, 1);
}

__m256 avx_dot_product(__m256 x, __m256 y) {
    __m256 mul = _mm256_mul_ps(x, y);
    return sum_m256(mul);
}

float avx_dot_product_array(const float* a, const float* b, size_t size) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    for (; i + 7 < size; i += 8) {
        __m256 va = _mm256_loadu_ps(a + i);
        __m256 vb = _mm256_loadu_ps(b + i);
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
    }

    __m256 hsum = sum_m256(acc);
    float sum = _mm256_cvtss_f32(hsum);

    for (; i < size; i++) {
        sum += a[i] * b[i];
    }

    return sum;
}

__m256 avx_convolution(__m256 x, __m256 h) {
    float res[8];
    float h_vals[8];
    _mm256_storeu_ps(h_vals, h);

    for (int i = 0; i < 8; i++) {
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

void avx_convolution_array(const float* x, size_t x_size, const float* h, size_t h_size, float* y) {
    for (size_t i = 0; i < x_size + h_size - 1; i++) {
        y[i] = 0;
    }

    for (size_t i = 0; i < x_size; i++) {
        size_t j = 0;
        __m256 vx = _mm256_set1_ps(x[i]);
        for (; j + 7 < h_size; j += 8) {
            __m256 vh = _mm256_loadu_ps(h + j);
            __m256 vy = _mm256_loadu_ps(y + i + j);
            vy = _mm256_add_ps(vy, _mm256_mul_ps(vx, vh));
            _mm256_storeu_ps(y + i + j, vy);
        }
        for (; j < h_size; j++) {
            y[i + j] += x[i] * h[j];
        }
    }
}

__m256 avx_fft(__m256 x) {
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

void avx_dft_array(const float* x, size_t size, float* out) {
    for (size_t k = 0; k < size; k++) {
        float re = 0, im = 0;
        for (size_t n = 0; n < size; n++) {
            float angle = -2.0f * (float)M_PI * k * n / size;
            float cos_val = cosf(angle);
            float sin_val = sinf(angle);
            re += x[2*n] * cos_val - x[2*n+1] * sin_val;
            im += x[2*n] * sin_val + x[2*n+1] * cos_val;
        }
        out[2*k] = re;
        out[2*k+1] = im;
    }
}
