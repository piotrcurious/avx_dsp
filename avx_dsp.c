#include "avx_dsp.h"
#include <math.h>
#include <string.h>
#include <stdlib.h>
#include <stdint.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

float* avx_malloc(size_t size) {
    void* ptr = NULL;
    if (posix_memalign(&ptr, 32, size * sizeof(float)) != 0) return NULL;
    return (float*)ptr;
}

void avx_free(float* ptr) {
    free(ptr);
}

// Helper to sum all 8 floats in __m256 and return a __m256 broadcast with the sum
static inline __m256 sum_m256(__m256 v) {
    __m128 low = _mm256_extractf128_ps(v, 0);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(low, high);

    __m128 hsum = _mm_hadd_ps(sum128, sum128);
    hsum = _mm_hadd_ps(hsum, hsum);

    return _mm256_insertf128_ps(_mm256_castps128_ps256(hsum), hsum, 1);
}

// Check if pointer is 32-byte aligned
#define IS_ALIGNED(p) (((uintptr_t)(p) & 31) == 0)

__m256 avx_dot_product(__m256 x, __m256 y) {
#ifdef __FMA__
    return sum_m256(_mm256_fmadd_ps(x, y, _mm256_setzero_ps()));
#else
    __m256 mul = _mm256_mul_ps(x, y);
    return sum_m256(mul);
#endif
}

float avx_dot_product_array(const float* a, const float* b, size_t size) {
    __m256 acc = _mm256_setzero_ps();
    size_t i = 0;
    int aligned = IS_ALIGNED(a) && IS_ALIGNED(b);

    for (; i + 7 < size; i += 8) {
        __m256 va = aligned ? _mm256_load_ps(a + i) : _mm256_loadu_ps(a + i);
        __m256 vb = aligned ? _mm256_load_ps(b + i) : _mm256_loadu_ps(b + i);
#ifdef __FMA__
        acc = _mm256_fmadd_ps(va, vb, acc);
#else
        acc = _mm256_add_ps(acc, _mm256_mul_ps(va, vb));
#endif
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

    int aligned_h = IS_ALIGNED(h);
    int aligned_y = IS_ALIGNED(y);

    for (size_t i = 0; i < x_size; i++) {
        size_t j = 0;
        __m256 vx = _mm256_set1_ps(x[i]);
        for (; j + 7 < h_size; j += 8) {
            __m256 vh = aligned_h ? _mm256_load_ps(h + j) : _mm256_loadu_ps(h + j);
            __m256 vy = (aligned_y && IS_ALIGNED(y + i + j)) ? _mm256_load_ps(y + i + j) : _mm256_loadu_ps(y + i + j);
#ifdef __FMA__
            vy = _mm256_fmadd_ps(vx, vh, vy);
#else
            vy = _mm256_add_ps(vy, _mm256_mul_ps(vx, vh));
#endif
            if (aligned_y && IS_ALIGNED(y + i + j)) _mm256_store_ps(y + i + j, vy);
            else _mm256_storeu_ps(y + i + j, vy);
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

// Radix-2 FFT Implementation
static void bit_reversal(float* x, size_t n) {
    size_t j = 0;
    for (size_t i = 0; i < n; i++) {
        if (i < j) {
            float temp_re = x[2*i];
            float temp_im = x[2*i+1];
            x[2*i] = x[2*j];
            x[2*i+1] = x[2*j+1];
            x[2*j] = temp_re;
            x[2*j+1] = temp_im;
        }
        size_t m = n >> 1;
        while (m >= 1 && j >= m) {
            j -= m;
            m >>= 1;
        }
        j += m;
    }
}

void avx_fft_array(float* x, size_t n) {
    bit_reversal(x, n);
    for (size_t s = 1; s <= (size_t)log2(n); s++) {
        size_t m = 1 << s;
        size_t m2 = m >> 1;
        for (size_t j = 0; j < m2; j++) {
            float w_re = cosf(-2.0f * (float)M_PI * j / m);
            float w_im = sinf(-2.0f * (float)M_PI * j / m);
            for (size_t k_idx = j; k_idx < n; k_idx += m) {
                float t_re = w_re * x[2*(k_idx + m2)] - w_im * x[2*(k_idx + m2) + 1];
                float t_im = w_re * x[2*(k_idx + m2) + 1] + w_im * x[2*(k_idx + m2)];
                float u_re = x[2*k_idx];
                float u_im = x[2*k_idx + 1];
                x[2*k_idx] = u_re + t_re;
                x[2*k_idx + 1] = u_im + t_im;
                x[2*(k_idx + m2)] = u_re - t_re;
                x[2*(k_idx + m2) + 1] = u_im - t_im;
            }
        }
    }
}

void avx_window_hann(float* x, size_t n) {
    size_t i = 0;
    int aligned = IS_ALIGNED(x);
    for (; i + 7 < n; i += 8) {
        float w[8];
        for (int j = 0; j < 8; j++) {
            w[j] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (i + j) / (n - 1)));
        }
        __m256 vw = _mm256_loadu_ps(w);
        __m256 vx = aligned ? _mm256_load_ps(x + i) : _mm256_loadu_ps(x + i);
        vx = _mm256_mul_ps(vx, vw);
        if (aligned) _mm256_store_ps(x + i, vx);
        else _mm256_storeu_ps(x + i, vx);
    }
    for (; i < n; i++) {
        x[i] *= 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (n - 1)));
    }
}

void avx_window_hamming(float* x, size_t n) {
    size_t i = 0;
    int aligned = IS_ALIGNED(x);
    for (; i + 7 < n; i += 8) {
        float w[8];
        for (int j = 0; j < 8; j++) {
            w[j] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * (i + j) / (n - 1));
        }
        __m256 vw = _mm256_loadu_ps(w);
        __m256 vx = aligned ? _mm256_load_ps(x + i) : _mm256_loadu_ps(x + i);
        vx = _mm256_mul_ps(vx, vw);
        if (aligned) _mm256_store_ps(x + i, vx);
        else _mm256_storeu_ps(x + i, vx);
    }
    for (; i < n; i++) {
        x[i] *= 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (n - 1));
    }
}

void avx_fir_filter(const float* x, size_t n, const float* h, size_t h_size, float* y) {
    for (size_t i = 0; i < n; i++) {
        y[i] = 0;
        size_t taps = (i < h_size) ? i + 1 : h_size;
        for (size_t j = 0; j < taps; j++) {
            y[i] += x[i - j] * h[j];
        }
    }
}
