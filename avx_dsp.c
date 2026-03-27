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

static inline __m256 sum_m256(__m256 v) {
    __m128 low = _mm256_extractf128_ps(v, 0);
    __m128 high = _mm256_extractf128_ps(v, 1);
    __m128 sum128 = _mm_add_ps(low, high);
    __m128 hsum = _mm_hadd_ps(sum128, sum128);
    hsum = _mm_hadd_ps(hsum, hsum);
    return _mm256_insertf128_ps(_mm256_castps128_ps256(hsum), hsum, 1);
}

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
    for (; i < size; i++) sum += a[i] * b[i];
    return sum;
}

__m256 avx_convolution(__m256 x, __m256 h) {
    float h_vals[8];
    _mm256_storeu_ps(h_vals, h);
    float res[8];
    for (int i = 0; i < 8; i++) {
        float h_rot[8];
        for (int j = 0; j < 8; j++) h_rot[j] = h_vals[(i - j + 8) % 8];
        __m256 h_vec = _mm256_loadu_ps(h_rot);
        __m256 dot = avx_dot_product(x, h_vec);
        res[i] = _mm256_cvtss_f32(dot);
    }
    return _mm256_loadu_ps(res);
}

void avx_convolution_array(const float* x, size_t x_size, const float* h, size_t h_size, float* y) {
    for (size_t i = 0; i < x_size + h_size - 1; i++) y[i] = 0;
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
        for (; j < h_size; j++) y[i + j] += x[i] * h[j];
    }
}

__m256 avx_fft(__m256 x) {
    __m128 x_low = _mm256_extractf128_ps(x, 0);
    __m128 x_high = _mm256_extractf128_ps(x, 1);
    __m128 t_plus = _mm_add_ps(x_low, x_high);
    __m128 t_minus = _mm_sub_ps(x_low, x_high);
    float t_p[4], t_m[4];
    _mm_storeu_ps(t_p, t_plus);
    _mm_storeu_ps(t_m, t_minus);
    float res[8];
    res[0] = t_p[0] + t_p[2]; res[1] = t_p[1] + t_p[3];
    res[4] = t_p[0] - t_p[2]; res[5] = t_p[1] - t_p[3];
    res[2] = t_m[0] + t_m[3]; res[3] = t_m[1] - t_m[2];
    res[6] = t_m[0] - t_m[3]; res[7] = t_m[1] + t_m[2];
    return _mm256_loadu_ps(res);
}

void avx_dft_array(const float* x, size_t size, float* out) {
    for (size_t k = 0; k < size; k++) {
        float re = 0, im = 0;
        for (size_t n = 0; n < size; n++) {
            float angle = -2.0f * (float)M_PI * k * n / size;
            re += x[2*n] * cosf(angle) - x[2*n+1] * sinf(angle);
            im += x[2*n] * sinf(angle) + x[2*n+1] * cosf(angle);
        }
        out[2*k] = re;
        out[2*k+1] = im;
    }
}

static void bit_reversal(float* x, size_t n) {
    size_t j = 0;
    for (size_t i = 0; i < n; i++) {
        if (i < j) {
            float temp_re = x[2*i]; float temp_im = x[2*i+1];
            x[2*i] = x[2*j]; x[2*i+1] = x[2*j+1];
            x[2*j] = temp_re; x[2*j+1] = temp_im;
        }
        size_t m = n >> 1;
        while (m >= 1 && j >= m) { j -= m; m >>= 1; }
        j += m;
    }
}

void avx_fft_array(float* x, size_t n) {
    if ((n & (n - 1)) != 0 || n == 0) return;
    bit_reversal(x, n);
    for (size_t s = 1; s <= (size_t)log2(n); s++) {
        size_t m = 1 << s; size_t m2 = m >> 1;
        for (size_t j = 0; j < m2; j++) {
            float w_re = cosf(-2.0f * (float)M_PI * j / m);
            float w_im = sinf(-2.0f * (float)M_PI * j / m);
            for (size_t k_idx = j; k_idx < n; k_idx += m) {
                float t_re = w_re * x[2*(k_idx + m2)] - w_im * x[2*(k_idx + m2) + 1];
                float t_im = w_re * x[2*(k_idx + m2) + 1] + w_im * x[2*(k_idx + m2)];
                float u_re = x[2*k_idx]; float u_im = x[2*k_idx + 1];
                x[2*k_idx] = u_re + t_re; x[2*k_idx + 1] = u_im + t_im;
                x[2*(k_idx + m2)] = u_re - t_re; x[2*(k_idx + m2) + 1] = u_im - t_im;
            }
        }
    }
}

avx_fft_ctx_t* avx_fft_init(size_t n) {
    if ((n & (n - 1)) != 0 || n == 0) return NULL;
    avx_fft_ctx_t *ctx = (avx_fft_ctx_t*)malloc(sizeof(avx_fft_ctx_t));
    ctx->n = n;
    size_t num_twiddles = 0;
    for (size_t m = 2; m <= n; m <<= 1) num_twiddles += m;
    ctx->twiddles = avx_malloc(num_twiddles);
    size_t offset = 0;
    for (size_t m = 2; m <= n; m <<= 1) {
        size_t m2 = m >> 1;
        for (size_t j = 0; j < m2; j++) {
            ctx->twiddles[offset + 2 * j] = cosf(-2.0f * (float)M_PI * j / m);
            ctx->twiddles[offset + 2 * j + 1] = sinf(-2.0f * (float)M_PI * j / m);
        }
        offset += m;
    }
    return ctx;
}

void avx_fft_execute(avx_fft_ctx_t *ctx, float *x) {
    size_t n = ctx->n;
    bit_reversal(x, n);
    size_t offset = 0;
    for (size_t s = 1; s <= (size_t)log2(n); s++) {
        size_t m = 1 << s; size_t m2 = m >> 1;
        for (size_t j = 0; j < m2; j++) {
            float w_re = ctx->twiddles[offset + 2 * j];
            float w_im = ctx->twiddles[offset + 2 * j + 1];
            for (size_t k_idx = j; k_idx < n; k_idx += m) {
                float t_re = w_re * x[2*(k_idx + m2)] - w_im * x[2*(k_idx + m2) + 1];
                float t_im = w_re * x[2*(k_idx + m2) + 1] + w_im * x[2*(k_idx + m2)];
                float u_re = x[2*k_idx]; float u_im = x[2*k_idx + 1];
                x[2*k_idx] = u_re + t_re; x[2*k_idx + 1] = u_im + t_im;
                x[2*(k_idx + m2)] = u_re - t_re; x[2*(k_idx + m2) + 1] = u_im - t_im;
            }
        }
        offset += m;
    }
}

void avx_fft_free(avx_fft_ctx_t *ctx) {
    if (ctx) { avx_free(ctx->twiddles); free(ctx); }
}

void avx_window_hann(float* x, size_t n) {
    size_t i = 0; int aligned = IS_ALIGNED(x);
    for (; i + 7 < n; i += 8) {
        float w[8];
        for (int j = 0; j < 8; j++) w[j] = 0.5f * (1.0f - cosf(2.0f * (float)M_PI * (i + j) / (n - 1)));
        __m256 vw = _mm256_loadu_ps(w);
        __m256 vx = aligned ? _mm256_load_ps(x + i) : _mm256_loadu_ps(x + i);
        vx = _mm256_mul_ps(vx, vw);
        if (aligned) _mm256_store_ps(x + i, vx); else _mm256_storeu_ps(x + i, vx);
    }
    for (; i < n; i++) x[i] *= 0.5f * (1.0f - cosf(2.0f * (float)M_PI * i / (n - 1)));
}

void avx_window_hamming(float* x, size_t n) {
    size_t i = 0; int aligned = IS_ALIGNED(x);
    for (; i + 7 < n; i += 8) {
        float w[8];
        for (int j = 0; j < 8; j++) w[j] = 0.54f - 0.46f * cosf(2.0f * (float)M_PI * (i + j) / (n - 1));
        __m256 vw = _mm256_loadu_ps(w);
        __m256 vx = aligned ? _mm256_load_ps(x + i) : _mm256_loadu_ps(x + i);
        vx = _mm256_mul_ps(vx, vw);
        if (aligned) _mm256_store_ps(x + i, vx); else _mm256_storeu_ps(x + i, vx);
    }
    for (; i < n; i++) x[i] *= 0.54f - 0.46f * cosf(2.0f * (float)M_PI * i / (n - 1));
}

void avx_fir_filter(const float* x, size_t n, const float* h, size_t h_size, float* y) {
    // Blocked implementation for better cache locality and SIMD usage
    for (size_t i = 0; i < n; i++) y[i] = 0;

    const size_t block_size = 64; // Process signal in blocks
    for (size_t ib = 0; ib < n; ib += block_size) {
        size_t ie = (ib + block_size < n) ? ib + block_size : n;
        for (size_t j = 0; j < h_size; j++) {
            __m256 vh = _mm256_set1_ps(h[j]);
            size_t i = ib;
            // Handle startup edge where i - j < 0
            while (i < ie && (intptr_t)i - (intptr_t)j < 0) {
                i++;
            }
            for (; i + 7 < ie; i += 8) {
                __m256 vx = _mm256_loadu_ps(x + i - j);
                __m256 vy = _mm256_loadu_ps(y + i);
#ifdef __FMA__
                vy = _mm256_fmadd_ps(vx, vh, vy);
#else
                vy = _mm256_add_ps(vy, _mm256_mul_ps(vx, vh));
#endif
                _mm256_storeu_ps(y + i, vy);
            }
            for (; i < ie; i++) {
                if ((intptr_t)i - (intptr_t)j >= 0) y[i] += x[i - j] * h[j];
            }
        }
    }
}

void avx_complex_multiply_array(const float* a, const float* b, size_t n, float* out) {
    size_t i = 0; int aligned = IS_ALIGNED(a) && IS_ALIGNED(b) && IS_ALIGNED(out);
    for (; i + 3 < n; i += 4) {
        __m256 va = aligned ? _mm256_load_ps(a + 2 * i) : _mm256_loadu_ps(a + 2 * i);
        __m256 vb = aligned ? _mm256_load_ps(b + 2 * i) : _mm256_loadu_ps(b + 2 * i);
        __m256 va_r = _mm256_shuffle_ps(va, va, _MM_SHUFFLE(2, 2, 0, 0));
        __m256 va_i = _mm256_shuffle_ps(va, va, _MM_SHUFFLE(3, 3, 1, 1));
        __m256 vb_ir = _mm256_shuffle_ps(vb, vb, _MM_SHUFFLE(2, 3, 0, 1));
#ifdef __FMA__
        __m256 res = _mm256_mul_ps(va_r, vb);
        __m256 i_part = _mm256_mul_ps(va_i, vb_ir);
        __m256 sign = _mm256_setr_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);
        res = _mm256_fmadd_ps(i_part, sign, res);
#else
        __m256 res_r = _mm256_mul_ps(va_r, vb);
        __m256 res_i = _mm256_mul_ps(va_i, vb_ir);
        __m256 sign = _mm256_setr_ps(-1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f, -1.0f, 1.0f);
        __m256 res = _mm256_add_ps(res_r, _mm256_mul_ps(res_i, sign));
#endif
        if (aligned) _mm256_store_ps(out + 2 * i, res); else _mm256_storeu_ps(out + 2 * i, res);
    }
    for (; i < n; i++) {
        float ar = a[2*i], ai = a[2*i+1]; float br = b[2*i], bi = b[2*i+1];
        out[2*i] = ar * br - ai * bi; out[2*i+1] = ar * bi + ai * br;
    }
}

void avx_vector_magnitude(const float* x, size_t n, float* out) {
    size_t i = 0; int aligned_x = IS_ALIGNED(x); int aligned_out = IS_ALIGNED(out);
    for (; i + 3 < n; i += 4) {
        __m256 vx = aligned_x ? _mm256_load_ps(x + 2 * i) : _mm256_loadu_ps(x + 2 * i);
        __m256 sq = _mm256_mul_ps(vx, vx); __m256 sum = _mm256_hadd_ps(sq, sq);
        __m128 low = _mm256_extractf128_ps(sum, 0); __m128 high = _mm256_extractf128_ps(sum, 1);
        __m128 combined = _mm_shuffle_ps(low, high, _MM_SHUFFLE(1, 0, 1, 0)); __m128 res = _mm_sqrt_ps(combined);
        if (aligned_out) _mm_store_ps(out + i, res); else _mm_storeu_ps(out + i, res);
    }
    for (; i < n; i++) out[i] = sqrtf(x[2*i] * x[2*i] + x[2*i+1] * x[2*i+1]);
}

void avx_vector_phase(const float* x, size_t n, float* out) {
    for (size_t i = 0; i < n; i++) {
        out[i] = atan2f(x[2*i+1], x[2*i]);
    }
}
