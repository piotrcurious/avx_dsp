#ifndef AVX_DSP_H
#define AVX_DSP_H

#include <immintrin.h> // for AVX intrinsics
#include <stddef.h>

#ifdef __cplusplus
extern "C" {
#endif

float* avx_malloc(size_t size);
void avx_free(float* ptr);

__m256 avx_dot_product(__m256 x, __m256 y);
float avx_dot_product_array(const float* a, const float* b, size_t size);
void avx_convolution_array(const float* x, size_t x_size, const float* h, size_t h_size, float* y);
__m256 avx_convolution(__m256 x, __m256 h);
void avx_dft_array(const float* x, size_t size, float* out);
void avx_fft_array(float* x, size_t n);
__m256 avx_fft(__m256 x);

typedef struct {
    float *twiddles;
    size_t n;
} avx_fft_ctx_t;

/**
 * @brief Initialize FFT context with precomputed twiddle factors.
 * @param n FFT size (power of 2)
 * @return avx_fft_ctx_t* Context pointer
 */
avx_fft_ctx_t* avx_fft_init(size_t n);

/**
 * @brief Perform FFT using precomputed context.
 */
void avx_fft_execute(avx_fft_ctx_t *ctx, float *x);

/**
 * @brief Cleanup FFT context.
 */
void avx_fft_free(avx_fft_ctx_t *ctx);

void avx_window_hann(float* x, size_t n);
void avx_window_hamming(float* x, size_t n);
void avx_fir_filter(const float* x, size_t n, const float* h, size_t h_size, float* y);
void avx_complex_multiply_array(const float* a, const float* b, size_t n, float* out);
void avx_vector_magnitude(const float* x, size_t n, float* out);
void avx_vector_phase(const float* x, size_t n, float* out);

#ifdef __cplusplus
}
#endif

#endif
