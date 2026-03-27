#ifndef AVX_DSP_H
#define AVX_DSP_H

#include <immintrin.h> // for AVX intrinsics
#include <stddef.h>

/**
 * @brief Aligned memory allocation for floats.
 */
float* avx_malloc(size_t size);

/**
 * @brief Free aligned memory.
 */
void avx_free(float* ptr);

/**
 * @brief Performs a dot product of two 256-bit vectors (8 single-precision floats).
 */
__m256 avx_dot_product(__m256 x, __m256 y);

/**
 * @brief Performs a dot product of two arrays of single-precision floats.
 */
float avx_dot_product_array(const float* a, const float* b, size_t size);

/**
 * @brief Performs a linear convolution of two arrays.
 */
void avx_convolution_array(const float* x, size_t x_size, const float* h, size_t h_size, float* y);

/**
 * @brief Performs a cyclic convolution of two 256-bit vectors (8 single-precision floats).
 */
__m256 avx_convolution(__m256 x, __m256 h);

/**
 * @brief Performs a complex DFT of an array of floats (real, imag pairs).
 */
void avx_dft_array(const float* x, size_t size, float* out);

/**
 * @brief Performs a complex FFT of an array of floats (real, imag pairs).
 */
void avx_fft_array(float* x, size_t n);

/**
 * @brief Performs a 4-point complex fast Fourier transform.
 */
__m256 avx_fft(__m256 x);

/**
 * @brief Applies a Hann window to an array.
 */
void avx_window_hann(float* x, size_t n);

/**
 * @brief Applies a Hamming window to an array.
 */
void avx_window_hamming(float* x, size_t n);

/**
 * @brief Applies a Finite Impulse Response (FIR) filter to an array.
 *
 * @param x Input signal.
 * @param n Input signal size.
 * @param h Filter coefficients.
 * @param h_size Number of coefficients.
 * @param y Output signal (size n).
 */
void avx_fir_filter(const float* x, size_t n, const float* h, size_t h_size, float* y);

#endif
