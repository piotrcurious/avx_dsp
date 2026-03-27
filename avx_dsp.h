#ifndef AVX_DSP_H
#define AVX_DSP_H

#include <immintrin.h> // for AVX intrinsics
#include <stddef.h>

/**
 * @brief Aligned memory allocation for floats.
 *
 * @param size Number of floats to allocate.
 * @return float* Aligned pointer (32-byte alignment).
 */
float* avx_malloc(size_t size);

/**
 * @brief Free aligned memory.
 *
 * @param ptr Aligned pointer.
 */
void avx_free(float* ptr);

/**
 * @brief Performs a dot product of two 256-bit vectors (8 single-precision floats).
 *
 * Returns a __m256 vector where all elements contain the scalar dot product result.
 */
__m256 avx_dot_product(__m256 x, __m256 y);

/**
 * @brief Performs a dot product of two arrays of single-precision floats.
 *
 * @param a The first array (should be 32-byte aligned for best performance).
 * @param b The second array (should be 32-byte aligned for best performance).
 * @param size The number of elements in each array (must be a multiple of 8).
 */
float avx_dot_product_array(const float* a, const float* b, size_t size);

/**
 * @brief Performs a linear convolution of two arrays.
 *
 * @param x The input array of size x_size.
 * @param h The impulse response array of size h_size.
 * @param y The output array of size (x_size + h_size - 1).
 */
void avx_convolution_array(const float* x, size_t x_size, const float* h, size_t h_size, float* y);

/**
 * @brief Performs a cyclic convolution of two 256-bit vectors (8 single-precision floats).
 */
__m256 avx_convolution(__m256 x, __m256 h);

/**
 * @brief Performs a complex DFT of an array of floats (real, imag pairs).
 *
 * @param x Input complex array (2*size elements).
 * @param size Number of complex points.
 */
void avx_dft_array(const float* x, size_t size, float* out);

/**
 * @brief Performs a complex FFT of an array of floats (real, imag pairs).
 *
 * @param x Input/Output complex array (2*size elements).
 * @param n Number of complex points (must be a power of 2).
 */
void avx_fft_array(float* x, size_t n);

/**
 * @brief Performs a 4-point complex fast Fourier transform.
 *
 * Treats the 8 floats in the vector as 4 pairs of (real, imaginary) numbers.
 */
__m256 avx_fft(__m256 x);

#endif
