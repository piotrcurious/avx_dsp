#ifndef AVX_DSP_H
#define AVX_DSP_H

#include <immintrin.h> // for AVX intrinsics

/**
 * @brief Performs a dot product of two 256-bit vectors (8 single-precision floats).
 *
 * Returns a __m256 vector where all elements contain the scalar dot product result.
 */
__m256 avx_dot_product(__m256 x, __m256 y);

/**
 * @brief Performs a cyclic convolution of two 256-bit vectors (8 single-precision floats).
 */
__m256 avx_convolution(__m256 x, __m256 h);

/**
 * @brief Performs a 4-point complex fast Fourier transform.
 *
 * Treats the 8 floats in the vector as 4 pairs of (real, imaginary) numbers.
 */
__m256 avx_fft(__m256 x);

#endif
