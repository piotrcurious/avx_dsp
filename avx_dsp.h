// A header file for the library
#ifndef AVX_DSP_H
#define AVX_DSP_H

#include <immintrin.h> // for AVX intrinsics

// A function that performs a dot product of two 256-bit vectors
__m256 avx_dot_product(__m256 x, __m256 y);

// A function that performs a convolution of two 256-bit vectors
__m256 avx_convolution(__m256 x, __m256 h);

// A function that performs a fast Fourier transform of a 256-bit vector
__m256 avx_fft(__m256 x);

#endif
