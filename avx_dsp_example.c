// An example C file that uses the avx_dsp library
#include <stdio.h>
#include "avx_dsp.h"

int main() {
    // Declare some variables
    __m256 x, y, z; // 256-bit vectors
    float a[8], b[8], c[8]; // arrays of floats
    int i; // loop index

    // Read 8 floats from stdin and store them in array a
    printf("Enter 8 floats for vector x:\n");
    for (i = 0; i < 8; i++) {
        scanf("%f", &a[i]);
    }

    // Read 8 floats from stdin and store them in array b
    printf("Enter 8 floats for vector y:\n");
    for (i = 0; i < 8; i++) {
        scanf("%f", &b[i]);
    }

    // Load arrays a and b into vectors x and y
    x = _mm256_loadu_ps(a);
    y = _mm256_loadu_ps(b);

    // Perform a dot product of x and y using the library function
    z = avx_dot_product(x, y);

    // Store the result in array c
    _mm256_storeu_ps(c, z);

    // Print the result
    printf("The dot product of x and y is:\n");
    for (i = 0; i < 8; i++) {
        printf("%f ", c[i]);
    }
    printf("\n");

    // Perform a convolution of x and y using the library function
    z = avx_convolution(x, y);

    // Store the result in array c
    _mm256_storeu_ps(c, z);

    // Print the result
    printf("The convolution of x and h is:\n");
    for (i = 0; i < 8; i++) {
        printf("%f ", c[i]);
    }
    printf("\n");

    // Perform a fast Fourier transform of x using the library function
    z = avx_fft(x);

    // Store the result in array c
    _mm256_storeu_ps(c, z);

    // Print the result
    printf("The fast Fourier transform of x is:\n");
    for (i = 0; i < 8; i++) {
        printf("%f ", c[i]);
    }
    printf("\n");

    return 0;
}
