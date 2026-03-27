#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "avx_dsp.h"

#define EPS 1e-4

void test_fft_large() {
    size_t size = 64;
    float *x = malloc(2 * size * sizeof(float));
    float *out_dft = malloc(2 * size * sizeof(float));
    float *out_fft = malloc(2 * size * sizeof(float));
    float *expected = malloc(2 * size * sizeof(float));

    FILE *fx = fopen("fft_x.bin", "rb");
    FILE *fe = fopen("fft_expected.bin", "rb");
    if (!fx || !fe) {
        printf("Error opening files for FFT test\n");
        exit(1);
    }
    fread(x, sizeof(float), 2 * size, fx);
    fread(expected, sizeof(float), 2 * size, fe);
    fclose(fx); fclose(fe);

    // Test DFT
    avx_dft_array(x, size, out_dft);
    for (size_t i = 0; i < 2 * size; i++) {
        assert(fabs(out_dft[i] - expected[i]) < EPS * 10);
    }
    printf("DFT test passed\n");

    // Test FFT
    for(size_t i=0; i<2*size; i++) out_fft[i] = x[i];
    avx_fft_array(out_fft, size);
    for (size_t i = 0; i < 2 * size; i++) {
        if (fabs(out_fft[i] - expected[i]) > EPS * 10) {
            printf("FFT mismatch at %zu: %f != %f\n", i, out_fft[i], expected[i]);
            exit(1);
        }
    }
    printf("FFT test passed\n");

    free(x); free(out_dft); free(out_fft); free(expected);
}

int main() {
    test_fft_large();
    return 0;
}
