#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "avx_dsp.h"

#define EPS 1e-4

void test_dot_product_large() {
    size_t size = 1024;
    float *a = malloc(size * sizeof(float));
    float *b = malloc(size * sizeof(float));
    float expected;

    FILE *fa = fopen("dot_a.bin", "rb");
    FILE *fb = fopen("dot_b.bin", "rb");
    FILE *fe = fopen("dot_expected.bin", "rb");
    if (!fa || !fb || !fe) {
        printf("Error opening files for dot product\n");
        exit(1);
    }
    if (fread(a, sizeof(float), size, fa) != size) printf("Read error a\n");
    if (fread(b, sizeof(float), size, fb) != size) printf("Read error b\n");
    if (fread(&expected, sizeof(float), 1, fe) != 1) printf("Read error expected\n");
    fclose(fa); fclose(fb); fclose(fe);

    float result = avx_dot_product_array(a, b, size);
    if (fabs(result - expected) > 1.0f) {
        printf("Dot product error: %f != %f\n", result, expected);
        exit(1);
    }

    free(a); free(b);
    printf("test_dot_product_large passed\n");
}

void test_convolution_large() {
    size_t x_size = 1024;
    size_t h_size = 8;
    size_t y_size = x_size + h_size - 1;
    float *x = malloc(x_size * sizeof(float));
    float *h = malloc(h_size * sizeof(float));
    float *y = malloc(y_size * sizeof(float));
    float *expected = malloc(y_size * sizeof(float));

    FILE *fx = fopen("conv_x.bin", "rb");
    FILE *fh = fopen("conv_h.bin", "rb");
    FILE *fe = fopen("conv_expected.bin", "rb");
    if (!fx || !fh || !fe) {
        printf("Error opening files for convolution\n");
        exit(1);
    }
    if (fread(x, sizeof(float), x_size, fx) != x_size) printf("Read error x\n");
    if (fread(h, sizeof(float), h_size, fh) != h_size) printf("Read error h\n");
    if (fread(expected, sizeof(float), y_size, fe) != y_size) printf("Read error expected\n");
    fclose(fx); fclose(fh); fclose(fe);

    avx_convolution_array(x, x_size, h, h_size, y);
    for (size_t i = 0; i < y_size; i++) {
        if (fabs(y[i] - expected[i]) > EPS) {
            printf("Conv error at %zu: %f != %f\n", i, y[i], expected[i]);
            exit(1);
        }
    }

    free(x); free(h); free(y); free(expected);
    printf("test_convolution_large passed\n");
}

void test_dft_large() {
    size_t size = 64;
    float *x = malloc(2 * size * sizeof(float));
    float *out = malloc(2 * size * sizeof(float));
    float *expected = malloc(2 * size * sizeof(float));

    FILE *fx = fopen("fft_x.bin", "rb");
    FILE *fe = fopen("fft_expected.bin", "rb");
    if (!fx || !fe) {
        printf("Error opening files for DFT\n");
        exit(1);
    }
    if (fread(x, sizeof(float), 2 * size, fx) != 2 * size) printf("Read error x\n");
    if (fread(expected, sizeof(float), 2 * size, fe) != 2 * size) printf("Read error expected\n");
    fclose(fx); fclose(fe);

    avx_dft_array(x, size, out);
    for (size_t i = 0; i < 2 * size; i++) {
        if (fabs(out[i] - expected[i]) > EPS * 10) {
            printf("DFT error at %zu: %f != %f\n", i, out[i], expected[i]);
            exit(1);
        }
    }

    free(x); free(out); free(expected);
    printf("test_dft_large passed\n");
}

int main() {
    test_dot_product_large();
    test_convolution_large();
    test_dft_large();
    return 0;
}
