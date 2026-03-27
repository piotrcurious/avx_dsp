#include <stdio.h>
#include <stdlib.h>
#include <assert.h>
#include <math.h>
#include "avx_dsp.h"

#define EPS 1e-3

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

void test_fft_large() {
    size_t size = 128;
    float *x = malloc(2 * size * sizeof(float));
    float *out = malloc(2 * size * sizeof(float));
    float *expected = malloc(2 * size * sizeof(float));

    FILE *fx = fopen("fft_x.bin", "rb");
    FILE *fe = fopen("fft_expected.bin", "rb");
    if (!fx || !fe) {
        printf("Error opening files for FFT\n");
        exit(1);
    }
    if (fread(x, sizeof(float), 2 * size, fx) != 2 * size) printf("Read error x\n");
    if (fread(expected, sizeof(float), 2 * size, fe) != 2 * size) printf("Read error expected\n");
    fclose(fx); fclose(fe);

    for(size_t i=0; i<2*size; i++) out[i] = x[i];
    avx_fft_array(out, size);
    for (size_t i = 0; i < 2 * size; i++) {
        if (fabs(out[i] - expected[i]) > 1.0f) {
            printf("FFT error at %zu: %f != %f\n", i, out[i], expected[i]);
            exit(1);
        }
    }

    free(x); free(out); free(expected);
    printf("test_fft_large passed\n");
}

void test_windowing_large() {
    size_t size = 1024;
    float *x = malloc(size * sizeof(float));
    float *expected_hann = malloc(size * sizeof(float));
    float *expected_hamming = malloc(size * sizeof(float));
    float *out = malloc(size * sizeof(float));

    FILE *fx = fopen("window_x.bin", "rb");
    FILE *fhann = fopen("window_hann_expected.bin", "rb");
    FILE *fhamm = fopen("window_hamming_expected.bin", "rb");
    if (!fx || !fhann || !fhamm) {
        printf("Error opening files for windowing\n");
        exit(1);
    }
    fread(x, sizeof(float), size, fx);
    fread(expected_hann, sizeof(float), size, fhann);
    fread(expected_hamming, sizeof(float), size, fhamm);
    fclose(fx); fclose(fhann); fclose(fhamm);

    for(size_t i=0; i<size; i++) out[i] = x[i];
    avx_window_hann(out, size);
    for(size_t i=0; i<size; i++) {
        if (fabs(out[i] - expected_hann[i]) > EPS) {
            printf("Hann error at %zu: %f != %f\n", i, out[i], expected_hann[i]);
            exit(1);
        }
    }

    for(size_t i=0; i<size; i++) out[i] = x[i];
    avx_window_hamming(out, size);
    for(size_t i=0; i<size; i++) {
        if (fabs(out[i] - expected_hamming[i]) > EPS) {
            printf("Hamming error at %zu: %f != %f\n", i, out[i], expected_hamming[i]);
            exit(1);
        }
    }

    free(x); free(expected_hann); free(expected_hamming); free(out);
    printf("test_windowing_large passed\n");
}

void test_fir_large() {
    size_t size = 1024;
    size_t h_size = 32;
    float *x = malloc(size * sizeof(float));
    float *h = malloc(h_size * sizeof(float));
    float *y = malloc(size * sizeof(float));
    float *expected = malloc(size * sizeof(float));

    FILE *fx = fopen("fir_x.bin", "rb");
    FILE *fh = fopen("fir_h.bin", "rb");
    FILE *fe = fopen("fir_expected.bin", "rb");
    if (!fx || !fh || !fe) {
        printf("Error opening files for FIR\n");
        exit(1);
    }
    fread(x, sizeof(float), size, fx);
    fread(h, sizeof(float), h_size, fh);
    fread(expected, sizeof(float), size, fe);
    fclose(fx); fclose(fh); fclose(fe);

    avx_fir_filter(x, size, h, h_size, y);
    for (size_t i = 0; i < size; i++) {
        if (fabs(y[i] - expected[i]) > EPS) {
            printf("FIR error at %zu: %f != %f\n", i, y[i], expected[i]);
            exit(1);
        }
    }

    free(x); free(h); free(y); free(expected);
    printf("test_fir_large passed\n");
}

void test_magnitude_large() {
    size_t size = 1024;
    float *x = malloc(2 * size * sizeof(float));
    float *mag = malloc(size * sizeof(float));
    float *expected = malloc(size * sizeof(float));

    FILE *fx = fopen("mag_x.bin", "rb");
    FILE *fe = fopen("mag_expected.bin", "rb");
    if (!fx || !fe) {
        printf("Error opening files for Magnitude\n");
        exit(1);
    }
    fread(x, sizeof(float), 2 * size, fx);
    fread(expected, sizeof(float), size, fe);
    fclose(fx); fclose(fe);

    avx_vector_magnitude(x, size, mag);
    for (size_t i = 0; i < size; i++) {
        if (fabs(mag[i] - expected[i]) > EPS) {
            printf("Magnitude error at %zu: %f != %f\n", i, mag[i], expected[i]);
            exit(1);
        }
    }

    free(x); free(mag); free(expected);
    printf("test_magnitude_large passed\n");
}

int main() {
    test_dot_product_large();
    test_convolution_large();
    test_fft_large();
    test_windowing_large();
    test_fir_large();
    test_magnitude_large();
    return 0;
}
