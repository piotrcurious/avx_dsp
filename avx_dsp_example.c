#include <stdio.h>
#include <stdlib.h>
#include "avx_dsp.h"

void print_vector(const char* label, float* v, int n) {
    printf("%s: ", label);
    for (int i = 0; i < n; i++) printf("%.2f ", v[i]);
    printf("\n");
}

int main() {
    size_t size = 8;
    float *x = avx_malloc(size);
    float *y = avx_malloc(size);
    float *out = avx_malloc(size);

    for (int i = 0; i < size; i++) {
        x[i] = i + 1;
        y[i] = (i < 3) ? 1.0f : 0.0f;
    }

    print_vector("Input x", x, size);
    print_vector("Impulse h", y, 3);

    // FIR Filter
    avx_fir_filter(x, size, y, 3, out);
    print_vector("FIR Output", out, size);

    // Windowing
    avx_window_hann(x, size);
    print_vector("Hann Windowed x", x, size);

    // Dot product
    float dot = avx_dot_product_array(x, x, size);
    printf("Self dot product of windowed x: %.2f\n", dot);

    // Magnitude
    float mag_in[4] = {3, 4, 0, 5}; // (3+4j), (0+5j)
    float mag_out[2];
    avx_vector_magnitude(mag_in, 2, mag_out);
    printf("Magnitude: |3+4j| = %.1f, |0+5j| = %.1f\n", mag_out[0], mag_out[1]);

    // Complex Multiplication
    float ca[4] = {1, 0, 0, 1}; // 1 + 0j, 0 + 1j
    float cb[4] = {0, 1, 1, 0}; // 0 + 1j, 1 + 0j
    float cout[4];
    avx_complex_multiply_array(ca, cb, 2, cout);
    printf("Complex mult: (1+0j)*(0+1j) = %.1f+%.1fj, (0+1j)*(1+0j) = %.1f+%.1fj\n",
           cout[0], cout[1], cout[2], cout[3]);

    avx_free(x); avx_free(y); avx_free(out);
    return 0;
}
